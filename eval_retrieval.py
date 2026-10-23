"""
检索效果评估：Top-1 命中率 / 低置信率对比（需求 1/2 效果验证）

对比项（同一样本集：扩容 KB 中 val+test 两个 split 的全部候选问法作为查询）：
  A. 基线 biencoder_cosine + 纯 Dense（改造前方案）
  B. 难负例微调 biencoder_hardneg + 纯 Dense
  C. biencoder_hardneg + 混合检索（BM25+RRF+CrossEncoder 精排，改造后生产方案）

指标：Top-1 命中率、Top-3 命中率、平均置信度、低置信率（<0.82 占比，阈值对 A/B 用
checkpoint 内最优阈值、对 C 用 CrossEncoder 概率阈值）

使用方式：
  python eval_retrieval.py
产物：outputs/logs/retrieval_eval.json
"""
# ↑ 模块级 docstring：用同一批查询对三套检索方案做 A/B/C 对比实验

import argparse                                   # CLI 参数
import json                                       # 读 KB、写报告
import sys                                        # 模块路径注入
import time                                       # C 方案建索引计时
from pathlib import Path                          # 路径处理

ROOT = Path(__file__).parent                      # 项目根目录
sys.path.insert(0, str(ROOT / "src"))             # 注入 src/ 到模块搜索路径

import torch  # noqa: E402                         # 设备与模型（E402：路径修改先于导入，告警忽略）

from hybrid_retriever import HybridRetriever           # noqa: E402  # 混合检索器（C 方案）
from onnx_runtime import load_dense_encoder  # noqa: E402  # 编码器后端工厂（A/B 方案）
from llm_synth import normalize                          # noqa: E402  # 文本归一化（leave-one-out 屏蔽用）

KB_PATH = ROOT / "data" / "blood_pressure_expanded" / "faq_extended.json"  # 扩容知识库
DATA_DIR = ROOT / "data" / "blood_pressure_expanded"  # split 数据目录
ONNX_DIR = ROOT / "outputs" / "onnx"               # ONNX 模型目录（C 方案线上同款）
BERT_PATH = r"F:\aaa123\pretrain_models\bert-base-chinese"  # 预训练权重路径
LOG_PATH = ROOT / "outputs" / "logs" / "retrieval_eval.json"  # 评估报告路径


def load_ce_score(ckpt_path, bert_path, device):
    """加载 CrossEncoder 并返回逐对打分函数 fn(queries, texts) -> list[float]（0~1 概率）。"""
    from model import build_crossencoder           # CE 模型工厂
    from transformers import BertTokenizer         # 分词器
    rckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)  # 加载 CE 权重
    m = build_crossencoder(bert_path=bert_path,
                           num_hidden_layers=rckpt.get("args", {}).get("num_hidden_layers")).to(device)  # 按训练超参重建模型
    m.load_state_dict(rckpt["state_dict"])         # 载入权重
    m.eval()                                       # 评估模式
    tok = BertTokenizer.from_pretrained(bert_path)  # 分词器

    @torch.no_grad()                               # 打分函数不建图
    def fn(queries, texts):
        # 契约与 HybridRetriever 一致：queries/texts 为等长列表，逐对打分
        enc = tok(list(zip(queries, texts)), max_length=128, truncation=True,
                  padding="max_length", return_tensors="pt")  # 句对列表直接 zip 给 tokenizer → 批量拼接编码
        logits = m(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                   enc["token_type_ids"].to(device))  # 批量前向 → [B, 2]
        return torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()  # softmax 取第 1 列 = 每对 P(相似)
    return fn                                      # 返回打分函数（供 HybridRetriever 注入）


def eval_dense(queries, labels, dense_enc, corpus_texts, corpus_fids, threshold):
    """纯 Dense：encode 全库句 + 查询，cosine Top-K

    leave-one-out：查询问法本身就在索引中，屏蔽归一化相同的语料列，避免 top1 泄漏。
    """
    import numpy as np                              # 矩阵运算
    q_vecs = np.asarray(dense_enc.encode(list(queries.keys())), dtype="float32")  # 全部查询向量 [Q, H]
    c_vecs = np.asarray(dense_enc.encode(corpus_texts), dtype="float32")  # 全部语料向量 [C, H]
    sims = q_vecs @ c_vecs.T                        # 一次矩阵乘得全部两两余弦 [Q, C]（向量已归一化）
    corpus_norms = [normalize(t) for t in corpus_texts]  # 预计算语料归一化文本
    top1_hit, top3_hit, low = 0, 0, 0               # 命中与低置信计数器
    top_sims = []                                   # 每条查询的最高相似度
    for i, q in enumerate(queries):                # 逐查询评估
        s = sims[i].copy()                          # 复制一行避免改原矩阵
        nq = normalize(q)                           # 查询的归一化键
        for j, nt in enumerate(corpus_norms):        # 扫描语料
            if nt == nq:                            # 语料里就是这句查询本身
                s[j] = -1e9                        # 屏蔽（leave-one-out 防自泄漏）
        order = np.argsort(-s)                      # 相似度降序下标
        top_sims.append(float(s[order[0]]))         # 记录 Top1 相似度（当作置信度）
        hit1 = corpus_fids[int(order[0])] == labels[q]  # Top1 的 FAQ id 是否等于真值
        top1_hit += hit1                            # 累计 Top1 命中
        top3_hit += any(corpus_fids[int(j)] == labels[q] for j in order[:3])  # Top3 内命中即算
        low += (float(s[int(order[0])]) < threshold)  # Top1 分低于阈值 → 低置信
    n = len(queries)                                # 查询总数
    return {"n": n, "top1": round(top1_hit / n, 4), "top3": round(top3_hit / n, 4),
            "low_conf_rate": round(low / n, 4),
            "avg_conf": round(float(np.mean(top_sims)), 4)}  # 返回四项检索指标


def eval_hybrid(queries, labels, retriever, threshold):
    top1_hit = top3_hit = low = 0                   # 计数器
    confs = []                                       # 每条查询的置信度
    for q in queries:                                # 逐查询走完整混合检索链路
        out = retriever.search(q, top_k=3, threshold=threshold, dense_threshold=threshold,
                               exclude_texts=q)  # leave-one-out：屏蔽查询问法自身  # 内部走 BM25+RRF+CE 精排，并自动屏蔽自身
        confs.append(out["confidence"])             # 收集置信度
        top1_hit += bool(out["results"]) and out["results"][0]["faq_id"] == labels[q]  # Top1 命中判定
        top3_hit += any(r["faq_id"] == labels[q] for r in out["results"])  # Top3 命中判定
        low += out["low_confidence"]                 # 低置信计数（拦截/拒答）
    n = len(queries)                                 # 查询总数
    return {"n": n, "top1": round(top1_hit / n, 4), "top3": round(top3_hit / n, 4),
            "low_conf_rate": round(low / n, 4),
            "avg_conf": round(sum(confs) / n, 4)}   # 返回四项检索指标（与 eval_dense 口径一致）


def build_queries():
    """查询集 = val+test split 的 FAQ 全部问法（标准问+变体），标签为 faq_id

    注意：只按 sentence1 推导所属 FAQ——负例行的 sentence2 来自其他 FAQ，
    若用 sentence2 反推会把训练集 FAQ 混入评估查询集造成泄漏。
    """
    faqs = json.loads(KB_PATH.read_text(encoding="utf-8"))  # 读扩容 KB
    text2fid = {t: f["id"] for f in faqs for t in [f["question"]] + f["variants"]}  # 文本 → FAQ id 映射
    eval_faq_ids = set()                             # 该进评估的 FAQ id 集合
    for split in ("validation", "test"):              # 只看 val/test 两个 split
        for line in (DATA_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():  # 逐行
            if line.strip():                         # 跳过空行
                r = json.loads(line)                  # 解析样本
                if r.get("faq_id"):                   # 行自带 faq_id 标记
                    eval_faq_ids.add(r["faq_id"])     # 直接采纳
                elif r["sentence1"] in text2fid:      # 否则用 sentence1 反查所属 FAQ
                    eval_faq_ids.add(text2fid[r["sentence1"]])  # （负例行 sentence2 属于别的 FAQ，不能反查）
    queries, labels = {}, {}                         # 查询字典与 标签字典
    for f in faqs:                                   # 遍历 KB 收集查询
        if f["id"] in eval_faq_ids:                   # 只收评估集 FAQ
            for t in [f["question"]] + f["variants"]:  # 标准问 + 全部变体
                queries[t] = None                   # dict 兼作有序去重集合
                labels[t] = f["id"]                  # 记录真值 faq_id
    corpus = [(f["question"], f["id"]) for f in faqs] + \
             [(v, f["id"]) for f in faqs for v in f["variants"]]  # 语料 = 全库文本与 id 对（含 draft，检索器内部会过滤）
    return queries, labels, corpus, faqs             # 返回评估全部要素


def main():
    """对比 A 基线纯 Dense / B 微调纯 Dense / C 微调混合+CE 三套方案的检索指标。"""
    args = parse_args()                              # 解析参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选设备
    queries, labels, corpus, faqs = build_queries()  # 构建查询/标签/语料
    print(f"评估查询数: {len(queries)}（val+test 的 {len(set(labels.values()))} 条 FAQ）")  # 打印评估规模

    full = {"n": len(queries), "corpus_size": len(corpus), "run": {}}  # 报告骨架
    corpus_texts = [t for t, _ in corpus]            # 语料文本列表
    corpus_fids = [fid for _, fid in corpus]         # 语料对应 faq_id 列表

    # ── A. 基线 cosine + 纯 Dense（改造前）───────────────────────────────
    base_enc, _ = load_dense_encoder(
        ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt", BERT_PATH,
        onnx_dir=None, device=device, prefer="onnx")  # 加载基线编码器（onnx_dir=None → 实际回退 torch 后端）
    ck_a = torch.load(ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt",
                      map_location="cpu", weights_only=False)  # 读基线 checkpoint（拿最优阈值）
    r_a = eval_dense(queries, labels, base_enc, corpus_texts, corpus_fids,
                     threshold=ck_a.get("threshold", 0.82))  # 用训练期搜出的阈值算低置信率
    full["run"]["A_baseline_dense"] = r_a            # 记入报告
    print(f"A. 基线 纯Dense        top1={r_a['top1']:.2%} top3={r_a['top3']:.2%} "
          f"low={r_a['low_conf_rate']:.2%}")         # 打印 A 方案指标

    # ── B. 难负例微调 + 纯 Dense ──────────────────────────────────────────
    hn_ckpt = ROOT / "outputs" / "checkpoints" / "biencoder_hardneg_best.pt"  # 微调权重路径
    hn_enc, _ = load_dense_encoder(hn_ckpt, BERT_PATH, onnx_dir=None,
                                   device=device, prefer="onnx")  # 加载微调编码器
    ck_b = torch.load(hn_ckpt, map_location="cpu", weights_only=False)  # 读微调 checkpoint
    if hn_enc is not None and hasattr(hn_enc, "encode"):  # 编码器加载成功才评
        r_b = eval_dense(queries, labels, hn_enc, corpus_texts, corpus_fids,
                         threshold=ck_b.get("threshold", 0.82))  # 同口径评估
        full["run"]["B_hardneg_dense"] = r_b         # 记入报告
        print(f"B. 微调 纯Dense        top1={r_b['top1']:.2%} top3={r_b['top3']:.2%} "
              f"low={r_b['low_conf_rate']:.2%}")     # 打印 B 方案指标

    # ── C. 难负例微调 + 混合检索（生产方案）───────────────────────────────
    t0 = time.time()                                 # 建链路计时开始
    ce = load_ce_score(ROOT / "outputs" / "checkpoints" / "crossencoder_best.pt",
                       BERT_PATH, device)            # 加载 CE 打分函数
    ret = HybridRetriever(faqs, dense_encode=load_dense_encoder(
        hn_ckpt, BERT_PATH, onnx_dir=ONNX_DIR, device=device, prefer="onnx")[0].encode,
        cross_score=ce)                              # 建混合检索器（dense 用线上同款 ONNX 后端）
    r_c = eval_hybrid(queries, labels, ret, threshold=args.threshold)  # 走完整链路评估
    r_c["index_build_s"] = round(time.time() - t0, 1)  # 记录建索引+评估耗时
    full["run"]["C_hardneg_hybrid_ce"] = r_c         # 记入报告
    full["run"]["C_retriever_stats"] = ret.stats()   # 附检索器运行统计（命中分布/延迟）
    print(f"C. 微调 混合+CE精排   top1={r_c['top1']:.2%} top3={r_c['top3']:.2%} "
          f"low={r_c['low_conf_rate']:.2%} avg_conf={r_c['avg_conf']}")  # 打印 C 方案指标

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
    LOG_PATH.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")  # 报告落盘
    print(f"\n报告 → {LOG_PATH}")                    # 提示产物位置
    if "r_a" in dir() and "r_c" in dir():           # A/C 两方案都有结果才给差值
        d = r_c["top1"] - r_a["top1"]               # Top-1 提升量
        print(f"Top-1 提升: {d:+.2%}（改造前 {r_a['top1']:.2%} → 改造后 {r_c['top1']:.2%}）")  # 打印改造收益


def parse_args():
    """解析检索评估命令行的各项参数。"""
    p = argparse.ArgumentParser()                    # 解析器
    p.add_argument("--threshold", default=0.82, type=float,
                   help="CrossEncoder 置信度阈值（C 方案）")  # C 方案判定阈值
    return p.parse_args()                           # 返回解析结果


if __name__ == "__main__":
    main()                                          # 直跑才执行评估
