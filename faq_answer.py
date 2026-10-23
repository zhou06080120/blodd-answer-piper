"""
血压计问答匹配主程序（混合检索两阶段 + 量化加速 + 低置信度拦截）

功能（本轮升级）：
  1. 默认走"混合检索两阶段"：Dense(S-BERT, 支持 ONNX-INT8 量化加速) + Sparse(BM25)
     双路召回 → RRF 融合 → CrossEncoder 精排（--rerank_ckpt）
  2. 置信度低于阈值（默认取 checkpoint 内最优阈值 0.82）时判定低置信度，
     自动异步写入待标注队列（data/feedback/queue.jsonl），供主动学习闭环消费
  3. --no-hybrid 退化为原始纯稠密检索路径（教学对比保留）
  4. --no-onnx 强制 PyTorch 后端（ONNX 缺失时也会自动回退，不会崩溃）

使用方式：
  python faq_answer.py --ckpt outputs/checkpoints/biencoder_hardneg_best.pt --query "血压机咋用？"
  python faq_answer.py --ckpt outputs/checkpoints/biencoder_cosine_best.pt --query "血压多少算正常？" --no-hybrid
"""
# ↑ 模块级 docstring：全项目的"用户入口"程序（命令行问答）。默认走混合检索两阶段
#   （Dense+BM25 双路召回 → RRF 融合 → CrossEncoder 精排），置信度不足自动拦截
#   并推送待标注队列；--no-hybrid 保留最初的纯 Dense 单路实现用于教学对比，
#   --no-onnx 可强制 PyTorch 后端（量化模型缺失时也会自动降级，不崩溃）。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 Windows 下 torch/ONNX 各带 OpenMP 重复加载导致的崩溃（必须在 import torch 前设置）
import argparse                                   # 命令行参数解析
import json                                       # 读取 FAQ 库 JSON
import sys                                         # 运行时模块路径注入
from pathlib import Path                           # 路径对象

import torch                                       # 张量与设备管理
import torch.nn.functional as F                    # 余弦相似度等函数
from transformers import BertTokenizer             # BERT 分词器

ROOT = Path(__file__).parent                       # 项目根目录
SRC = ROOT / "src"                                 # 源码目录
sys.path.insert(0, str(SRC))                      # 把 src/ 插到模块搜索路径最前，才能导入下方组件
from dataset import encode_single                      # noqa: E402  # 单句编码工具（改过 sys.path 后导入，故豁免 E402）
from model import build_biencoder, build_crossencoder  # noqa: E402  # 双塔/交叉编码器模型工厂

FAQ_PATH = ROOT / "data" / "blood_pressure" / "faq.json"       # 基础版知识库（30 条种子 FAQ）
KB_EXTENDED_PATH = ROOT / "data" / "blood_pressure_expanded" / "faq_extended.json"  # 扩容版知识库（LLM/模板扩到千级问法）
ONNX_DIR = ROOT / "outputs" / "onnx"               # ONNX 量化模型目录（存在则启用 INT8 加速）
MEDICAL_DISCLAIMER = "（本回答仅供健康教育参考，不能替代医生诊疗，如有不适请及时就医。）"  # 医疗免责声明（每轮回答末尾固定输出）

# ── 编码工具（保留原教学实现，供 --no-hybrid 路径使用）──────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# 函数：encode_text
# 功能：把单条文本编码成 1 个 L2 归一化句向量（纯 Dense 路径的最小编码单元）
# 流水线位置：建库（build_faq_index）与查询（retrieve）共用；@torch.no_grad()
#            关闭梯度记录，省显存、提速
# 输入：model/tokenizer/text/device/max_length　输出：[1, H] 句向量
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_text(model, tokenizer, text, device, max_length):
    enc = encode_single(tokenizer, text, max_length)  # 分词并转 ids/attention_mask/segment（返回 1D 张量）
    return model.encode(
        enc["input_ids"].unsqueeze(0).to(device),    # 升为 [1, L] 批维并搬到目标设备
        enc["attention_mask"].unsqueeze(0).to(device),  # 同上：注意力掩码
        enc["token_type_ids"].unsqueeze(0).to(device),  # 同上：句对分段（单句全 0）
    )                                                # 模型内部：BERT 前向 → 池化 → L2 归一化 → [1, H]


# ═══════════════════════════════════════════════════════════════════════════
# 函数：build_faq_index
# 功能：把 FAQ 库全部问法（标准问+同义改写）离线编码成"向量索引"矩阵
# 流水线位置：--no-hybrid 路径启动阶段执行一次；之后每次查询只需编码用户
#            问句、再与本索引算相似度，避免重复建库
# 输入：faqs=FAQ 列表等　输出：index=[N, H] 张量 + mapping=(问法, faq_id) 列表
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_faq_index(model, tokenizer, faqs, device, max_length):
    texts, mapping = [], []                          # texts=全部候选问法；mapping=与 texts 下标对齐的 (文本, faq_id)
    for f in faqs:                                   # 逐条 FAQ 展开
        for t in [f["question"]] + f["variants"]:   # 标准问 + 全部同义改写都入索引（提高召回面）
            texts.append(t)                          # 收集问法文本
            mapping.append((t, f["id"]))            # 记录"文本 ↔ FAQ 编号"对应关系
    index = torch.cat([encode_text(model, tokenizer, t, device, max_length)
                       for t in texts], dim=0)      # 逐句编码后沿 batch 维拼接 → [N, H] 索引矩阵
    return index, mapping                            # 返回向量索引与下标映射


# ═══════════════════════════════════════════════════════════════════════════
# 函数：retrieve
# 功能：纯 Dense 检索（--no-hybrid 教学对比路径）——查询向量与全库索引逐一算
#       余弦相似度，取 Top 候选并按 FAQ 去重；可选 CrossEncoder 对候选精排
# 输入：向量索引/映射/FAQ 查表/查询串/超参（可选 reranker 精排器）
# 输出：[{faq_id, question, answer, matched, similarity}] Top-k 结果列表
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def retrieve(model, tokenizer, index, mapping, faq_by_id, query, device,
             max_length, top_k=3, threshold=0.5, reranker=None, rerank_tokenizer=None):
    """纯 Dense 检索（--no-hybrid 教学对比路径，与原版一致）"""
    q_vec = encode_text(model, tokenizer, query, device, max_length)  # 查询句 → [1, H] 向量
    sims = F.cosine_similarity(q_vec, index, dim=-1)  # 查询与库内 N 个问法逐一算余弦 → [N] 相似度向量
    k = min(top_k * 3, index.size(0))                # 先取 3 倍候选给去重留余量，且不超过库容量
    top_idx = sims.topk(k).indices.tolist()         # 相似度最高的 k 个下标（降序排列）
    seen, candidates = set(), []                    # seen=已收录 FAQ 编号集合；candidates=去重后的最终候选
    for i in top_idx:                                # 按相似度从高到低遍历候选下标
        text, fid = mapping[i]                       # 取回该下标对应的问法与 FAQ 编号
        if fid in seen:                              # 同一 FAQ 的多个问法只保留最相似的一个
            continue                                 # 该 FAQ 已收录 → 跳过
        seen.add(fid)                                # 标记该 FAQ 已收录
        candidates.append((sims[i].item(), fid, text))  # 收录 (相似度, FAQ编号, 命中问法)
        if len(candidates) >= top_k:                 # 已凑齐 top_k 个不同 FAQ
            break                                   # 提前结束遍历
    if reranker is not None:                         # 传入了 CrossEncoder → 对粗排幸存者精排
        scored = []                                  # 精排打分列表
        for sim, fid, text in candidates:           # 只精排 top_k 条（控制算力成本）
            enc = rerank_tokenizer(query, text, max_length=128, truncation=True,
                                   padding="max_length", return_tensors="pt")  # (query, 候选) 拼成句对编码
            logits = reranker(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                              enc["token_type_ids"].to(device))  # 前向 → [1, 2] 两类 logits
            scored.append((torch.softmax(logits, dim=-1)[0, 1].item(), sim, fid, text))  # softmax 取"匹配类"概率作为精排分
        scored.sort(reverse=True)                    # 按精排分降序重排候选
        candidates = [(sim, fid, text) for _, sim, fid, text in scored]  # 抛弃精排分，恢复 (sim, fid, text) 结构
    results = []                                     # 组装最终返回结果
    for sim, fid, text in candidates:               # 逐条候选组装
        f = faq_by_id[fid]                           # 查 FAQ 元数据（取标准问/标准答）
        results.append({"faq_id": fid, "question": f["question"], "answer": f["answer"],
                        "matched": sim >= threshold, "similarity": sim})  # matched=是否过阈值；附原始相似度
    return results                                    # 返回 Top-k 结果


# ═══════════════════════════════════════════════════════════════════════════
# 函数：print_results
# 功能：统一的终端结果打印——兼容 hybrid/dense 两种输出结构，展示置信度、
#       命中标记、双路分数明细、低置信提示与医疗免责声明
# 输入：query=原查询串；out=两种模式各自的结果结构；mode="hybrid"|"dense"
# 输出：无（只打印）
# ═══════════════════════════════════════════════════════════════════════════
def print_results(query, out, mode):
    """统一打印：mode=hybrid 时读 retriever 输出；mode=dense 时读原格式"""
    print(f"\n用户提问：{query}")                      # 回显用户输入
    if mode == "hybrid":                              # 混合模式：out 是 retriever.search 的完整输出
        results, ret = out["results"], out            # 候选列表 + 全局元信息（置信度/阈值/拦截标记）
        print(f"置信度: {ret['confidence']:.4f} (阈值 {ret['threshold']:.2f})"
              f"{'  ⚠ 低置信度 → 已推送待标注队列' if ret['low_confidence'] else ''}")  # 打印综合置信度；低置信时追加警示
    else:                                             # 纯 Dense 模式：out 是结果列表，无全局元信息
        results, ret = out, {"confidence": None, "low_confidence": False}  # 用空元信息占位，统一后续取值逻辑
    if not results:                                   # FAQ 库为空的极端情况
        print("（FAQ 库为空）")                       # 直接提示并返回
        return
    print("=" * 60)                                   # 打印分隔线
    for i, r in enumerate(results, 1):               # 从 1 开始编号遍历候选
        if mode == "hybrid":                          # 混合模式：读 confidence + 双路分数
            flag = "✓ 命中" if ret["matched"] else "△ 低于阈值"  # 整体是否过阈值决定标记
            conf = r["confidence"]                    # 该候选的融合置信度
            extra = f"  dense={r['dense_sim']:.3f} bm25={r['bm25_score']:.2f}"  # 附展示 Dense/BM25 两路原始分数
        else:                                         # Dense 模式：读 similarity
            flag = "✓ 匹配" if r["matched"] else "△ 低于阈值"  # 单条是否过阈值
            conf, extra = r["similarity"], ""         # 余弦相似度即置信度；无附加分数
        print(f"[Top-{i}] {flag}  置信度={conf:.4f}  FAQ#{r['faq_id']}{extra}")  # 打印名次/标记/置信度/编号
        print(f"    标准问：{r['question']}")         # 该 FAQ 的标准问
        print(f"    回答  ：{r['answer']}")           # 该 FAQ 的标准答
    if mode == "dense" and not any(r["matched"] for r in results):  # Dense 模式下全部候选低于阈值
        print("\n提示：所有结果相似度均低于阈值，建议补充该问法到 FAQ 库后重新训练。")  # 给出运营改进建议
    if mode == "hybrid" and not ret["matched"]:       # 混合模式整体未过阈值（已被拦截）
        print("\n提示：置信度不足，已拦截并进入待标注队列；可换种问法或转人工。")  # 向用户解释拦截行为
    print(MEDICAL_DISCLAIMER)                         # 固定输出医疗免责声明


# ── 参数与主流程 ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# 函数：parse_args
# 功能：定义并解析问答主程序全部命令行参数（模型路径/检索开关/阈值等）
# 输入：无　输出：argparse.Namespace（main 中以 args.xxx 消费）
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    """解析问答主程序命令行的各项参数。"""
    parser = argparse.ArgumentParser(description="血压计问答匹配（混合检索版）")  # 参数解析器（--help 里的程序描述）
    parser.add_argument("--ckpt", required=True, type=str,
                        help="BiEncoder checkpoint 路径")  # 必填：双塔（Dense）模型权重文件
    parser.add_argument("--rerank_ckpt", default=str(ROOT / "outputs" / "checkpoints" /
                                                     "crossencoder_best.pt"), type=str,
                        help="CrossEncoder checkpoint；传 none 关闭精排")  # 精排权重，默认取产物目录；传字符串 none 显式关闭
    parser.add_argument("--bert_path", default=r"F:\aaa123\pretrain_models\bert-base-chinese",
                        type=str)                     # BERT 中文预训练基座目录
    parser.add_argument("--faq", default=None, type=str,
                        help="FAQ 库路径（默认自动优先 faq_extended.json）")  # 显式指定知识库；不传则自动优先扩容库
    parser.add_argument("--onnx_dir", default=str(ONNX_DIR), type=str,
                        help="ONNX 模型目录（不存在时自动回退 PyTorch）")  # ONNX 目录；缺失自动降级 PyTorch 不报错
    parser.add_argument("--max_length", default=64, type=int)  # 编码最大长度（一般沿用训练时的值）
    parser.add_argument("--top_k", default=3, type=int)        # 返回 Top-k 条候选
    parser.add_argument("--threshold", default=None, type=float,
                        help="置信度阈值（默认取 checkpoint 最优阈值）")  # 显式阈值；默认用训练时阈值搜索存下的最优值
    parser.add_argument("--query", default=None, type=str)    # 单次查询文本；不传则进入交互模式
    parser.add_argument("--no-hybrid", action="store_true", help="切回纯 Dense 检索")  # 布尔开关：走教学对比路径
    parser.add_argument("--no-onnx", action="store_true", help="强制 PyTorch 编码后端")  # 布尔开关：禁用 ONNX 加速
    parser.add_argument("--no-queue", action="store_true", help="低置信度不写入待标注队列")  # 布尔开关：关闭主动学习入队
    return parser.parse_args()                        # 解析并返回参数命名空间


# ═══════════════════════════════════════════════════════════════════════════
# 函数：main
# 功能：入口——加载 checkpoint 与知识库，按"混合检索两阶段（默认）/纯 Dense"
#       两种模式之一组装 answer 闭包，回答单条 --query 或进入交互问答循环
# 输入：无（读命令行参数）　输出：无（打印回答 / 写待标注队列）
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """入口：加载 KB/模型，按混合检索或纯 Dense 两种模式回答单条查询或进入交互问答。"""
    args = parse_args()                               # 解析命令行参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 GPU 用 GPU，否则 CPU

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)  # 加载 BiEncoder checkpoint（含权重+训练超参+最优阈值）
    saved = ckpt.get("args", {})                       # 训练时保存的超参（pool/层数/max_length 等）
    bert_path = args.bert_path                         # BERT 基座目录（以命令行为准）
    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 分词器
    threshold = args.threshold if args.threshold is not None else ckpt.get("threshold", 0.82)  # 阈值优先级：命令行 > checkpoint 内最优阈值 > 兜底 0.82
    max_length = saved.get("max_length", args.max_length)  # 编码长度与训练时保持一致（默认 64）

    # ── FAQ 库：自动优先扩容库 ────────────────────────────────────────────
    faq_path = Path(args.faq) if args.faq else None    # 显式指定则直接用指定的库
    if faq_path is None:                               # 未指定 → 自动选择
        faq_path = KB_EXTENDED_PATH if KB_EXTENDED_PATH.exists() else FAQ_PATH  # 有扩容库（千级问法）优先，否则退回 30 条种子库
    faqs = json.loads(faq_path.read_text(encoding="utf-8"))  # 读库文件并反序列化
    faq_by_id = {f["id"]: f for f in faqs}             # 编号 → FAQ 查表
    print(f"FAQ 库: {faq_path.name}（{len(faqs)} 条 FAQ）")  # 打印当前使用的库与规模

    if args.no_hybrid:                                # ── 教学路径：切回纯 Dense 检索 ──
        # ── 传统纯 Dense 路径 ─────────────────────────────────────────────
        model = build_biencoder(
            bert_path=bert_path, pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers")).to(device)  # 按 checkpoint 记录的结构重建双塔并搬上设备
        model.load_state_dict(ckpt["state_dict"])      # 载入训练权重
        model.eval()                                   # 切推理模式（关 dropout 等）
        reranker = rerank_tokenizer = None             # 先假定无精排
        if args.rerank_ckpt and args.rerank_ckpt.lower() != "none":  # 传了 CE 权重且未显式关闭
            rckpt = torch.load(args.rerank_ckpt, map_location=device, weights_only=False)  # 加载 CrossEncoder checkpoint
            reranker = build_crossencoder(
                bert_path=bert_path,
                num_hidden_layers=rckpt.get("args", {}).get("num_hidden_layers")).to(device)  # 按 CE 训练层数重建交叉编码器
            reranker.load_state_dict(rckpt["state_dict"])  # 载入 CE 权重
            reranker.eval()                            # 推理模式
            rerank_tokenizer = BertTokenizer.from_pretrained(bert_path)  # CE 分词器
            print("纯 Dense 模式 + CrossEncoder 精排")   # 提示当前生效配置
        index, mapping = build_faq_index(model, tokenizer, faqs, device, max_length)  # 全库问法一次性编码成索引

        def answer(query):                             # 纯 Dense 模式的单次问答闭包
            results = retrieve(model, tokenizer, index, mapping, faq_by_id, query,
                               device, max_length, top_k=args.top_k,
                               threshold=threshold, reranker=reranker,
                               rerank_tokenizer=rerank_tokenizer)  # 单路余弦检索（可带 CE 精排）
            print_results(query, results, mode="dense")  # 按旧格式打印结果
    else:                                              # ── 默认路径：混合检索两阶段 ──
        # ── 混合检索两阶段（默认）────────────────────────────────────────
        from onnx_runtime import load_dense_encoder   # 局部导入：Dense 编码后端工厂（支持 ONNX-INT8→PyTorch 降级链）
        from hybrid_retriever import HybridRetriever  # 局部导入：两阶段检索器（BM25+Dense 召回→RRF→精排）

        dense_enc, meta = load_dense_encoder(          # 构建 Dense 编码器（自动尝试 ONNX→PyTorch 降级）
            args.ckpt, bert_path,
            onnx_dir=None if args.no_onnx else args.onnx_dir,  # --no-onnx 时传 None 强制 PyTorch 后端
            max_length=max_length, prefer="onnx")      # 优先选 ONNX 后端
        print(f"Dense 编码后端: {meta['backend']}-{meta.get('kind') or meta.get('device', '?')}"
              f"  ({meta.get('onnx') or meta.get('device')})")  # 打印实际启用的后端/量化类型（int8/fp32/cpu）

        cross_score = None                             # 先声明精排打分函数为空（降级态）
        if args.rerank_ckpt and args.rerank_ckpt.lower() != "none" \
                and Path(args.rerank_ckpt).exists():   # 传了精排权重（非 none）且文件确实存在
            rckpt = torch.load(args.rerank_ckpt, map_location=device, weights_only=False)  # 加载 CE checkpoint
            reranker = build_crossencoder(
                bert_path=bert_path,
                num_hidden_layers=rckpt.get("args", {}).get("num_hidden_layers")).to(device)  # 按 CE 训练层数重建
            reranker.load_state_dict(rckpt["state_dict"])  # 载入 CE 权重
            reranker.eval()                            # 推理模式
            rerank_tokenizer = BertTokenizer.from_pretrained(bert_path)  # CE 句对分词器

            @torch.no_grad()                           # 精排打分闭包：无梯度推理
            def cross_score(queries, texts):           # 签名对齐 HybridRetriever 要求：批量 (query, text) → 概率列表
                enc = rerank_tokenizer(list(zip(queries, texts)), max_length=128,
                                       truncation=True, padding="max_length",
                                       return_tensors="pt")  # 逐对拼接编码成 batch
                logits = reranker(enc["input_ids"].to(device),
                                  enc["attention_mask"].to(device),
                                  enc["token_type_ids"].to(device))  # 前向 → [B, 2]
                return torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()  # softmax 取正类概率并转 Python list
            print("CrossEncoder 精排已加载")            # 精排就绪
        else:                                          # 无 CE 权重 → 降级打分
            print("未加载 CrossEncoder，使用 RRF+余弦组合分（阈值按 Dense 尺度）")  # 说明降级后的打分方式

        retriever = HybridRetriever(faqs, dense_encode=dense_enc.encode,
                                    cross_score=cross_score)  # 建两阶段检索器（内部完成 BM25 统计与全库 Dense 编码）
        print(f"混合检索索引建立完成，共 {retriever.n} 个候选问法。")  # 打印索引规模

        def answer(query):                             # 混合模式的单次问答闭包
            out = retriever.search(query, top_k=args.top_k, threshold=threshold,
                                   dense_threshold=threshold)  # 两阶段检索：双路召回→RRF 融合→精排→阈值判定
            out["threshold"] = threshold               # 回填阈值供打印使用
            # 低置信度 → 待标注队列（主动学习闭环）
            if out["low_confidence"] and not args.no_queue:  # 低置信且未显式关闭入队
                try:                                   # 入队属附加功能，失败不应影响主回答
                    from active_learning import AnnotationQueue  # 延迟导入标注队列
                    AnnotationQueue().append(          # 追加记录：原句 + 置信度 + Top 候选
                        query, out["confidence"],
                        [{"faq_id": r["faq_id"], "question": r["question"],
                          "confidence": r["confidence"]} for r in out["results"]])
                except Exception as e:                 # 任何写入异常都捕获
                    print(f"（队列写入失败，忽略: {e}）")  # 仅提示，不中断回答流程
            print_results(query, out, mode="hybrid")   # 按混合模式格式打印

    if args.query:                                     # 命令行带了单条查询
        answer(args.query)                             # 回答一次即退出
    else:                                              # 未指定查询 → 交互式问答
        print("\n进入交互式问答（输入 exit / quit 退出）：")  # 提示用法
        while True:                                    # 循环直到用户主动退出
            try:                                       # input 可能被管道关闭/Ctrl+Z 打断
                q = input("\n请输入血压相关问题：").strip()  # 读入一行并去掉首尾空白
            except EOFError:                           # 输入流关闭
                break                                  # 优雅退出循环
            if q.lower() in ("exit", "quit", "退出"):  # 命中退出关键词（大小写不敏感）
                break                                  # 结束会话
            if q:                                      # 跳过空行
                answer(q)                              # 正常回答当前问题


if __name__ == "__main__":                             # 直接运行本脚本时才执行（被导入时不触发）
    main()                                             # 调用主流程
