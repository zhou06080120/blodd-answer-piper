"""
BadCase 分析与优化方向建议

教学重点：
  1. FP（假阳性）vs FN（假阴性）的不同成因：
       FP：模型认为相似但实际不同 → 通常因为表面词汇高度重叠
       FN：模型认为不同但实际相似 → 通常因为换了表达方式
  2. 高置信度错误 vs 临界错误：错在边界和错在"确信"说明不同问题
  3. 从 bad case 出发推导优化方向（数据、模型、训练策略）
  4. 相似度分布中正负样本的重叠区 = 模型最难判断的区间

使用方式：
  # 分析 BiEncoder（CosineEmbeddingLoss）
  python analyze_badcases.py

  # 分析 CrossEncoder
  python analyze_badcases.py --model_type crossencoder \
    --ckpt ../outputs/checkpoints/crossencoder_best.pt

  # 分析更多 bad case
  python analyze_badcases.py --n_cases 20

依赖：
  pip install torch transformers scikit-learn matplotlib
"""
# ↑ 模块级 docstring：把模型"错在哪、为什么错、怎么改"做成可复现的分析报告

import os                                        # 环境变量
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 Windows OpenMP 冲突

import argparse                                   # CLI 参数
from pathlib import Path                          # 路径

import matplotlib                                 # 绘图
matplotlib.use("Agg")                             # 无界面后端
import matplotlib.pyplot as plt                    # 绘图 API
import numpy as np                                 # 统计计算
import torch                                      # 模型加载与推理
import torch.nn.functional as F                   # 余弦相似度
from torch.utils.data import DataLoader            # 批量加载
from transformers import BertTokenizer             # 分词器

from dataset import PairDataset, CrossEncoderDataset, load_jsonl  # 数据集与原始行读取（要拿原句展示）
from model import build_biencoder, build_crossencoder  # 模型工厂

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent           # 项目根
DATA_DIR  = ROOT / "data" / "blood_pressure"       # 数据目录
BERT_PATH = r"F:\aaa123\pretrain_models\bert-base-chinese"  # 预训练权重绝对路径
CKPT_DIR  = ROOT / "outputs" / "checkpoints"       # checkpoint 目录
FIG_DIR   = ROOT / "outputs" / "figures"           # 图表目录


# ── 收集预测结果 ──────────────────────────────────────────────────────────

@torch.no_grad()                                  # 推理不需要梯度
def collect_biencoder_preds(model, loader, raw_rows, device, threshold):
    """返回每条样本的 {sentence1, sentence2, label, sim, pred} 列表"""
    model.eval()                                  # 评估模式
    results = []                                  # 逐条预测记录（含原句文本）
    idx = 0                                       # 全局样本下标（与 raw_rows 对齐）
    for batch in loader:                          # 遍历 batch
        batch_a = {                                # 塔一输入
            "input_ids":      batch["input_ids_a"].to(device),       # token ids
            "attention_mask": batch["attention_mask_a"].to(device),  # 掩码
            "token_type_ids": batch["token_type_ids_a"].to(device),  # 类型
        }
        batch_b = {                                # 塔二输入
            "input_ids":      batch["input_ids_b"].to(device),       # token ids
            "attention_mask": batch["attention_mask_b"].to(device),  # 掩码
            "token_type_ids": batch["token_type_ids_b"].to(device),  # 类型
        }
        emb_a, emb_b = model(batch_a, batch_b)      # 双塔前向
        sims   = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()  # 逐对余弦
        labels = batch["label"].tolist()            # 本批标签

        for sim, label in zip(sims, labels):        # 逐条对齐（loader 未 shuffle → 顺序与 raw_rows 一致）
            row = raw_rows[idx]                     # 取对应原始行（拿原句展示用）
            results.append({                        # 记录完整信息
                "sentence1": row["sentence1"],      # 原句 1
                "sentence2": row["sentence2"],      # 原句 2
                "label": label,                     # 真实标签
                "score": sim,                       # 模型打分（余弦）
                "pred":  int(sim >= threshold),     # 按阈值二值化预测
            })
            idx += 1                                # 全局下标推进
    return results                                 # 返回全部预测记录


@torch.no_grad()                                  # 推理无梯度
def collect_crossencoder_preds(model, loader, raw_rows, device):
    model.eval()                                   # 评估模式
    results = []                                   # 预测记录
    idx = 0                                        # 全局样本下标
    for batch in loader:                           # 遍历 batch
        logits = model(                            # 前向 → [B, 2]
            batch["input_ids"].to(device),         # token ids
            batch["attention_mask"].to(device),    # 掩码
            batch["token_type_ids"].to(device),    # 句界标记
        ).cpu()                                    # 搬回 CPU 便于转 list
        probs  = torch.softmax(logits, dim=-1)[:, 1].tolist()  # softmax 后取第 1 列 = P(相似)
        preds  = logits.argmax(dim=-1).tolist()    # argmax 预测类别
        labels = batch["label"].tolist()           # 真实标签

        for prob, pred, label in zip(probs, preds, labels):  # 逐条组装
            row = raw_rows[idx]                     # 对齐原始行
            results.append({                        # 记录
                "sentence1": row["sentence1"],      # 原句 1
                "sentence2": row["sentence2"],      # 原句 2
                "label": label,                     # 真实标签
                "score": prob,   # P(相似)          # 分数含义是相似概率（与 BiEncoder 的余弦统一成 score 字段）
                "pred":  pred,                      # 预测类别
            })
            idx += 1                                # 下标推进
    return results                                 # 返回全部记录


# ── Bad Case 分类 ─────────────────────────────────────────────────────────

def split_badcases(results, threshold=0.5):
    """
    分类 bad case：
      FP（假阳性）：pred=1, label=0  → 模型过度自信"相似"
      FN（假阴性）：pred=0, label=1  → 模型错过真实相似对

    同时按置信度分级：
      high confidence error : score 远离 threshold（偏差 > 0.15）
      borderline error      : score 接近 threshold（偏差 ≤ 0.15）
    """
    fp_high, fp_border = [], []                    # FP 的高置信/临界两个桶
    fn_high, fn_border = [], []                    # FN 的两个桶

    for r in results:                              # 遍历全部预测
        if r["pred"] == r["label"]:                # 预测正确
            continue                              # 不进 bad case
        gap = abs(r["score"] - threshold)           # 分数离阈值多远（置信度代理）
        if r["pred"] == 1 and r["label"] == 0:   # FP  # 模型判相似但实际不相似
            (fp_high if gap > 0.15 else fp_border).append(r)  # 偏差>0.15 属高置信错误，否则临界
        else:                                      # FN  # 模型判不相似但实际相似
            (fn_high if gap > 0.15 else fn_border).append(r)  # 同上分级

    # 按置信度排序（最错的在前）
    fp_high.sort(key=lambda x: x["score"], reverse=True)  # FP：分数越高错得越离谱 → 降序
    fn_high.sort(key=lambda x: x["score"])                # FN：分数越低错得越离谱 → 升序

    return {                                       # 四桶打包返回
        "fp_high":   fp_high,                      # 高置信 FP
        "fp_border": fp_border,                    # 临界 FP
        "fn_high":   fn_high,                      # 高置信 FN
        "fn_border": fn_border,                    # 临界 FN
    }


# ── 模式分析 ──────────────────────────────────────────────────────────────

def analyze_patterns(cases, label):
    """对一批 bad case 做词汇重叠、长度差等简单统计"""
    if not cases:                                  # 没有该类错误
        return                                    # 直接返回
    len_diffs  = [abs(len(r["sentence1"]) - len(r["sentence2"])) for r in cases]  # 两句长度差
    lens_s1    = [len(r["sentence1"]) for r in cases]  # 句 1 长度列表
    lens_s2    = [len(r["sentence2"]) for r in cases]  # 句 2 长度列表

    # 字符级 Jaccard 相似度（用 unigram 集合）
    def jaccard(a, b):                             # 内部工具函数
        sa, sb = set(a), set(b)                     # 两句的字符集合
        return len(sa & sb) / max(len(sa | sb), 1)  # 交集/并集（max 防除零）

    jaccards = [jaccard(r["sentence1"], r["sentence2"]) for r in cases]  # 全部句对的 Jaccard

    print(f"\n  【{label}】共 {len(cases)} 条")        # 分组标题与规模
    print(f"    长度差     : 均值={np.mean(len_diffs):.1f}  中位={np.median(len_diffs):.0f}")  # 长度差统计
    print(f"    s1 长度    : 均值={np.mean(lens_s1):.1f}")   # 句 1 平均长度
    print(f"    s2 长度    : 均值={np.mean(lens_s2):.1f}")   # 句 2 平均长度
    print(f"    字符 Jaccard: 均值={np.mean(jaccards):.3f}  "
          f"（1=完全重叠，0=无共同字符）")            # 字符重叠度（词面相近程度）


def print_cases(cases, title, n=5):
    print(f"\n  {title} (展示 {min(n, len(cases))} 条)：")  # 标题（实际展示数不超过存量）
    for r in cases[:n]:                            # 展示前 n 条
        score_str = f"score={r['score']:.3f}"       # 分数字符串
        print(f"    {score_str}  | {r['sentence1']!r}")  # 打印分数与句 1
        print(f"    {'':>12}  | {r['sentence2']!r}")   # 对齐缩进打印句 2
        print()                                    # 空行分隔每条案例


# ── Score 分布可视化 ──────────────────────────────────────────────────────

def plot_score_dist_with_errors(results, threshold, save_path, model_label):
    """
    正确 vs 错误预测的分数分布——观察错误集中在哪个 score 区间
    """
    scores  = np.array([r["score"]  for r in results])  # 全部分数
    labels  = np.array([r["label"]  for r in results])  # 全部标签
    correct = np.array([r["pred"] == r["label"] for r in results])  # 是否预测正确（布尔）

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 左右两联图

    # 左图：正/负样本分数分布（同 evaluate.py，基准参考）
    ax = axes[0]                                    # 左轴
    ax.hist(scores[labels==1], bins=40, alpha=0.6, label="positive", color="#2196F3", density=True)  # 正样本分数分布
    ax.hist(scores[labels==0], bins=40, alpha=0.6, label="negative", color="#F44336", density=True)  # 负样本分数分布
    if threshold is not None:                        # 有阈值就画线
        ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")  # 阈值竖线
    ax.set_xlabel("Score (cosine sim / P(similar))")  # x 轴（两种模型的分数含义注明）
    ax.set_title(f"[{model_label}] Positive vs Negative Distribution")  # 标题
    ax.legend(fontsize=8)                            # 图例

    # 右图：正确 vs 错误预测的分数分布
    ax = axes[1]                                    # 右轴
    ax.hist(scores[correct],  bins=40, alpha=0.6, label="correct", color="#4CAF50", density=True)  # 预测正确的分数分布（绿）
    ax.hist(scores[~correct], bins=40, alpha=0.6, label="error",   color="#F44336", density=True)  # 预测错误的分数分布（红）
    if threshold is not None:                        # 有阈值
        ax.axvline(threshold, color="black", linestyle="--")   # 同样画阈值线
    ax.set_xlabel("Score")                           # x 轴
    ax.set_title(f"[{model_label}] Correct vs Error Score Distribution")  # 标题
    ax.legend(fontsize=8)                             # 图例

    fig.tight_layout()                                # 布局
    save_path = Path(save_path)                      # 包装路径
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 建目录
    fig.savefig(save_path, dpi=150)                  # 存图
    plt.close(fig)                                   # 关画布
    print(f"  图表已保存 → {save_path}")               # 提示


# ── 优化方向输出 ──────────────────────────────────────────────────────────

def print_optimization_directions(badcases, model_type, fp_jaccard_mean, fn_jaccard_mean):
    """
    根据 bad case 的实际模式，输出有针对性的优化方向。
    """
    print(f"\n{'='*60}")                            # 分隔线
    print("优化方向建议（基于当前 bad case 分析）")      # 标题
    print(f"{'='*60}")                                # 分隔线

    # 下面 print 的大段三引号内容是给用户阅读的优化建议文本（数据层面的通用三板斧）
    print("""
【1】数据层面
  ├─ 难负样本增强（Hard Negative Mining）
  │    当前负样本是随机采样，FP 案例中很多是"话题相关但语义不同"的句对。
  │    → 用训练好的 BiEncoder 在大规模数据中挖掘相似度高但标签为 0 的对，
  │      加入训练集，提升负例的区分难度。
  │
  ├─ 数据增强（正样本扩充）
  │    AFQMC 正样本只有 10K 条（31%），TripletLoss 三元组因此受限。
  │    → 对正样本做同义改写（换词、调序），扩充正例数量。
  │      可用 LLM API 批量生成改写句。
  │
  └─ 跨数据集迁移
       LCQMC / BQ Corpus（已下载）包含更多样的问句对。
       → 先在 LCQMC（238K 对）上预训练 Sentence-BERT，再 fine-tune 到 AFQMC，
         利用大数据集的语义泛化能力。
""")

    if fp_jaccard_mean > 0.5:                       # FP 的字符重叠度高 → 错在"字面太像"
        # 针对高重叠 FP 的模型层建议（加大 margin、加交互特征）
        print("""【2】模型层面（针对 FP：字符重叠高但语义不同）
  ├─ 加大 CosineEmbeddingLoss 的 margin（0.3 → 0.5）
  │    让模型对负例要求更严格，不只是让余弦相似度 < margin，
  │    而是要推得更远，减少高词汇重叠带来的假阳性。
  │
  └─ 引入 Interaction 特征辅助
       BiEncoder 看不到词序交互，而这类 FP 往往需要理解"借呗还款"vs"借呗账单"
       的细粒度区别。可在顶层加一个轻量级 Cross-Attention（保留向量化能力）。
""")
    else:                                          # FP 重叠度不高 → 错在语义理解不足
        # 针对语义理解不足的建议（加深层数、换领域预训练）
        print("""【2】模型层面（FP 字符重叠不高，主要是语义理解不足）
  ├─ 增加 BERT 层数（4 → 8 → 12 层）
  │    浅层 BERT 对语义的建模能力有限，更深层能捕捉更细粒度的语义差异。
  │
  └─ 换用金融领域预训练模型
       FinBERT / MacBERT / RoBERTa-Chinese 在金融/客服语料上有更好的初始化，
       AFQMC 中很多错误源于领域术语理解不准确。
""")

    if fn_jaccard_mean < 0.3:                      # FN 的字符重叠度低 → 错在"换说法认不出"
        # 针对 FN 的训练策略建议（SimCSE、调小 margin）
        print("""【3】训练策略层面（针对 FN：词汇重叠低但语义相同的同义句）
  ├─ SimCSE 对比学习预训练
  │    同一句话 dropout 两次得到两个正例，大 batch 内其他句子为负例。
  │    这种方式能让模型学到"用不同词说同一个意思"的不变性。
  │
  └─ 调小 TripletLoss 的 margin（0.3 → 0.1）
       如果正例本身语义就不太相似（同义但换了词），过大的 margin 反而
       要求 sim(a,p) 比 sim(a,n) 高出太多，训练信号消失。
""")

    # 评估与部署层面的建议（阈值校准、级联精排、加深训练）
    print("""【4】评估与部署层面
  ├─ 阈值校准
  │    当前阈值在 val 集上网格搜索，但 val 正负比（31:69）和线上分布可能不同。
  │    → 收集真实线上日志，按实际分布重新校准阈值（Platt scaling 等）。
  │
  ├─ 两阶段级联（最实用的工程改进）
  │    BiEncoder（召回 Top-K）→ CrossEncoder（精排 Top-1）
  │    这是 rag_annual_report Reranker 的完整版。
  │    → 可用当前两个 checkpoint 直接组合，无需重新训练。
  │
  └─ 训练更多 epoch + 全量 12 层
       本次演示用 4 层 × 1 epoch 快速验证，完整训练预计提升 5~10 个 F1 点。
       → 建议学生实验：4 层 vs 12 层，1 epoch vs 5 epoch 的 2×2 消融。
""")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    """解析 Bad Case 分析脚本命令行的各项参数。"""
    parser = argparse.ArgumentParser(description="Bad Case 分析与优化方向")  # 解析器
    parser.add_argument("--model_type", default="biencoder",
                        choices=["biencoder", "crossencoder"])  # 分析哪种模型
    parser.add_argument("--ckpt", default=None, type=str,
                        help="checkpoint 路径（默认自动选 biencoder_cosine_best.pt）")  # 可显式指定权重
    parser.add_argument("--split",      default="validation",
                        choices=["validation", "test"])  # 分析哪个数据集
    parser.add_argument("--batch_size", default=64, type=int)  # 推理批大小
    parser.add_argument("--n_cases",    default=5, type=int,
                        help="每类 bad case 展示条数")  # 展示条数
    return parser.parse_args()                     # 返回解析结果


def main():
    """分析主流程：收集预测 → 分类 FP/FN → 模式统计与典型案例展示 → 可视化与优化建议。"""
    args   = parse_args()                          # 解析参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选设备

    # ── 确定 checkpoint 路径 ──────────────────────────────────────────────
    if args.ckpt:                                  # 显式传了路径
        ckpt_path = Path(args.ckpt)                # 直接用
    elif args.model_type == "biencoder":           # 双塔默认权重
        ckpt_path = CKPT_DIR / "biencoder_cosine_best.pt"  # cosine 版
    else:                                          # 交叉编码器默认权重
        ckpt_path = CKPT_DIR / "crossencoder_best.pt"  # ce 版

    if not ckpt_path.exists():                     # 权重不存在
        print(f"❌ checkpoint 不存在: {ckpt_path}")  # 报错提示
        print("请先运行训练脚本（train_biencoder.py / train_crossencoder.py）")  # 指引训练
        return                                    # 退出

    print(f"加载 checkpoint: {ckpt_path}")         # 打印权重路径
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)  # 加载 checkpoint
    saved_args = ckpt.get("args", {})              # 训练超参

    # ── 加载原始数据和 DataLoader ─────────────────────────────────────────
    data_path = DATA_DIR / f"{args.split}.jsonl"   # split 数据文件
    raw_rows  = load_jsonl(data_path)              # 原始行（要拿原句展示）
    print(f"数据集: {data_path.name}  共 {len(raw_rows):,} 条")  # 打印数据规模

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))  # 分词器

    # ── 收集预测结果 ──────────────────────────────────────────────────────
    if args.model_type == "biencoder":            # ── 双塔分支 ──
        model = build_biencoder(                   # 重建结构
            bert_path=str(BERT_PATH),              # 预训练路径
            pool=saved_args.get("pool", "mean"),    # 池化
            num_hidden_layers=saved_args.get("num_hidden_layers"),  # 层数
        ).to(device)                                # 上设备
        model.load_state_dict(ckpt["state_dict"])   # 载权重
        threshold = ckpt.get("threshold", 0.5)     # 用训练时搜出的阈值（存过就取，没存默认 0.5）

        ds     = PairDataset(data_path, tokenizer, saved_args.get("max_length", 64))  # 句对数据集
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)  # 不 shuffle 保持与 raw_rows 对齐
        results = collect_biencoder_preds(model, loader, raw_rows, device, threshold)  # 收集预测
        model_label = f"BiEncoder(loss={saved_args.get('loss','cosine')},threshold={threshold:.2f})"  # 图表标签

    else:                                          # ── 交叉编码器分支 ──
        model = build_crossencoder(                 # 重建结构
            bert_path=str(BERT_PATH),               # 预训练路径
            num_hidden_layers=saved_args.get("num_hidden_layers"),  # 层数
        ).to(device)                                # 上设备
        model.load_state_dict(ckpt["state_dict"])   # 载权重
        threshold = 0.5  # P(相似)=0.5 作为分界    # 概率型输出直接 0.5 作分界，无需搜索

        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)  # 拼接数据集
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)  # loader
        results = collect_crossencoder_preds(model, loader, raw_rows, device)  # 收集预测
        model_label = "CrossEncoder(threshold=P≥0.5)"  # 图表标签

    # ── 整体指标 ──────────────────────────────────────────────────────────
    labels    = [r["label"] for r in results]      # 真实标签列表
    preds     = [r["pred"]  for r in results]      # 预测列表
    n_correct = sum(1 for lab, pr in zip(labels, preds) if lab == pr)  # 正确数
    accuracy  = n_correct / len(results)            # 准确率
    print(f"\n整体准确率: {accuracy:.4f}  错误数: {len(results)-n_correct}")  # 打印整体情况

    # ── 分类 bad case ─────────────────────────────────────────────────────
    badcases = split_badcases(results, threshold)   # 四桶分类
    fp_all = badcases["fp_high"] + badcases["fp_border"]  # FP 总数（高置信+临界）
    fn_all = badcases["fn_high"] + badcases["fn_border"]  # FN 总数

    print(f"\n{'='*60}")                           # 分隔线
    print(f"Bad Case 汇总  (共 {len(fp_all)+len(fn_all)} 个错误)")  # 汇总标题
    print(f"{'─'*60}")                             # 分隔线
    print(f"  FP 假阳性（预测相似，实际不同）: {len(fp_all):>4} 条")   # FP 总数
    print(f"    其中高置信度错误  (Δscore>0.15): {len(badcases['fp_high']):>3} 条")  # FP 高置信
    print(f"    其中临界错误     (Δscore≤0.15): {len(badcases['fp_border']):>3} 条")  # FP 临界
    print(f"  FN 假阴性（预测不同，实际相似）: {len(fn_all):>4} 条")   # FN 总数
    print(f"    其中高置信度错误  (Δscore>0.15): {len(badcases['fn_high']):>3} 条")  # FN 高置信
    print(f"    其中临界错误     (Δscore≤0.15): {len(badcases['fn_border']):>3} 条")  # FN 临界

    # ── 模式分析 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")                           # 分隔线
    print("Bad Case 语言特征分析：")                 # 小节标题
    analyze_patterns(fp_all,  "FP（假阳性）")        # FP 的长度/重叠特征
    analyze_patterns(fn_all,  "FN（假阴性）")        # FN 的长度/重叠特征

    fp_jaccard_mean = (np.mean([                    # FP 平均字符 Jaccard
        len(set(r["sentence1"]) & set(r["sentence2"])) /
        max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
        for r in fp_all
    ]) if fp_all else 0)                           # 空 FP 时记 0（防 NaN）
    fn_jaccard_mean = (np.mean([                    # FN 平均字符 Jaccard
        len(set(r["sentence1"]) & set(r["sentence2"])) /
        max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
        for r in fn_all
    ]) if fn_all else 0)                            # 空 FN 记 0

    # ── 展示典型案例 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")                            # 分隔线
    print_cases(badcases["fp_high"],  f"FP 高置信度错误（score最高的{args.n_cases}条）",
                n=args.n_cases)                    # 展示最离谱的 FP
    print_cases(badcases["fp_border"], f"FP 临界错误（{args.n_cases}条）",
                n=args.n_cases)                    # 展示边界 FP
    print_cases(badcases["fn_high"],  f"FN 高置信度错误（score最低的{args.n_cases}条）",
                n=args.n_cases)                    # 展示最离谱的 FN
    print_cases(badcases["fn_border"], f"FN 临界错误（{args.n_cases}条）",
                n=args.n_cases)                    # 展示边界 FN

    # ── 可视化 ────────────────────────────────────────────────────────────
    fig_path = FIG_DIR / f"{args.model_type}_badcase_dist.png"  # 图路径（带模型类型）
    plot_score_dist_with_errors(results, threshold, fig_path, model_label)  # 画双联分布图

    # ── 优化方向 ──────────────────────────────────────────────────────────
    print_optimization_directions(badcases, args.model_type,
                                  fp_jaccard_mean, fn_jaccard_mean)  # 按 Jaccard 特征给定制建议


if __name__ == "__main__":
    main()                                          # 直跑才执行分析
