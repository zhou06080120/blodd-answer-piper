"""
多方法效果对比脚本

对比三种文本匹配方式在 AFQMC validation 集上的效果：
  1. BiEncoder + CosineEmbeddingLoss
  2. BiEncoder + TripletLoss
  3. CrossEncoder + CrossEntropyLoss

教学重点：
  1. 三种方式都用 4 层 BERT，1 epoch 训练，控制变量对比 Loss 函数差异
  2. BiEncoder 需阈值搜索，CrossEncoder 直接 argmax——评估方式本身就是知识点
  3. 速度/精度权衡：CrossEncoder 精度最高但推理慢，BiEncoder 可向量化
  4. 输出对比图：正负样本相似度分布（BiEncoder × 2）+ 混淆矩阵（CrossEncoder）

使用方式：
  python compare_methods.py
  python compare_methods.py --split validation --batch_size 64

依赖：
  pip install torch transformers scikit-learn matplotlib
"""
# ↑ 模块级 docstring：同一数据集、同一预算下三种训练方案的公平对比实验

import os                                        # 环境变量
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 OpenMP 重复加载崩溃

import argparse                                   # 命令行参数
import json                                       # 对比日志落盘
from pathlib import Path                          # 路径

import matplotlib                                 # 绘图库
matplotlib.use("Agg")                             # 无界面后端（只存文件）
import matplotlib.pyplot as plt                    # 绘图 API
import numpy as np                                 # 柱状图 x 轴定位等
import torch                                      # 加载 checkpoint
from torch.utils.data import DataLoader            # 批量加载
from transformers import BertTokenizer             # 分词器

from dataset import PairDataset, CrossEncoderDataset  # 两类数据集
from evaluate import eval_biencoder, eval_crossencoder  # 两类评估与分布图
from model import build_biencoder, build_crossencoder  # 两类模型工厂

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent          # 项目根
DATA_DIR   = ROOT / "data" / "blood_pressure"     # 数据目录
BERT_PATH  = r"F:\aaa123\pretrain_models\bert-base-chinese"  # 预训练权重（绝对路径）
CKPT_DIR   = ROOT / "outputs" / "checkpoints"     # 三个模型的权重目录
FIG_DIR    = ROOT / "outputs" / "figures"         # 图表输出
LOG_DIR    = ROOT / "outputs" / "logs"             # 对比日志

METHODS = [                                      # 参与对比的三种方法配置表
    {                                             # 方法 1：双塔 + 余弦损失
        "key":       "biencoder_cosine",           # 标识键（日志/表格用）
        "label":     "BiEncoder\n(CosineEmbeddingLoss)",  # 图表标签（含换行）
        "ckpt":      "biencoder_cosine_best.pt",   # 权重文件名
        "type":      "biencoder",                   # 模型类型（决定加载与评估分支）
        "color":     "#2196F3",                     # 图表配色（蓝）
    },
    {                                             # 方法 2：双塔 + 三元组损失
        "key":       "biencoder_triplet",           # 标识键
        "label":     "BiEncoder\n(TripletLoss)",   # 标签
        "ckpt":      "biencoder_triplet_best.pt",  # 权重文件名
        "type":      "biencoder",                   # 同为双塔
        "color":     "#4CAF50",                     # 配色（绿）
    },
    {                                             # 方法 3：交叉编码器
        "key":       "crossencoder",                # 标识键
        "label":     "CrossEncoder\n(CrossEntropyLoss)",  # 标签
        "ckpt":      "crossencoder_best.pt",        # 权重文件名
        "type":      "crossencoder",                # 交互型
        "color":     "#FF9800",                     # 配色（橙）
    },
]


# ── 加载并评估单个方法 ─────────────────────────────────────────────────────

def load_and_eval(method, tokenizer, device, split, batch_size):
    ckpt_path = CKPT_DIR / method["ckpt"]           # 拼权重路径
    if not ckpt_path.exists():                      # 该方法还没训练过
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")  # 跳过并提示
        return None                                 # 返回 None 由上层忽略

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)  # 加载 checkpoint（权重+超参）
    saved     = ckpt.get("args", {})                # 取训练超参

    if method["type"] == "biencoder":               # ── 双塔分支 ──
        model = build_biencoder(                    # 重建结构
            bert_path=str(BERT_PATH),               # 预训练路径
            pool=saved.get("pool", "mean"),         # 池化策略
            num_hidden_layers=saved.get("num_hidden_layers"),  # 层数
        ).to(device)                                # 上设备
        model.load_state_dict(ckpt["state_dict"])   # 载权重
        model.eval()                                # 评估模式

        data_path = DATA_DIR / f"{split}.jsonl"     # split 数据文件
        ds     = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))  # 句对数据集（max_length 沿用训练值）
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)  # loader
        metrics = eval_biencoder(model, loader, device)  # 评估（阈值搜索内含）

    else:  # crossencoder                          # ── 交叉编码器分支 ──
        model = build_crossencoder(                 # 重建结构
            bert_path=str(BERT_PATH),               # 预训练路径
            num_hidden_layers=saved.get("num_hidden_layers"),  # 层数
        ).to(device)                                # 上设备
        model.load_state_dict(ckpt["state_dict"])   # 载权重
        model.eval()                                # 评估模式

        data_path = DATA_DIR / f"{split}.jsonl"     # split 数据文件
        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)  # 拼接数据集
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)  # loader
        metrics = eval_crossencoder(model, loader, device)  # 评估（argmax）

    metrics["model"] = model                        # 把模型对象也塞进 metrics（后续画图可能用）
    metrics["ckpt"]  = ckpt                          # checkpoint 原样附带
    return metrics                                  # 返回该方法的完整指标


# ── 对比可视化 ────────────────────────────────────────────────────────────

def plot_comparison_bar(results, save_path):
    """准确率 / F1 对比柱状图"""
    names    = [m["label"]    for m in results]     # 方法名列表
    accs     = [m["accuracy"] for m in results]     # 各方法准确率
    f1s      = [m["f1"]       for m in results]     # 各方法 F1
    colors   = [m["color"]    for m in results]     # 各方法配色

    x = np.arange(len(names))                       # 方法在 x 轴的刻度位置 0,1,2
    w = 0.35                                         # 柱宽（两柱并列时留出间隙）

    fig, ax = plt.subplots(figsize=(9, 5))            # 画布
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.85)  # 左柱：准确率
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color=colors, alpha=0.5,
                   hatch="//", edgecolor="white")    # 右柱：F1（半透明+斜线纹理区分）

    for bar in bars1:                                # 准确率柱顶标注数值
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)  # 柱顶上方居中写数
    for bar in bars2:                                # F1 柱顶同样标注
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)  # 同上

    ax.set_xticks(x)                                 # 设置 x 刻度位置
    ax.set_xticklabels(names, fontsize=9)            # 刻度标签用方法名（字号 9 放得下换行标签）
    ax.set_ylim(0, 1.0)                              # y 轴固定 0~1（指标是比例）
    ax.set_ylabel("Score")                           # y 轴名
    ax.set_title("Method Comparison on AFQMC Validation (4-layer BERT, 1 epoch)")  # 标题注明控制变量
    ax.legend()                                      # 图例
    ax.grid(axis="y", alpha=0.3)                     # y 向浅网格便于读数
    fig.tight_layout()                                # 布局

    save_path = Path(save_path)                      # 包装路径
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 建目录
    fig.savefig(save_path, dpi=150)                   # 存图
    plt.close(fig)                                    # 关画布
    print(f"  图表已保存 → {save_path}")               # 提示


def plot_sim_distributions(biencoder_results, save_path):
    """所有 BiEncoder 方法的相似度分布叠放对比"""
    n = len(biencoder_results)                       # 双塔方法数量
    if n == 0:                                       # 没有双塔结果
        return                                       # 直接跳过
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)  # 一行 n 张子图（每张 6×4）
    if n == 1:                                       # 单图时 subplots 不返回数组
        axes = [axes]                                # 统一列表

    for ax, m in zip(axes, biencoder_results):        # 子图与方法一一对应
        sims   = np.array(m["similarities"])          # 该方法全部相似度
        labels = np.array(m["labels"])                # 对应标签
        ax.hist(sims[labels==1], bins=40, alpha=0.6, label="positive", color="#2196F3", density=True)  # 正样本分布
        ax.hist(sims[labels==0], bins=40, alpha=0.6, label="negative", color="#F44336", density=True)  # 负样本分布
        ax.axvline(m["threshold"], color="black", linestyle="--",
                   label=f"threshold={m['threshold']:.2f}")  # 最优阈值竖线
        ax.set_title(m["label"].replace("\n", " "))   # 子图标题（去换行）
        ax.set_xlabel("Cosine Similarity")             # x 轴
        ax.legend(fontsize=8)                          # 图例

    fig.suptitle("BiEncoder Similarity Distribution (positive vs negative)", y=1.01)  # 总标题
    fig.tight_layout()                                 # 布局
    fig.savefig(save_path, dpi=150, bbox_inches="tight")  # 存图（紧贴边界）
    plt.close(fig)                                     # 关画布
    print(f"  图表已保存 → {save_path}")                 # 提示


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    """解析方法对比脚本命令行的各项参数。"""
    parser = argparse.ArgumentParser(description="三种文本匹配方法效果对比")  # 解析器
    parser.add_argument("--split",      default="validation", choices=["validation", "test"])  # 评估集选择
    parser.add_argument("--batch_size", default=64, type=int)  # 推理批大小
    parser.add_argument("--bert_path", type=str, required=True, help="BERT预训练模型本地路径")  # 必填：预训练路径
    return parser.parse_args()                        # 返回解析结果


def main():
    """对比主流程：逐方法评估 → 控制台对比表 → 生成柱状图与相似度分布图并落日志。"""
    args   = parse_args()                             # 解析参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选设备
    print(f"设备: {device}  评估集: {args.split}")       # 打印环境信息

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))  # 加载分词器（用常量路径，参数备用）
    FIG_DIR.mkdir(parents=True, exist_ok=True)         # 确保图表目录存在

    # ── 逐方法评估 ────────────────────────────────────────────────────────
    all_results = []                                   # 汇总可用方法的结果
    for m in METHODS:                                  # 遍历三种方法配置
        print(f"\n{'='*55}")                           # 分隔线
        print(f"加载 {m['key']} ...")                   # 进度提示
        metrics = load_and_eval(m, tokenizer, device, args.split, args.batch_size)  # 加载并评估
        if metrics is None:                           # 权重缺失被跳过
            continue                                  # 不计入对比
        metrics.update({"label": m["label"], "color": m["color"], "key": m["key"],
                        "type": m["type"]})            # 把展示信息合并进指标
        all_results.append(metrics)                   # 收集

    if not all_results:                                # 一个权重都没有
        print("没有可用的 checkpoint，请先运行训练脚本。")  # 提示先训练
        return                                        # 结束

    # ── 控制台对比表 ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")                              # 分隔线
    print(f"{'方法':<30} {'Accuracy':>9} {'F1(weighted)':>13} {'额外信息':>15}")  # 表头（定宽对齐）
    print(f"{'-'*65}")                                # 分隔线
    for m in all_results:                             # 逐方法打印
        extra = (f"threshold={m['threshold']:.2f}"
                 if m["type"] == "biencoder" else "argmax")  # 双塔显示阈值，交叉显示 argmax
        print(f"  {m['key']:<28} {m['accuracy']:>9.4f} {m['f1']:>13.4f} {extra:>15}")  # 一行指标

    print(f"\n{'─'*65}")                              # 分隔线
    print("结论速览：")                                  # 小结标题
    best_acc = max(all_results, key=lambda x: x["accuracy"])  # 准确率冠军
    best_f1  = max(all_results, key=lambda x: x["f1"])     # F1 冠军
    print(f"  最高 Accuracy : {best_acc['key']} ({best_acc['accuracy']:.4f})")  # 打印冠军
    print(f"  最高 F1       : {best_f1['key']}  ({best_f1['f1']:.4f})")   # 同上

    # 列出不同 Loss 的 BiEncoder 对比
    bi_results = [m for m in all_results if m["type"] == "biencoder"]  # 只取双塔结果
    if len(bi_results) == 2:                           # 恰好两种 Loss 都在
        a, b = bi_results                              # a=cosine b=triplet（按 METHODS 顺序）
        delta_acc = b["accuracy"] - a["accuracy"]       # Triplet - Cosine 的准确率差
        delta_f1  = b["f1"] - a["f1"]                   # F1 差
        print("\n  Cosine vs Triplet (Δ):")            # 差值标题
        print(f"    Accuracy: {delta_acc:+.4f}  F1: {delta_f1:+.4f}")  # 打印差值（带符号）
        if abs(delta_f1) < 0.01:                        # 差距不到 1%
            print("    → 两种 Loss 差距不大（1 epoch + 少量三元组限制了 Triplet 的优势）")  # 解读一
        elif delta_f1 > 0:                              # Triplet 更高
            print("    → TripletLoss 更优，三元组对语义距离的约束更精确")  # 解读二
        else:                                          # Cosine 更高
            print("    → CosineEmbeddingLoss 更优，AFQMC 数据量下直接对标签优化更稳定")  # 解读三

    # ── 保存对比日志 ──────────────────────────────────────────────────────
    SKIP_KEYS = {"model", "similarities", "labels", "logits", "ckpt"}  # 日志里剔除大对象与原始数组

    def _to_py(v):                                     # 序列化辅助
        if hasattr(v, "item"):  # numpy scalar / torch scalar  # 标量对象（np/torch）
            return v.item()                            # 转成 Python 原生数值（JSON 才能序列化）
        return v                                       # 其他类型原样返回

    log = [{k: _to_py(v) for k, v in m.items() if k not in SKIP_KEYS}
           for m in all_results]                      # 每个方法剔除 SKIP_KEYS 并转原生类型
    log_path = LOG_DIR / "method_comparison.json"     # 日志路径
    LOG_DIR.mkdir(parents=True, exist_ok=True)        # 建日志目录
    with open(log_path, "w", encoding="utf-8") as f:   # 写文件
        json.dump(log, f, ensure_ascii=False, indent=2)  # 缩进 2、中文原样
    print(f"\n对比日志 → {log_path}")                   # 提示

    # ── 可视化 ────────────────────────────────────────────────────────────
    plot_comparison_bar(all_results, FIG_DIR / "method_comparison_bar.png")  # 三方法指标柱状图

    bi_with_sim = [m for m in all_results if m["type"] == "biencoder" and "similarities" in m]  # 有相似度数据的双塔结果
    if bi_with_sim:                                    # 至少一个
        plot_sim_distributions(bi_with_sim, FIG_DIR / "biencoder_sim_distributions.png")  # 分布对比图


if __name__ == "__main__":
    main()                                             # 直跑才执行
