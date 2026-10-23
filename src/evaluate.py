"""
文本匹配评估工具（可作模块导入，也可独立运行）

教学重点：
  1. BiEncoder 评估的关键差异——输出是相似度分数，需要阈值搜索才能变为预测标签
  2. 阈值搜索 (threshold search)：在验证集上枚举阈值，选 F1 最高的
  3. CrossEncoder 评估与分类任务完全一样——直接取 argmax 即为预测标签
  4. 相似度分布图——正负样本在相似度轴上的分离程度是模型质量的直观反映

使用方式：
  # 独立运行（加载 checkpoint 评估）
  python evaluate.py --model_type biencoder --ckpt ../outputs/checkpoints/biencoder_best.pt
  python evaluate.py --model_type crossencoder --ckpt ../outputs/checkpoints/crossencoder_best.pt

  # 作为模块导入（训练脚本内调用）
  from evaluate import eval_biencoder, eval_crossencoder

依赖：
  pip install torch transformers scikit-learn matplotlib
"""
# ↑ 模块级 docstring：训练脚本每个 epoch 末都会 import 这里的评估函数

import os                                        # 环境变量设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 Windows 下 numpy/matplotlib 与 torch 的 OpenMP 冲突

import argparse                                   # 独立运行时的命令行解析
from pathlib import Path                          # 路径处理

import matplotlib                                 # 绘图库
matplotlib.use("Agg")                             # 无界面后端：不弹窗，只存文件（服务器环境必需）
import matplotlib.pyplot as plt                    # 绘图 API
import numpy as np                                # 数值计算
import torch                                      # 深度学习框架
import torch.nn.functional as F                   # cosine_similarity 函数
from sklearn.metrics import (                     # sklearn 评估指标
    accuracy_score,                               # 准确率
    classification_report,                        # 每类 precision/recall/f1 报告
    f1_score,                                     # F1 分数
    roc_auc_score,                                # ROC-AUC（阈值无关指标）
)
from transformers import BertTokenizer            # 分词器

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent          # 项目根目录
DATA_DIR   = ROOT / "data" / "blood_pressure"     # 数据目录
BERT_PATH  = ROOT.parent.parent.parent / "pretrain_models" / "bert-base-chinese"  # 预训练权重
OUTPUT_DIR = ROOT / "outputs"                     # 输出根目录
CKPT_DIR   = OUTPUT_DIR / "checkpoints"           # checkpoint 目录
FIG_DIR    = OUTPUT_DIR / "figures"               # 图表输出目录


# ── BiEncoder 评估 ────────────────────────────────────────────────────────

@torch.no_grad()                                  # 装饰器：整个函数内不建计算图（省显存提速，评估不需要梯度）
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5):
    """
    BiEncoder 评估：计算每对句子的余弦相似度，然后在 val 集上搜索最优阈值。

    返回 dict：
      similarities : list[float]  每对的余弦相似度
      labels       : list[int]    真实标签
      accuracy     : float        最优阈值下的准确率
      f1           : float        最优阈值下的 F1（weighted）
      threshold    : float        最优阈值（若 find_threshold=True）
      auc          : float        ROC-AUC（不依赖阈值）
    """
    model.eval()                                  # 切评估模式（关 Dropout，保证打分稳定）
    all_sims, all_labels = [], []                 # 收集全数据的相似度与标签

    for batch in loader:                          # 遍历评估 loader
        batch_a = {                               # 塔一输入（_a 后缀三项搬设备）
            "input_ids":      batch["input_ids_a"].to(device),       # 句 1 token ids
            "attention_mask": batch["attention_mask_a"].to(device),  # 句 1 掩码
            "token_type_ids": batch["token_type_ids_a"].to(device),  # 句 1 类型
        }
        batch_b = {                               # 塔二输入
            "input_ids":      batch["input_ids_b"].to(device),       # 句 2 token ids
            "attention_mask": batch["attention_mask_b"].to(device),  # 句 2 掩码
            "token_type_ids": batch["token_type_ids_b"].to(device),  # 句 2 类型
        }
        emb_a, emb_b = model(batch_a, batch_b)     # 双塔编码 → 两个 [B, H] 归一化向量
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()  # 逐对算余弦 → 回 CPU 转 list
        all_sims.extend(sims)                      # 累积相似度
        all_labels.extend(batch["label"].tolist())  # 累积标签

    sims   = np.array(all_sims)                   # 转 numpy 便于向量化阈值搜索
    labels = np.array(all_labels)                 # 标签同样转 numpy

    if find_threshold:                             # 默认开：在验证集上搜最优阈值
        threshold = _find_best_threshold(sims, labels)  # 枚举候选阈值选 F1 最高者

    preds    = (sims >= threshold).astype(int)     # 相似度 ≥ 阈值 判为"匹配"（1），否则 0
    accuracy = accuracy_score(labels, preds)       # 计算准确率
    f1       = f1_score(labels, preds, average="weighted", zero_division=0)  # 加权 F1（zero_division 兜底空类）

    # AUC：若 labels 只有一类（如 AFQMC test 全为 0），跳过
    try:
        auc = roc_auc_score(labels, sims)          # 用连续相似度当分数算 AUC
    except ValueError:                             # 单一类别会抛 ValueError
        auc = float("nan")                         # 记 NaN 表示不可算

    return {                                       # 打包返回所有评估产物
        "similarities": all_sims,                 # 每对相似度（画分布图用）
        "labels":       all_labels,               # 对应真实标签
        "accuracy":     accuracy,                 # 阈值下准确率
        "f1":           f1,                       # 阈值下加权 F1
        "threshold":    threshold,                 # 搜出的最优阈值（部署时直接用）
        "auc":          auc,                      # 阈值无关的 AUC
    }


def _find_best_threshold(sims, labels):
    """枚举 [0.0, 1.0] 区间 101 个候选阈值，返回使 weighted-F1 最高的那个。"""
    best_f1, best_thresh = -1.0, 0.5              # 初始化最优 F1 为负值保证必被更新；默认阈值 0.5
    for t in np.linspace(0.0, 1.0, 101):          # 0.00, 0.01, ..., 1.00 共 101 个候选
        preds = (sims >= t).astype(int)            # 当前阈值下的二值预测
        f1    = f1_score(labels, preds, average="weighted", zero_division=0)  # weighted-F1
        if f1 > best_f1:                          # 创新高
            best_f1    = f1                       # 更新最优 F1
            best_thresh = t                       # 记下对应阈值
    return float(best_thresh)                     # 返回最优阈值


# ── CrossEncoder 评估 ─────────────────────────────────────────────────────

@torch.no_grad()                                  # 推理无梯度
def eval_crossencoder(model, loader, device):
    """
    CrossEncoder 评估：与分类任务完全一致，取 argmax 得预测标签。

    返回 dict：
      accuracy : float
      f1       : float（weighted）
      logits   : list[list[float]]
      labels   : list[int]
    """
    model.eval()                                  # 评估模式
    all_logits, all_labels = [], []                # 收集 logits 与标签

    for batch in loader:                          # 遍历 loader
        input_ids      = batch["input_ids"].to(device)        # 拼接序列 → 设备
        attention_mask = batch["attention_mask"].to(device)   # 掩码 → 设备
        token_type_ids = batch["token_type_ids"].to(device)   # 句界 → 设备
        labels         = batch["label"]                      # 标签留 CPU（不参与计算图）

        logits = model(input_ids, attention_mask, token_type_ids).cpu()  # 前向得 [B,2] 分数并搬回 CPU
        all_logits.extend(logits.tolist())          # 累积 logits
        all_labels.extend(labels.tolist())          # 累积标签

    preds    = np.argmax(all_logits, axis=1)       # 沿类别维取最大下标 = 预测类别（无需阈值）
    labels   = np.array(all_labels)                # 标签转 numpy
    accuracy = accuracy_score(labels, preds)       # 准确率
    f1       = f1_score(labels, preds, average="weighted", zero_division=0)  # 加权 F1

    return {                                       # 返回评估结果
        "logits":   all_logits,                   # 原始 logits（可复算概率）
        "labels":   all_labels,                   # 真实标签
        "accuracy": accuracy,                      # 准确率
        "f1":       f1,                            # F1
    }


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_similarity_distribution(sims, labels, threshold, save_path, title="相似度分布"):
    """
    绘制正/负样本的余弦相似度分布直方图（BiEncoder 专属）。

    教学价值：
      - 理想情况：正样本集中在 [threshold, 1]，负样本集中在 [-1, threshold)
      - 分布重叠越少 → 模型判别能力越强
    """
    sims   = np.array(sims)                       # 相似度转 numpy
    labels = np.array(labels)                     # 标签转 numpy

    pos_sims = sims[labels == 1]                  # 正样本对的相似度
    neg_sims = sims[labels == 0]                   # 负样本对的相似度

    fig, ax = plt.subplots(figsize=(8, 4))         # 建 8×4 英寸画布
    ax.hist(pos_sims, bins=50, alpha=0.6, label=f"正样本 (n={len(pos_sims)})",  # 正样本直方图（半透明蓝）
            color="#2196F3", density=True)         # density=True 归一成密度，两类数量不同也能对比形状
    ax.hist(neg_sims, bins=50, alpha=0.6, label=f"负样本 (n={len(neg_sims)})",  # 负样本直方图（半透明红）
            color="#F44336", density=True)         # 同上归一化
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"最优阈值 = {threshold:.2f}")   # 画黑色竖虚线标出最优阈值位置
    ax.set_xlabel("余弦相似度")                     # x 轴标签
    ax.set_ylabel("密度")                          # y 轴标签
    ax.set_title(title)                            # 图标题
    ax.legend()                                    # 图例
    fig.tight_layout()                             # 自动收紧布局防标签裁切

    save_path = Path(save_path)                    # 包装 Path
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    fig.savefig(save_path, dpi=150)                # 150 DPI 存 PNG
    plt.close(fig)                                 # 关闭画布释放内存（防止图越攒越多）
    print(f"  图表已保存 → {save_path}")            # 打印落盘位置


# ── 独立运行入口 ──────────────────────────────────────────────────────────

def parse_args():
    """解析评估脚本命令行的各项参数。"""
    parser = argparse.ArgumentParser(description="文本匹配模型评估")  # 解析器
    parser.add_argument("--model_type", required=True, choices=["biencoder", "crossencoder"])  # 必填：评估哪种模型
    parser.add_argument("--ckpt",       required=True, type=str, help="checkpoint .pt 文件路径")  # 必填：权重路径
    parser.add_argument("--data_dir",   default=str(DATA_DIR), type=str)  # 数据目录
    parser.add_argument("--bert_path",  default=str(BERT_PATH), type=str)  # 分词器路径
    parser.add_argument("--batch_size", default=64, type=int)    # 评估批大小（可大于训练批）
    parser.add_argument("--max_length", default=64,  type=int, help="BiEncoder 单句最大长度")  # 单句截断长
    parser.add_argument("--split",      default="validation", choices=["validation", "test"],
                        help="评估的数据集分割（AFQMC test 无正样本，建议用 validation）")  # 评估哪个 split
    return parser.parse_args()                     # 返回解析结果


def main():
    """独立运行入口：加载 checkpoint，对指定 split 评估并输出指标与图表。"""
    args   = parse_args()                          # 解析参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选设备
    print(f"设备: {device}")                        # 打印设备
    print(f"加载 checkpoint: {args.ckpt}")           # 打印权重路径

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)  # 加载 checkpoint 到设备（含权重与超参）
    print(f"训练信息: {ckpt.get('args', {})}")       # 回显训练时超参（确认结构与训练一致）

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)  # 加载分词器
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"   # 拼 split 数据文件路径

    FIG_DIR.mkdir(parents=True, exist_ok=True)      # 确保图表目录存在

    if args.model_type == "biencoder":              # ── BiEncoder 分支 ──
        from model import build_biencoder            # 函数内导入（用哪个加载哪个，省内存）
        from dataset import PairDataset              # 句对数据集
        from torch.utils.data import DataLoader      # 批量加载器

        saved_args = ckpt.get("args", {})            # 取训练超参
        model = build_biencoder(                     # 按 checkpoint 结构重建模型
            bert_path=args.bert_path,                # 预训练路径
            pool=saved_args.get("pool", "mean"),     # 池化策略（默认 mean）
            num_hidden_layers=saved_args.get("num_hidden_layers"),  # 裁剪层数
        ).to(device)                                # 上设备
        model.load_state_dict(ckpt["state_dict"])    # 载入训练权重

        ds     = PairDataset(data_path, tokenizer, args.max_length)  # 评估数据集
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)  # 不打乱保证可复现

        metrics = eval_biencoder(model, loader, device)  # 跑评估（内含阈值搜索）
        print(f"\n{'='*50}")                        # 分隔线
        print(f"BiEncoder 评估结果（{args.split}，{len(ds)} 条）")  # 标题（split 与规模）
        print(f"  最优阈值: {metrics['threshold']:.2f}")   # 搜出的阈值
        print(f"  Accuracy: {metrics['accuracy']:.4f}")   # 准确率
        print(f"  F1      : {metrics['f1']:.4f}")         # F1
        print(f"  AUC     : {metrics['auc']:.4f}")        # AUC

        plot_similarity_distribution(               # 画相似度分布图
            metrics["similarities"], metrics["labels"], metrics["threshold"],  # 数据与阈值
            save_path=FIG_DIR / f"biencoder_{args.split}_sim_dist.png",  # 图文件名带 split
            title=f"BiEncoder 相似度分布（{args.split}）",  # 图标题
        )

        # 打印分类报告
        preds = (np.array(metrics["similarities"]) >= metrics["threshold"]).astype(int)  # 按最优阈值重算预测
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")  # 逐类 P/R/F1 报告

    elif args.model_type == "crossencoder":          # ── CrossEncoder 分支 ──
        from model import build_crossencoder         # 交叉编码器工厂
        from dataset import CrossEncoderDataset     # 交互型数据集
        from torch.utils.data import DataLoader       # 加载器

        saved_args = ckpt.get("args", {})            # 训练超参
        model = build_crossencoder(                  # 重建结构
            bert_path=args.bert_path,                # 预训练路径
            num_hidden_layers=saved_args.get("num_hidden_layers"),  # 层数
        ).to(device)                                 # 上设备
        model.load_state_dict(ckpt["state_dict"])     # 载权重

        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)  # 句对拼接数据集（固定 128）
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)  # loader

        metrics = eval_crossencoder(model, loader, device)  # 跑评估
        print(f"\n{'='*50}")                         # 分隔线
        print(f"CrossEncoder 评估结果（{args.split}，{len(ds)} 条）")  # 标题
        print(f"  Accuracy: {metrics['accuracy']:.4f}")   # 准确率
        print(f"  F1      : {metrics['f1']:.4f}")         # F1

        preds = np.argmax(metrics["logits"], axis=1)  # argmax 出预测
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")  # 分类报告


if __name__ == "__main__":
    main()                                          # 直跑才执行独立评估
