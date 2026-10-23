"""
AFQMC 数据集探索与可视化

教学重点：
  1. 文本匹配数据的结构——sentence pair + binary label，与分类任务的本质区别
  2. 类别不均衡（~31% 正例）——实际业务中"不同义"的问题对比"同义"更多见
  3. 句子长度分布——BERT max_length 截断阈值的选择依据
  4. 正/负样本的长度差异——是否存在"长句倾向于不相似"的捷径（shortcut）
  5. Token 数 vs 字符数——BERT 中文字节对编码的粒度

使用方式：
  python explore_data.py
  python explore_data.py --data_dir ../data/afqmc --output_dir ../outputs/figures

依赖：
  pip install matplotlib transformers
"""
# ↑ 模块级 docstring：建模前的第一步——看数据长什么样、有没有偏差、截断长度怎么选

import os                                        # 环境变量设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 Windows 上多库共存时的 OpenMP 崩溃

import argparse                                   # 命令行参数
import json                                       # 解析 JSONL
from collections import Counter                  # 标签计数器
from pathlib import Path                          # 路径操作

import matplotlib                                 # 绘图库
matplotlib.use("Agg")                             # 无界面后端：只存文件不弹窗
import matplotlib.pyplot as plt                    # 绘图 API
import matplotlib.font_manager as fm               # 字体管理（找中文字体用）
import numpy as np                                 # 数值统计
from transformers import BertTokenizer             # 分析 token 长度需要真分词器

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent          # 项目根目录
DATA_DIR   = ROOT / "data" / "blood_pressure"     # 数据目录
BERT_PATH  = ROOT.parent.parent.parent / "pretrain_models" / "bert-base-chinese"  # 预训练权重
OUTPUT_DIR = ROOT / "outputs" / "figures"         # 图表输出目录

# 中文字体（matplotlib 默认不支持中文）
_CN_FONT = None                                   # 缓存找到的字体（只找一次）
def _get_font():
    """查找系统中支持中文的字体，供 matplotlib 复用（找不到时回退默认字体）。"""
    global _CN_FONT                                 # 声明用全局缓存
    if _CN_FONT is None:                            # 首次调用
        try:
            font_path = next(                       # 找第一个满足条件的系统字体
                p for p in fm.findSystemFonts()     # 遍历系统字体路径
                if any(k in p.lower() for k in ("simhei", "msyh", "simsun", "notosans"))  # 黑体/雅黑/宋体/NotoSans 任一
            )
            _CN_FONT = fm.FontProperties(fname=font_path)  # 用该字体文件建字体属性对象
        except StopIteration:                       # 一个中文字体都没有
            _CN_FONT = fm.FontProperties()           # 回退默认字体（中文会显示成方块，但不崩溃）
    return _CN_FONT                                 # 返回字体属性（后续调用直接命中缓存）


def load_jsonl(path):
    rows = []                                       # 样本收集列表
    with open(path, encoding="utf-8") as f:         # UTF-8 打开
        for line in f:                               # 逐行读
            line = line.strip()                      # 去空白
            if line:                                 # 非空行
                rows.append(json.loads(line))       # 解析成 dict 收集
    return rows                                     # 返回样本列表


# ── 图 1：标签分布 ────────────────────────────────────────────────────────

def plot_label_distribution(splits_data, output_dir):
    fig, axes = plt.subplots(1, len(splits_data), figsize=(10, 4))  # 一行子图：每个 split 一张
    if len(splits_data) == 1:                        # subplots 单图时不返回数组
        axes = [axes]                                # 统一包成列表供 zip

    fp = _get_font()                                 # 中文字体
    for ax, (split_name, rows) in zip(axes, splits_data.items()):  # 子图与 split 数据配对遍历
        labels = [r["label"] for r in rows]           # 抽出全部标签
        cnt = Counter(labels)                        # 计数 0/1（可能含 -1）
        counts = [cnt.get(0, 0), cnt.get(1, 0)]       # 按类别取数量（缺失记 0）
        bars = ax.bar(["不相似 (0)", "相似 (1)"], counts,  # 画柱状图
                      color=["#F44336", "#2196F3"], width=0.5)  # 红负蓝正、宽 0.5
        for bar, c in zip(bars, counts):              # 每根柱子标注数值
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,  # 柱顶上方
                    f"{c}\n({c/len(rows)*100:.1f}%)", ha="center", va="bottom",  # 数量与占比
                    fontproperties=fp, fontsize=9)    # 用中文字体
        ax.set_title(f"{split_name}（{len(rows):,} 条）", fontproperties=fp)  # 子图标题含规模
        ax.set_ylabel("数量", fontproperties=fp)      # y 轴标签
        ax.tick_params(axis="x", labelsize=9)         # x 刻度字号缩一点防重叠

    fig.suptitle("血压计问答 标签分布", fontproperties=fp,
                 fontsize=12, y=1.02)                 # 总标题（y=1.02 略高于子图）
    fig.tight_layout()                                # 收紧布局
    save_path = output_dir / "label_distribution.png"  # 保存路径
    fig.savefig(save_path, dpi=150, bbox_inches="tight")  # bbox_tight 连图例一起保住
    plt.close(fig)                                    # 关闭画布
    print(f"  图表已保存 → {save_path}")               # 打印落盘位置


# ── 图 2：句子字符长度分布 ────────────────────────────────────────────────

def plot_char_length(rows, output_dir):
    pos_rows = [r for r in rows if r["label"] == 1]    # 正样本行
    neg_rows = [r for r in rows if r["label"] == 0]    # 负样本行

    def lens(rs):                                    # 辅助函数：取一组行两句的长度
        return [len(r["sentence1"]) for r in rs] + [len(r["sentence2"]) for r in rs]  # s1、s2 长度合并

    pos_lens = lens(pos_rows)                         # 正样本句长
    neg_lens = lens(neg_rows)                         # 负样本句长

    fp = _get_font()                                   # 中文字体
    fig, ax = plt.subplots(figsize=(8, 4))             # 画布
    ax.hist(pos_lens, bins=40, alpha=0.6, label="正样本（相似）",  # 正样本直方图
            color="#2196F3", density=True)             # 归一化密度便于形状对比
    ax.hist(neg_lens, bins=40, alpha=0.6, label="负样本（不相似）",  # 负样本直方图
            color="#F44336", density=True)             # 同上
    ax.axvline(32, color="black", linestyle="--", linewidth=1,
               label="max_length=32")                  # 参考线：截断 32 时会切掉多少
    ax.axvline(64, color="gray", linestyle="--", linewidth=1,
               label="max_length=64")                  # 参考线：截断 64
    ax.set_xlabel("句子字符长度", fontproperties=fp)     # x 轴
    ax.set_ylabel("密度", fontproperties=fp)           # y 轴
    ax.set_title("正/负样本句子长度分布（train）", fontproperties=fp)  # 标题
    ax.legend(prop=fp)                                 # 图例（中文）
    fig.tight_layout()                                 # 布局

    save_path = output_dir / "char_length_distribution.png"  # 保存路径
    fig.savefig(save_path, dpi=150)                    # 存图
    plt.close(fig)                                     # 关图
    print(f"  图表已保存 → {save_path}")                # 提示

    all_lens = [len(r["sentence1"]) for r in rows] + [len(r["sentence2"]) for r in rows]  # 全部句子长度
    print("  字符长度统计（train 全部句子）：")           # 控制台统计
    print(f"    均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")  # 均值/中位/P95/最长
    for threshold in [32, 48, 64, 96]:                # 几个候选 max_length
        cover = sum(1 for ln in all_lens if ln <= threshold) / len(all_lens) * 100  # 覆盖率 = 不被截断的句子占比
        print(f"    max_length={threshold:3d} 覆盖率: {cover:.1f}%")  # 打印覆盖率辅助选截断长度


# ── 图 3：Token 数分布（BERT Tokenizer） ─────────────────────────────────

def plot_token_length(rows, tokenizer, output_dir):
    print("  计算 Token 长度（需要 tokenize，稍慢...）")    # 提示（分词是逐句的，确实慢）
    token_lens = []                                    # 收集 token 长度
    for r in rows[:5000]:  # 取前 5000 条避免太慢         # 只取前 5000 条做抽样统计
        t1 = len(tokenizer.tokenize(r["sentence1"]))    # 句 1 的 token 数
        t2 = len(tokenizer.tokenize(r["sentence2"]))    # 句 2 的 token 数
        token_lens.extend([t1, t2])                     # 两个都收

    fp = _get_font()                                    # 中文字体
    fig, ax = plt.subplots(figsize=(8, 4))              # 画布
    ax.hist(token_lens, bins=40, color="#4CAF50", alpha=0.8, density=True)  # 绿色直方图
    ax.axvline(np.mean(token_lens), color="red", linestyle="-",
               label=f"均值={np.mean(token_lens):.1f}")   # 均值竖线
    ax.axvline(np.percentile(token_lens, 95), color="orange", linestyle="--",
               label=f"P95={np.percentile(token_lens, 95):.0f}")  # P95 竖线（max_length 依据）
    ax.set_xlabel("单句 Token 数（不含 [CLS]/[SEP]）", fontproperties=fp)  # x 轴
    ax.set_ylabel("密度", fontproperties=fp)            # y 轴
    ax.set_title("单句 Token 数分布（train 前 5000 条）", fontproperties=fp)  # 标题
    ax.legend(prop=fp)                                  # 图例
    fig.tight_layout()                                  # 布局

    save_path = output_dir / "token_length_distribution.png"  # 保存路径
    fig.savefig(save_path, dpi=150)                    # 存图
    plt.close(fig)                                     # 关图
    print(f"  图表已保存 → {save_path}")                 # 提示
    print(f"  Token 长度：均值={np.mean(token_lens):.1f}  "
          f"P95={np.percentile(token_lens, 95):.0f}  最长={max(token_lens)}")  # 控制台摘要统计


# ── 图 4：正/负样本长度差（捷径检测） ────────────────────────────────────

def plot_length_diff(rows, output_dir):
    """
    检测长度差是否可作为"判别捷径"：
    若正样本句子长度差 << 负样本，则模型可能学到"长度接近 → 相似"这个捷径。
    教学价值：启发学生思考数据集偏差对模型泛化的影响。
    """
    pos_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 1]      # 正样本的 |len(s1)-len(s2)|
    neg_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 0]      # 负样本的长度差

    fp = _get_font()                                    # 中文字体
    fig, ax = plt.subplots(figsize=(8, 4))              # 画布
    ax.hist(pos_diffs, bins=30, alpha=0.6, label=f"正样本 均值={np.mean(pos_diffs):.1f}",
            color="#2196F3", density=True)             # 正样本长度差分布
    ax.hist(neg_diffs, bins=30, alpha=0.6, label=f"负样本 均值={np.mean(neg_diffs):.1f}",
            color="#F44336", density=True)             # 负样本长度差分布
    ax.set_xlabel("|len(s1) - len(s2)| 字符数", fontproperties=fp)  # x 轴
    ax.set_ylabel("密度", fontproperties=fp)            # y 轴
    ax.set_title("正/负样本句子长度差分布（length bias 检测）", fontproperties=fp)  # 标题
    ax.legend(prop=fp)                                  # 图例
    fig.tight_layout()                                  # 布局

    save_path = output_dir / "length_diff_distribution.png"  # 保存路径
    fig.savefig(save_path, dpi=150)                    # 存图
    plt.close(fig)                                     # 关图
    print(f"  图表已保存 → {save_path}")                 # 提示
    print(f"  长度差：正样本均值={np.mean(pos_diffs):.1f}  负样本均值={np.mean(neg_diffs):.1f}")  # 打印两组均值
    if np.mean(pos_diffs) < np.mean(neg_diffs) * 0.7:  # 正样本均值低于负样本 70%
        print("  ⚠️  正样本长度差明显更小，存在 length bias 风险")  # 警告存在捷径风险
    else:
        print("  ✓ 正/负样本长度差接近，无明显 length bias")  # 否则认为安全


# ── 控制台统计输出 ────────────────────────────────────────────────────────

def print_stats(name, rows):
    labels  = [r["label"] for r in rows]               # 全部标签
    cnt     = Counter(labels)                          # 计数
    s1_lens = [len(r["sentence1"]) for r in rows]      # s1 长度
    s2_lens = [len(r["sentence2"]) for r in rows]      # s2 长度
    all_lens = s1_lens + s2_lens                       # 合并

    print(f"\n{'='*50}")                               # 分隔线
    print(f"【{name}】共 {len(rows):,} 条")             # split 名与规模
    print(f"{'='*50}")                                 # 分隔线

    n_pos = cnt.get(1, 0)                              # 正例数
    n_neg = cnt.get(0, 0)                              # 负例数
    n_unlabeled = sum(v for k, v in cnt.items() if k not in (0, 1))  # 既非 0 也非 1 的（如 -1）
    if n_unlabeled:                                    # 存在未公开标签
        # AFQMC 竞赛测试集标签为 -1（未公开），不计入正/负统计
        print(f"  标签未公开（CLUE 竞赛格式）: {n_unlabeled:>6,} 条  —— 仅供参考，不用于评估")  # 单独提示
    else:
        print(f"  正样本（相似）  : {n_pos:>6,} ({n_pos/len(rows)*100:.1f}%)")  # 正例数与占比
        print(f"  负样本（不相似）: {n_neg:>6,} ({n_neg/len(rows)*100:.1f}%)")  # 负例数与占比
        print(f"  不均衡比 (neg/pos): {n_neg/max(n_pos, 1):.1f}x")   # 不均衡倍数（max 防除零）
    print(f"  句子字符长度 — 均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")  # 长度统计
    print("  示例正样本：")                             # 标题
    for r in [r for r in rows if r["label"] == 1][:2]: # 前两个正样本
        print(f"    ✓  {r['sentence1']!r}  ||  {r['sentence2']!r}")  # 展示句对（!r 保留引号）
    print("  示例负样本：")                             # 标题
    for r in [r for r in rows if r["label"] == 0][:2]: # 前两个负样本
        print(f"    ✗  {r['sentence1']!r}  ||  {r['sentence2']!r}")  # 展示句对


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    """解析数据探索脚本命令行的各项参数。"""
    parser = argparse.ArgumentParser(description="AFQMC 数据集探索")  # 解析器
    parser.add_argument("--data_dir",   default=str(DATA_DIR),   type=Path)  # 数据目录
    parser.add_argument("--bert_path",  default=str(BERT_PATH),  type=str)    # 分词器路径
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=Path)  # 图表目录
    parser.add_argument("--skip_token", action="store_true", help="跳过 Token 长度分析（较慢）")  # 开关：跳过慢步骤
    return parser.parse_args()                          # 返回解析结果


def main():
    """探索主流程：打印各 split 统计并生成标签/长度/Token/长度差四类可视化图表。"""
    args = parse_args()                                 # 解析参数
    args.output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    splits = {}                                         # split 名 → 样本行
    for split in ["train", "validation", "test"]:        # 三个标准 split
        path = args.data_dir / f"{split}.jsonl"         # 拼路径
        if path.exists():                               # 存在才加载（缺某个也能继续跑）
            splits[split] = load_jsonl(path)             # 加载

    for name, rows in splits.items():                    # 逐 split 打印控制台统计
        print_stats(name, rows)                          # 统计输出

    train_rows = splits.get("train", [])                 # 拿 train 行
    if not train_rows:                                  # 没有 train
        print("train.jsonl 不存在，请先运行 download_data.py")  # 提示先下载数据
        return                                          # 直接退出

    print(f"\n{'='*50}")                                 # 分隔线
    print("生成可视化图表...")                            # 阶段提示

    plot_label_distribution(splits, args.output_dir)     # 图 1：各 split 标签分布
    plot_char_length(train_rows, args.output_dir)        # 图 2：train 句长分布
    plot_length_diff(train_rows, args.output_dir)        # 图 4：length bias 检测

    if not args.skip_token:                              # 未跳过 token 分析
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)  # 加载分词器
        plot_token_length(train_rows, tokenizer, args.output_dir)   # 图 3：token 长度分布

    print(f"\n所有图表已保存至 → {args.output_dir}")       # 最终提示


if __name__ == "__main__":
    main()                                              # 直跑才执行探索
