"""
CrossEncoder 训练脚本（交互型文本匹配）

教学重点：
  1. CrossEncoder vs BiEncoder 的关键差异：
       CrossEncoder 让两句在 BERT 每一层都充分交互（Self-Attention 跨句），
       表达能力更强但无法预计算，不适合向量检索；
       BiEncoder 两句独立编码，可预计算向量，适合大规模检索（如 RAG Recall）
  2. 输入格式：[CLS] s1 [SEP] s2 [SEP]，token_type_ids 区分两段
       这是 BERT 原始预训练任务（NSP，Next Sentence Prediction）的格式
  3. CrossEncoder 评估与普通分类完全相同，无需阈值搜索（argmax 即为预测）

使用方式：
  # 默认参数（4 层 BERT，3 epoch）
  python train_crossencoder.py

  # 自定义参数
  python train_crossencoder.py --num_hidden_layers 6 --epochs 5 --batch_size 16

依赖：
  pip install torch transformers scikit-learn tqdm
"""
# ↑ 模块级 docstring：交叉编码器训练入口，说明它与 BiEncoder 的分工（精排 vs 召回）

import os                                        # 环境变量设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 Windows 上 OpenMP 库重复加载崩溃

import argparse                                   # 命令行参数解析
import json                                       # 训练日志 JSON 落盘
import time                                       # epoch 计时
from pathlib import Path                          # 路径拼接

import torch                                      # 深度学习框架
import torch.nn as nn                             # 损失函数与梯度裁剪
from torch.optim import AdamW                     # 优化器
from tqdm import tqdm                             # 进度条
from transformers import BertTokenizer, get_linear_schedule_with_warmup  # 分词器与学习率调度

from dataset import build_crossencoder_loaders    # 交互型数据集工厂
from evaluate import eval_crossencoder            # 评估函数（标准分类指标）
from model import build_crossencoder              # CrossEncoder 工厂

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent          # 项目根目录
DATA_DIR   = ROOT / "data" / "blood_pressure"     # 数据目录
BERT_PATH  = ROOT.parent.parent.parent / "pretrain_models" / "bert-base-chinese"  # 预训练权重路径
OUTPUT_DIR = ROOT / "outputs"                     # 产物目录
CKPT_DIR   = OUTPUT_DIR / "checkpoints"           # checkpoint 目录
LOG_DIR    = OUTPUT_DIR / "logs"                   # 日志目录


# ── 训练一个 epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs, grad_accum):
    model.train()                                 # 训练模式（Dropout 生效）
    total_loss, total_correct, total_samples = 0.0, 0, 0  # 累计损失/正确数/样本数
    optimizer.zero_grad()                         # epoch 开头清梯度（梯度累积模式）

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [CrossEncoder]", leave=False)  # 进度条
    for step, batch in enumerate(pbar):            # 遍历 batch
        input_ids      = batch["input_ids"].to(device)        # 拼接序列 token ids → 设备
        attention_mask = batch["attention_mask"].to(device)   # 掩码 → 设备
        token_type_ids = batch["token_type_ids"].to(device)   # 句界标记 → 设备
        labels         = batch["label"].to(device)           # 0/1 标签 → 设备

        logits = model(input_ids, attention_mask, token_type_ids)  # 前向：输出 [B, 2] 分类分数
        loss   = criterion(logits, labels)        # CrossEntropyLoss 内部自带 softmax+取标签列

        (loss / grad_accum).backward()             # 缩放反传（累积梯度）
        if (step + 1) % grad_accum == 0:            # 累积到指定步数
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防爆
            optimizer.step()                       # 更新权重
            scheduler.step()                       # 学习率推进
            optimizer.zero_grad()                 # 清梯度

        preds = logits.argmax(dim=-1)              # 取每行最大值的下标作为预测类别（0 或 1）
        total_loss    += loss.item() * labels.size(0)   # 累计加权损失
        total_correct += (preds == labels).sum().item()  # 累计预测正确数
        total_samples += labels.size(0)                 # 累计样本数
        pbar.set_postfix(                          # 进度条实时显示两个运行指标
            loss=f"{total_loss / total_samples:.4f}",    # 运行平均 loss
            acc=f"{total_correct / total_samples:.4f}",  # 运行平均训练准确率
        )

    return total_loss / total_samples, total_correct / total_samples  # 返回 (平均损失, 训练准确率)


# ── 主训练流程 ────────────────────────────────────────────────────────────

def main():
    """CrossEncoder 训练主流程：构建数据与模型，逐 epoch 训练/评估并保存最优 checkpoint。"""
    args = parse_args()                            # 解析超参

    CKPT_DIR.mkdir(parents=True, exist_ok=True)    # 确保 checkpoint 目录存在
    LOG_DIR.mkdir(parents=True, exist_ok=True)     # 确保日志目录存在

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选设备
    print(f"设备: {device}")                       # 打印设备
    print(f"BERT 层数: {args.num_hidden_layers}  Epochs: {args.epochs}  "
          f"Batch size: {args.batch_size}")        # 打印关键超参

    # ── Tokenizer & DataLoader ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)  # 加载分词器
    print("\nDataLoader 构建中...")                 # 阶段提示
    train_loader, val_loader, _ = build_crossencoder_loaders(  # 构建三 loader（test 此处不用，丢弃）
        args.data_dir, tokenizer,
        max_length=args.max_length, batch_size=args.batch_size,
        train_file=args.train_file,                # 训练文件可替换成增强集（如 ce_train.jsonl）
    )

    # ── 模型 ─────────────────────────────────────────────────────────────
    print("\n构建模型...")                          # 阶段提示
    model = build_crossencoder(                    # 实例化交叉编码器
        bert_path=args.bert_path,                 # 预训练权重
        num_hidden_layers=args.num_hidden_layers,  # 裁剪层数
    ).to(device)                                   # 搬上计算设备

    # ── 分层学习率 ────────────────────────────────────────────────────────
    bert_params = list(model.bert.parameters())     # BERT 骨干参数组
    head_params = (list(model.dropout.parameters()) +
                   list(model.classifier.parameters()))  # 分类头参数组（dropout+linear 合并）

    optimizer = AdamW([                            # 两组不同学习率
        {"params": bert_params, "lr": args.lr},     # 骨干小 lr 微调
        {"params": head_params, "lr": args.lr * args.head_lr_mult},  # 新初始化的头给更大 lr
    ], weight_decay=0.01)                          # 权重衰减

    total_steps  = len(train_loader) * args.epochs // args.grad_accum  # 真实更新步数
    warmup_steps = int(total_steps * args.warmup_ratio)  # warmup 步数
    scheduler = get_linear_schedule_with_warmup(   # warmup + 线性衰减调度
        optimizer,                                 # 绑定优化器
        num_warmup_steps=warmup_steps,             # 预热步数
        num_training_steps=total_steps,            # 总步数
    )
    print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")  # 打印调度信息

    # AFQMC 正负比约 31%:69%，CrossEntropyLoss 默认不加权
    # 如需加权，可参考 bert_text_classification 的 compute_class_weight 做法
    criterion = nn.CrossEntropyLoss()               # 普通交叉熵即可（类别不平衡程度可接受）

    # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_path   = CKPT_DIR / "crossencoder_best.pt"  # 固定 checkpoint 文件名（精排器全局唯一）
    best_val_f1 = 0.0                               # 最优 val F1 记录
    log_records = []                                # 日志记录列表

    for epoch in range(1, args.epochs + 1):         # epoch 循环（1 起）
        t0 = time.time()                            # 计时开始

        train_loss, train_acc = train_one_epoch(    # 训练一轮，拿回 loss 与训练准确率
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum,
        )

        val_metrics = eval_crossencoder(model, val_loader, device)  # val 评估（argmax 即预测，无阈值搜索）
        elapsed = time.time() - t0                  # 本轮耗时

        val_acc = val_metrics["accuracy"]           # 验证准确率
        val_f1  = val_metrics["f1"]                 # 验证 F1
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")                    # 汇总打印本轮指标

        log_records.append({                        # 追加日志记录
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,  # 轮次与训练指标
            "val_acc": val_acc, "val_f1": val_f1, "elapsed_s": elapsed,  # 验证指标与耗时
        })

        if val_f1 > best_val_f1:                    # F1 提升才覆盖保存
            best_val_f1 = val_f1                    # 更新最优值
            torch.save({                            # 打包保存
                "epoch":      epoch,                # 最优轮次
                "state_dict": model.state_dict(),   # 模型权重
                "val_acc":    val_acc,              # 验证准确率
                "val_f1":     val_f1,               # 验证 F1
                "args":       vars(args),           # 超参记录（部署侧据此重建结构）
            }, ckpt_path)                           # 覆盖写
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")  # 保存提示

    # ── 训练完成，保存日志 ────────────────────────────────────────────────
    log_path = LOG_DIR / "crossencoder_log.json"    # 日志路径（固定名）
    with open(log_path, "w", encoding="utf-8") as f:  # UTF-8 写文件
        json.dump(log_records, f, ensure_ascii=False, indent=2)  # 缩进 2 格、中文原样
    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")  # 汇报最优成绩
    print(f"训练日志 → {log_path}")                 # 日志位置
    print(f"最优 checkpoint → {ckpt_path}")          # 模型位置
    print(f"\n运行评估：python evaluate.py --model_type crossencoder --ckpt {ckpt_path}")  # 下一步提示


# ── 参数解析 ──────────────────────────────────────────────────────────────

def parse_args():
    """解析 CrossEncoder 训练命令行的各项参数。"""
    parser = argparse.ArgumentParser(description="CrossEncoder 训练（交互型文本匹配）")  # 解析器（--help 描述）
    parser.add_argument("--bert_path",         default=str(BERT_PATH),   type=str)  # 预训练路径
    parser.add_argument("--data_dir",          default=str(DATA_DIR),    type=str)  # 数据目录
    parser.add_argument("--train_file",        default="train.jsonl",    type=str,
                        help="训练文件名（可用增强集 ce_train.jsonl）")  # 可替换训练集
    parser.add_argument("--num_hidden_layers", default=4,    type=int,
                        help="BERT Transformer 层数（默认 4 层；全量 12 层留给学生自行实验）")  # 裁剪层数
    parser.add_argument("--epochs",            default=3,    type=int)  # 训练轮数
    parser.add_argument("--batch_size",        default=32,   type=int)  # 批大小
    parser.add_argument("--max_length",        default=128,  type=int,
                        help="句对总最大 token 数（两句拼接，建议 128）")  # 拼接后总长上限
    parser.add_argument("--lr",                default=2e-5, type=float, help="BERT 层学习率")  # 骨干学习率
    parser.add_argument("--head_lr_mult",      default=5.0,  type=float, help="分类头学习率倍数")  # 头部倍数
    parser.add_argument("--warmup_ratio",      default=0.1,  type=float)  # 预热比例
    parser.add_argument("--grad_accum",        default=1,    type=int)  # 梯度累积
    return parser.parse_args()                     # 返回解析结果


if __name__ == "__main__":
    main()                                         # 脚本直跑才执行训练
