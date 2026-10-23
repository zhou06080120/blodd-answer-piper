"""
BiEncoder 训练脚本（表示型文本匹配，Sentence-BERT 架构）

教学重点：
  1. CosineEmbeddingLoss — 直接优化余弦相似度：
       正例对 (label=1) → cosine_sim 趋向 +1
       负例对 (label=0) → cosine_sim 低于 margin（默认 0.3），超出才计入 loss
  2. TripletLoss — 约束三角关系：
       sim(anchor, positive) > sim(anchor, negative) + margin
       无需标签，只需正/负样本的相对关系
  3. 评估时的阈值搜索 — BiEncoder 输出连续相似度，需在 val 上搜最优分类阈值
  4. num_hidden_layers 限层 — 4 层约为全量的 1/3 时间，适合课堂快速演示

Loss 对比总结（供学生参考）：
  CosineEmbeddingLoss：
    - 直接用已有 (s1, s2, label) 对，无需额外构造
    - 负样本到一定距离后梯度归零（margin 起到边界作用）
  TripletLoss：
    - 需构造 (anchor, positive, negative) 三元组
    - 更明确地告诉模型"相对远近"关系，适合检索/排序场景
    - 负样本质量影响训练效果（随机 vs 难负样本）

使用方式：
  # CosineEmbeddingLoss（默认）
  python train_biencoder.py

  # TripletLoss
  python train_biencoder.py --loss triplet

  # 自定义参数
  python train_biencoder.py --loss cosine --pool mean --num_hidden_layers 4 --epochs 3

依赖：
  pip install torch transformers scikit-learn tqdm
"""
# ↑ 模块级 docstring：双塔模型训练入口，重点讲两种损失函数与阈值搜索

import os                                        # 设置环境变量（须在任何 numpy/torch 导入前执行）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 允许 OpenMP 重复加载：Windows 上 numpy+torch 同时用 OpenMP 会崩

import argparse                                   # 命令行参数解析
import json                                       # 训练日志落盘为 JSON
import time                                       # 每个 epoch 计时
from pathlib import Path                          # 路径常量定义

import torch                                      # 深度学习框架
import torch.nn as nn                             # 用它的梯度裁剪工具
import torch.nn.functional as F                   # 用它的两种损失函数
from torch.optim import AdamW                     # 带解耦权重衰减的 Adam 优化器（Transformer 标配）
from tqdm import tqdm                             # 进度条可视化
from transformers import BertTokenizer, get_linear_schedule_with_warmup  # 分词器与带 warmup 的线性衰减调度器

from dataset import build_pair_loaders, build_triplet_loader  # 按 loss 类型选数据集工厂
from evaluate import eval_biencoder               # 每个 epoch 末在 val 上评估（内含阈值搜索）
from model import build_biencoder                 # 双塔模型工厂

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent          # 项目根目录（src/ 的上一级）
DATA_DIR   = ROOT / "data" / "blood_pressure"     # 训练数据目录（AFQMC 血压子集）
BERT_PATH  = ROOT.parent.parent.parent / "pretrain_models" / "bert-base-chinese"  # 本地预训练权重路径
OUTPUT_DIR = ROOT / "outputs"                     # 训练产物总目录
CKPT_DIR   = OUTPUT_DIR / "checkpoints"           # checkpoint 保存目录
LOG_DIR    = OUTPUT_DIR / "logs"                   # 训练日志目录


# ── 训练一个 epoch ────────────────────────────────────────────────────────

def train_one_epoch_cosine(model, loader, optimizer, scheduler, device,
                           epoch, total_epochs, margin, grad_accum):
    """CosineEmbeddingLoss 训练 loop"""
    model.train()                                 # 切训练模式（启用 Dropout）
    total_loss, total_samples = 0.0, 0            # epoch 累计损失与样本数（用于算平均 loss）
    optimizer.zero_grad()                         # 梯度累积模式下：epoch 开头先清一次梯度

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Cosine]", leave=False)  # 进度条（leave=False 结束后清屏）
    for step, batch in enumerate(pbar):            # 遍历每个 batch
        batch_a = {                               # 塔一输入：从扁平 key 里取出 _a 后缀的三项并搬设备
            "input_ids":      batch["input_ids_a"].to(device),       # 句 1 token ids
            "attention_mask": batch["attention_mask_a"].to(device),  # 句 1 掩码
            "token_type_ids": batch["token_type_ids_a"].to(device),  # 句 1 类型标记
        }
        batch_b = {                               # 塔二输入：同上，_b 后缀
            "input_ids":      batch["input_ids_b"].to(device),       # 句 2 token ids
            "attention_mask": batch["attention_mask_b"].to(device),  # 句 2 掩码
            "token_type_ids": batch["token_type_ids_b"].to(device),  # 句 2 类型标记
        }
        labels  = batch["label"].to(device)        # 0/1 标签搬到设备

        emb_a, emb_b = model(batch_a, batch_b)     # 双塔前向：两路归一化句向量 [B, H]

        # label 0→-1，1→+1（cosine_embedding_loss 要求 target ∈ {-1, +1}）
        cos_target = (labels.float() * 2 - 1)      # 数学映射：0→-1（推远），1→+1（拉近）
        loss = F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=margin)  # 按余弦嵌入损失计算：正对拉向 +1，负对压到 margin 以下

        (loss / grad_accum).backward()             # 反传；除以累积步数保证多步累加后等于完整 batch 的梯度
        if (step + 1) % grad_accum == 0:            # 攒够 grad_accum 个 step 才真正更新一次
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪：L2 范数封顶 1.0，防 BERT 微调常见的梯度爆炸
            optimizer.step()                       # 优化器更新权重
            scheduler.step()                       # 学习率调度器推进（warmup→线性衰减）
            optimizer.zero_grad()                 # 更新完清空梯度，进入下个累积周期

        total_loss    += loss.item() * labels.size(0)  # 累计加权损失（item() 取标量脱离计算图）
        total_samples += labels.size(0)            # 累计样本数
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")  # 进度条实时显示运行平均 loss

    return total_loss / total_samples             # 返回本 epoch 平均损失


def train_one_epoch_triplet(model, loader, optimizer, scheduler, device,
                            epoch, total_epochs, margin, grad_accum):
    """TripletLoss 训练 loop"""
    model.train()                                 # 训练模式
    total_loss, total_samples = 0.0, 0            # 累计统计
    optimizer.zero_grad()                         # epoch 开头清梯度

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Triplet]", leave=False)  # 三元组训练进度条
    for step, batch in enumerate(pbar):           # 遍历三元组 batch
        enc_a = {                                 # 锚点句输入
            "input_ids":      batch["input_ids_a"].to(device),       # 锚点 token ids
            "attention_mask": batch["attention_mask_a"].to(device),  # 锚点掩码
            "token_type_ids": batch["token_type_ids_a"].to(device),  # 锚点类型标记
        }
        enc_p = {                                 # 正例句输入
            "input_ids":      batch["input_ids_p"].to(device),       # 正例 token ids
            "attention_mask": batch["attention_mask_p"].to(device),  # 正例掩码
            "token_type_ids": batch["token_type_ids_p"].to(device),  # 正例类型标记
        }
        enc_n = {                                 # 负例句输入
            "input_ids":      batch["input_ids_n"].to(device),       # 负例 token ids
            "attention_mask": batch["attention_mask_n"].to(device),  # 负例掩码
            "token_type_ids": batch["token_type_ids_n"].to(device),  # 负例类型标记
        }

        emb_a = model.encode(**enc_a)              # 分别对三句话编码（** 解包字典成 encode 的关键字参数）
        emb_p = model.encode(**enc_p)              # 正例向量
        emb_n = model.encode(**enc_n)              # 负例向量

        # triplet_margin_loss 默认用欧氏距离；
        # 由于 encode() 已 L2 归一化，向量在单位球上，欧氏距离与余弦距离单调相关
        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=margin)  # 约束 d(a,p) + margin < d(a,n)，违反 margin 才产生损失

        (loss / grad_accum).backward()             # 缩放反传（梯度累积）
        if (step + 1) % grad_accum == 0:           # 累积满一轮
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()                       # 更新权重
            scheduler.step()                       # 推进学习率
            optimizer.zero_grad()                 # 清梯度

        bs = emb_a.size(0)                         # 本 batch 实际样本数（最后一批可能不足）
        total_loss    += loss.item() * bs          # 累计损失
        total_samples += bs                        # 累计样本
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")  # 显示运行平均 loss

    return total_loss / total_samples             # 返回 epoch 平均损失


# ── 主训练流程 ────────────────────────────────────────────────────────────

def main():
    """BiEncoder 训练主流程：构建数据与模型，逐 epoch 训练/评估并保存最优 checkpoint。"""
    args = parse_args()                            # 解析命令行超参

    CKPT_DIR.mkdir(parents=True, exist_ok=True)    # 确保 checkpoint 目录存在（含父级，已存在不报错）
    LOG_DIR.mkdir(parents=True, exist_ok=True)     # 确保日志目录存在

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 GPU 用 GPU，否则 CPU
    print(f"设备: {device}")                       # 打印实际设备
    print(f"Loss 类型: {args.loss}  池化策略: {args.pool}  "
          f"BERT 层数: {args.num_hidden_layers}  Epochs: {args.epochs}")  # 打印关键超参组合，便于实验管理

    # ── Tokenizer & DataLoader ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)  # 加载与预训练一致的中文分词器
    print("\nDataLoader 构建中...")                 # 阶段提示

    if args.loss == "cosine":                      # 句对损失 → 用 PairDataset
        train_loader, val_loader, _ = build_pair_loaders(  # 构建训练/验证 loader（test 用不上，以 _ 丢弃）
            args.data_dir, tokenizer,
            max_length=args.max_length, batch_size=args.batch_size,
        )
    else:  # triplet                               # 三元组损失 → 用 TripletDataset
        train_loader, val_loader = build_triplet_loader(  # 训练是三元组，验证仍是句对
            args.data_dir, tokenizer,
            max_length=args.max_length, batch_size=args.batch_size,
        )

    # ── 模型 ─────────────────────────────────────────────────────────────
    print("\n构建模型...")                          # 阶段提示
    model = build_biencoder(                       # 实例化双塔模型并搬上设备
        bert_path=args.bert_path,                  # 预训练权重路径
        pool=args.pool,                            # 池化策略
        num_hidden_layers=args.num_hidden_layers,  # 裁剪层数
    ).to(device)

    # ── 分层学习率 ────────────────────────────────────────────────────────
    # BERT 骨干用较小 lr，防止预训练知识被过度破坏
    bert_params = list(model.bert.parameters())     # 收集 BERT 骨干全部参数
    head_params = list(model.dropout.parameters())  # 收集头部参数（dropout 无参不影响，保持结构一致）

    optimizer = AdamW([                            # 参数组形式：不同部件配不同学习率
        {"params": bert_params, "lr": args.lr},     # 骨干用基准 lr（如 2e-5）
        {"params": head_params, "lr": args.lr * args.head_lr_mult},  # 头部从零学起，乘 5 倍加速收敛
    ], weight_decay=0.01)                           # 统一权重衰减防过拟合

    total_steps  = len(train_loader) * args.epochs // args.grad_accum  # 梯度累积下的真实更新次数
    warmup_steps = int(total_steps * args.warmup_ratio)  # 前 10% 步数做 warmup（学习率从 0 线性爬升）
    scheduler = get_linear_schedule_with_warmup(   # 构建调度器：warmup 后线性衰减到 0
        optimizer,                                 # 绑定优化器
        num_warmup_steps=warmup_steps,             # 预热步数
        num_training_steps=total_steps,            # 总步数（决定衰减斜率）
    )
    print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")  # 打印调度信息

    # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_name = f"biencoder_{args.loss}_best.pt"   # checkpoint 文件名带 loss 类型（两种实验互不覆盖）
    ckpt_path = CKPT_DIR / ckpt_name               # 完整保存路径
    best_val_f1 = 0.0                              # 迄今最优 val F1（选模型依据）
    log_records = []                               # 每个 epoch 的指标记录

    for epoch in range(1, args.epochs + 1):        # 从 1 开始计数的 epoch 循环
        t0 = time.time()                            # epoch 计时开始

        if args.loss == "cosine":                  # 按 loss 类型分派训练循环
            train_loss = train_one_epoch_cosine(   # 余弦损失训练一轮
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )
        else:
            train_loss = train_one_epoch_triplet(  # 三元组损失训练一轮
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )

        # 每个 epoch 末评估 val（BiEncoder：相似度 + 阈值搜索）
        val_metrics = eval_biencoder(model, val_loader, device)  # 返回 accuracy/f1/threshold 等指标
        elapsed = time.time() - t0                  # 本 epoch 总耗时

        val_acc = val_metrics["accuracy"]           # 验证集准确率
        val_f1  = val_metrics["f1"]                 # 验证集 F1（正类是"匹配"）
        val_thr = val_metrics["threshold"]          # 阈值搜索选出的最优相似度阈值
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} threshold={val_thr:.2f} | "
              f"{elapsed:.0f}s")                    # 单行汇总本 epoch 全部关键指标

        log_records.append({                        # 本轮指标写入日志结构
            "epoch": epoch, "train_loss": train_loss,  # 轮次与训练损失
            "val_acc": val_acc, "val_f1": val_f1,     # 验证指标
            "threshold": val_thr, "elapsed_s": elapsed,  # 阈值与耗时
        })

        if val_f1 > best_val_f1:                    # F1 创新高 → 保存 checkpoint（早停式留最优）
            best_val_f1 = val_f1                    # 更新最优记录
            torch.save({                            # 打包保存模型与元信息
                "epoch":      epoch,                # 最优出现在第几轮
                "state_dict": model.state_dict(),   # 模型权重（ONNX 导出/服务加载都靠它）
                "threshold":  val_thr,              # 搜出的最优阈值（部署时直接用）
                "val_acc":    val_acc,              # 验证准确率
                "val_f1":     val_f1,               # 验证 F1
                "args":       vars(args),           # 全部超参转 dict（onnx_runtime 工厂按它恢复结构）
            }, ckpt_path)                           # 覆盖写最优 checkpoint
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")  # 提示保存成功

    # ── 训练完成，保存日志 ────────────────────────────────────────────────
    log_path = LOG_DIR / f"biencoder_{args.loss}_log.json"  # 日志文件名带 loss 类型
    with open(log_path, "w", encoding="utf-8") as f:  # UTF-8 打开（日志含中文）
        json.dump(log_records, f, ensure_ascii=False, indent=2)  # 中文原样输出、缩进 2 格便于阅读
    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")  # 汇报最优成绩
    print(f"训练日志 → {log_path}")                 # 指出日志位置
    print(f"最优 checkpoint → {ckpt_path}")          # 指出模型位置
    print(f"\n运行评估：python evaluate.py --model_type biencoder --ckpt {ckpt_path}")  # 提示下一步评估命令


# ── 参数解析 ──────────────────────────────────────────────────────────────

def parse_args():
    """解析 BiEncoder 训练命令行的各项参数。"""
    parser = argparse.ArgumentParser(description="BiEncoder 训练（表示型文本匹配）")  # 参数解析器（--help 展示描述）
    parser.add_argument("--bert_path",         default=str(BERT_PATH),   type=str)  # 预训练模型路径
    parser.add_argument("--data_dir",          default=str(DATA_DIR),    type=str)  # 数据目录
    parser.add_argument("--loss",              default="cosine",
                        choices=["cosine", "triplet"],
                        help="训练损失类型：cosine（CosineEmbeddingLoss）或 triplet（TripletLoss）")  # 限定二选一
    parser.add_argument("--pool",              default="mean",
                        choices=["cls", "mean", "max"],
                        help="句向量池化策略（Sentence-BERT 论文推荐 mean）")  # 池化策略三选一
    parser.add_argument("--num_hidden_layers", default=4,    type=int,
                        help="BERT Transformer 层数（默认 4 层快速验证；全量 12 层留给学生）")  # 裁剪层数
    parser.add_argument("--epochs",            default=3,    type=int)  # 训练轮数
    parser.add_argument("--batch_size",        default=32,   type=int)  # 批大小
    parser.add_argument("--max_length",        default=64,   type=int,  help="单句最大 token 数")  # 单句截断长度
    parser.add_argument("--lr",                default=2e-5, type=float, help="BERT 层学习率")  # 骨干学习率
    parser.add_argument("--head_lr_mult",      default=5.0,  type=float, help="dropout 层学习率倍数")  # 头部学习率倍数
    parser.add_argument("--warmup_ratio",      default=0.1,  type=float)  # 预热比例
    parser.add_argument("--grad_accum",        default=1,    type=int)  # 梯度累积步数
    parser.add_argument("--margin",            default=0.3,  type=float,
                        help="CosineEmbeddingLoss 的 margin；TripletLoss 的 margin 同参数")  # 两种损失共用的间隔超参
    return parser.parse_args()                     # 解析并返回 Namespace 对象


if __name__ == "__main__":
    main()                                         # 作为脚本直接运行时才执行训练；被 import 时不触发
