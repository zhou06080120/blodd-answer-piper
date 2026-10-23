"""
难负例微调 S-BERT（需求 1 落地：LLM 合成数据 + 硬负例挖掘 → 微调 BiEncoder）

与原 train_biencoder.py 的区别：
  1. 数据来自 synthesize_data.py 产出的扩容训练集（data/blood_pressure_expanded）：
     正例包含 LLM 生成的上千条同义改写配对；负例包含三大类——
       硬负例挖掘（全库最近邻跨意图样本）、LLM 边界负例、随机跨 FAQ 负例
  2. 支持 --init_from 从既有 checkpoint 热启动（微调而非从零训练），
     并默认沿用其 pool / num_hidden_layers / max_length 结构
  3. 训练循环、评估、阈值搜索完全复用 train_biencoder / evaluate 的成熟实现

使用方式：
  python train_sbert_hardneg.py                    # 默认 3 epoch，GPU 加速
  python train_sbert_hardneg.py --epochs 2
"""
# ↑ 模块级 docstring：在基线模型上用"更难的负例"继续微调，产出 biencoder_hardneg 权重

import argparse                                   # CLI 参数
import json                                       # 训练日志
import os                                         # 环境变量
import sys                                        # 模块路径注入
import time                                       # epoch 计时
from pathlib import Path                          # 路径处理

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 OpenMP 重复加载崩溃
import torch                                     # noqa: E402  # 深度学习框架（路径修改后才导入）
from torch.optim import AdamW                    # noqa: E402  # 优化器
from transformers import BertTokenizer, get_linear_schedule_with_warmup  # noqa: E402  # 分词器与调度器

ROOT = Path(__file__).parent                      # 项目根目录
sys.path.insert(0, str(ROOT / "src"))             # 注入 src/ 路径

from dataset import build_pair_loaders              # noqa: E402  # 数据集工厂
from evaluate import eval_biencoder                 # noqa: E402  # 评估函数
from model import build_biencoder                  # noqa: E402  # 模型工厂
from train_biencoder import train_one_epoch_cosine  # noqa: E402  # 复用成熟的训练循环

DATA_DIR = ROOT / "data" / "blood_pressure_expanded"  # 扩容训练数据目录
INIT_CKPT = ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt"  # 热启动基线权重
OUT_CKPT = ROOT / "outputs" / "checkpoints" / "biencoder_hardneg_best.pt"  # 微调产物权重
LOG_PATH = ROOT / "outputs" / "logs" / "biencoder_hardneg_log.json"  # 训练日志


def main():
    """热启动微调主流程：加载基线 checkpoint、训练并监控坍缩，保存最优模型。"""
    args = parse_args()                            # 解析超参
    torch.manual_seed(args.seed)                   # 固定全局种子（可复现）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选设备
    print(f"设备: {device} | 数据: {args.data_dir}")  # 打印环境

    init = torch.load(args.init_from, map_location=device, weights_only=False)  # 加载基线 checkpoint
    saved = init.get("args", {})                   # 基线训练超参
    bert_path = saved.get("bert_path") or args.bert_path  # 结构参数以 checkpoint 为准
    pool = saved.get("pool", "mean")                # 池化策略
    layers = saved.get("num_hidden_layers")         # 裁剪层数
    max_length = saved.get("max_length", args.max_length)  # 编码长度
    print(f"热启动: {Path(args.init_from).name} (pool={pool}, layers={layers}, "
          f"max_length={max_length}, 基线 val_f1={init.get('val_f1', 0):.4f})")  # 打印热启动信息（含基线成绩）

    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 分词器
    train_loader, val_loader, test_loader = build_pair_loaders(
        args.data_dir, tokenizer, max_length=max_length, batch_size=args.batch_size,
        train_file=args.train_file)                # 三个 loader（此处 test 也用：检索级监控）

    model = build_biencoder(bert_path=bert_path, pool=pool,
                            num_hidden_layers=layers).to(device)  # 按基线结构重建模型
    model.load_state_dict(init["state_dict"])      # 载入基线权重（热启动的关键）

    bert_params = list(model.bert.parameters())    # 骨干参数组
    head_params = list(model.dropout.parameters())  # 头部参数组
    optimizer = AdamW([                           # 分层学习率
        {"params": bert_params, "lr": args.lr},     # 骨干小步微调
        {"params": head_params, "lr": args.lr * args.head_lr_mult},  # 头部大步收敛
    ], weight_decay=0.01)                          # 权重衰减

    total_steps = len(train_loader) * args.epochs // args.grad_accum  # 实际更新步数
    warmup_steps = int(total_steps * args.warmup_ratio)  # warmup 步数
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)  # warmup+线性衰减
    print(f"训练步数: {total_steps} | warmup: {warmup_steps}")  # 打印调度信息

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)  # 确保产物目录存在
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
    best_score, best_test = 0.0, {}                # 最优分（val+test 双指标）与对应 test 结果
    log_records = []                                # 日志记录

    for epoch in range(1, args.epochs + 1):        # epoch 循环
        t0 = time.time()                           # 计时开始
        train_loss = train_one_epoch_cosine(       # 复用基线脚本的训练循环
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.margin, args.grad_accum)
        val_m = eval_biencoder(model, val_loader, device)  # val 评估（含阈值搜索）
        # 坍缩监控：正/负例平均相似度。若负例均值逼近 1.0，说明模型已退化
        # （全库向量同向），训练数据或正负比有问题，应立即停止而非继续保存。
        pos_sims = [s for s, lab in zip(val_m["similarities"], val_m["labels"]) if lab == 1]  # 正例相似度列表
        neg_sims = [s for s, lab in zip(val_m["similarities"], val_m["labels"]) if lab == 0]  # 负例相似度列表
        pos_mean = sum(pos_sims) / max(1, len(pos_sims))  # 正例均值（max 防除零）
        neg_mean = sum(neg_sims) / max(1, len(neg_sims))  # 负例均值
        collapsed = neg_mean > 0.98 or pos_mean > 0.998  # 坍缩判定：负例相似度过高即全挤到一处
        # 用 part of test 做检索级监控（test 与 val 同构，避免过拟合 val）
        test_m = eval_biencoder(model, test_loader, device, find_threshold=False,
                                threshold=val_m["threshold"])  # test 上沿用 val 阈值（不再搜索）
        elapsed = time.time() - t0                 # 本轮耗时
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_f1={val_m['f1']:.4f} acc={val_m['accuracy']:.4f} "
              f"thr={val_m['threshold']:.3f} | pos_sim_mean={pos_mean:.4f} "
              f"neg_sim_mean={neg_mean:.4f} | test_f1={test_m['f1']:.4f} "
              f"test_acc={test_m['accuracy']:.4f} | {elapsed:.0f}s")  # 一行汇总全部监控指标
        if collapsed:                              # 检测到坍缩
            print("  ✗ 检测到模型坍缩（负例相似度均值接近 1.0），终止训练！"
                  "请检查数据正负比与硬负例窗口（synthesize_data.py）。")  # 报警并指引排查方向
            break                                 # 立即终止（不保存本轮）
        log_records.append({"epoch": epoch, "train_loss": train_loss,
                            "pos_sim_mean": pos_mean, "neg_sim_mean": neg_mean,
                            **val_m, "test_f1": test_m["f1"],
                            "test_acc": test_m["accuracy"],
                            "elapsed_s": elapsed})  # 日志：含坍缩监控指标与双集成绩
        score = val_m["f1"] + test_m["f1"]         # 选型分 = 双集 F1 之和（更稳健）
        if score > best_score:                     # 创新高才保存
            best_score, best_test = score, test_m  # 更新最优
            torch.save({                           # 保存产物与元信息
                "epoch": epoch,                    # 最优轮次
                "state_dict": model.state_dict(),   # 权重
                "threshold": val_m["threshold"],   # 最优阈值（部署用）
                "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],  # val 成绩
                "test_acc": test_m["accuracy"], "test_f1": test_m["f1"],  # test 成绩
                "init_from": str(args.init_from), "hardneg": True,  # 溯源：难负例微调产物
                "args": vars(args) | {"bert_path": bert_path, "pool": pool,
                                      "num_hidden_layers": layers,
                                      "max_length": max_length},  # 超参（dict 合并，保证结构参数完整）
            }, args.out_ckpt)                      # 覆盖写最优
            print(f"  ✓ 新最优已保存 → {Path(args.out_ckpt).name}"
                  f"（val_f1+test_f1={score:.4f}）")  # 保存提示

    with open(LOG_PATH, "w", encoding="utf-8") as f:  # 打开日志文件
        json.dump(log_records, f, ensure_ascii=False, indent=2)  # 写日志
    print(f"\n完成。best score(val_f1+test_f1)={best_score:.4f} | "
          f"对应 test_f1={best_test.get('f1', 0):.4f} | 日志 → {LOG_PATH}")  # 汇报最终成绩与日志位置


def parse_args():
    """解析难负例微调命令行的各项参数。"""
    p = argparse.ArgumentParser(description="难负例微调 BiEncoder（S-BERT）")  # 解析器
    p.add_argument("--data_dir", default=str(DATA_DIR), type=str)  # 数据目录
    p.add_argument("--train_file", default="train.jsonl", type=str,
                   help="训练文件名（可用域外负例增强集 bien_train_chitchat.jsonl）")  # 可换增强训练集
    p.add_argument("--init_from", default=str(INIT_CKPT), type=str)  # 热启动权重
    p.add_argument("--out_ckpt", default=str(OUT_CKPT), type=str)  # 产物权重
    p.add_argument("--bert_path", default=r"F:\aaa123\pretrain_models\bert-base-chinese", type=str)  # 预训练路径
    p.add_argument("--epochs", default=3, type=int)  # 轮数
    p.add_argument("--batch_size", default=32, type=int)  # 批大小
    p.add_argument("--max_length", default=64, type=int)  # 单句长度
    p.add_argument("--lr", default=2e-5, type=float)  # 骨干学习率
    p.add_argument("--head_lr_mult", default=5.0, type=float)  # 头部倍数
    p.add_argument("--warmup_ratio", default=0.1, type=float)  # 预热比例
    p.add_argument("--grad_accum", default=1, type=int)  # 梯度累积
    p.add_argument("--margin", default=0.3, type=float)  # 损失间隔
    p.add_argument("--seed", default=42, type=int)  # 随机种子
    return p.parse_args()                         # 返回解析结果


if __name__ == "__main__":
    main()                                        # 直跑才执行微调
