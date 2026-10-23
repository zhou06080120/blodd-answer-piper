"""
BiEncoder ONNX 导出 + INT8 动态量化（需求 3 落地脚本）

流程：
  1. 加载 checkpoint 的 BiEncoder → 导出 FP32 ONNX（静态图，seq=64 固定、batch 动态）
  2. INT8 动态量化（CPU 友好，无需标定集）
  3. 与原 PyTorch 输出做余弦一致性校验（8 句样例，阈值 0.9999）

使用方式：
  python export_onnx.py --ckpt outputs/checkpoints/biencoder_hardneg_best.pt
  python export_onnx.py --ckpt outputs/checkpoints/biencoder_cosine_best.pt --precision fp32
"""
# ↑ 模块级 docstring：把训练产物转成线上部署用的 ONNX（含量化和一致性校验）

import argparse                                   # CLI 参数
import sys                                        # 修改模块搜索路径
from pathlib import Path                          # 路径处理

ROOT = Path(__file__).parent                      # 项目根目录
sys.path.insert(0, str(ROOT / "src"))             # 把 src/ 塞进 sys.path，才能 import onnx_runtime 等模块

from onnx_runtime import export_biencoder_onnx, quantize_dynamic_int8  # noqa: E402  # 导出与量化函数（路径修改在导入前，E402 告警可忽略）

ONNX_DIR = ROOT / "outputs" / "onnx"              # ONNX 模型输出目录
BERT_PATH = r"F:\aaa123\pretrain_models\bert-base-chinese"  # 预训练权重绝对路径


def verify_parity(onnx_path, model, tokenizer, device, max_length, n=8):
    """ONNX 与 PyTorch 输出余弦一致性校验"""
    import numpy as np                              # 向量点积算余弦
    import torch                                   # PyTorch 前向
    from onnx_runtime import OnnxEncoder           # ORT 编码器（读刚导出的模型）

    samples = ["血压多少算正常？", "血压计怎么正确测量血压？", "电子血压计需要多久校准一次？",
               "收缩压和舒张压是什么意思？", "腕式血压计准吗？", "高血压一定要吃药吗？",
               "血压计显示错误代码怎么办？", "饭后能马上测血压吗？"][:n]  # 真实业务样例句（截前 n 条）
    ort_enc = OnnxEncoder(onnx_path, tokenizer, max_length=max_length)  # ONNX 编码器
    a = ort_enc.encode(samples)                     # ONNX 侧输出向量 [n, H]（已归一化）

    from dataset import encode_single              # 复用单句编码工具
    with torch.no_grad():                           # PyTorch 侧推理无梯度
        rows = [encode_single(tokenizer, t, max_length) for t in samples]  # 逐句编码成张量 dict
        b = model.encode(                           # PyTorch 侧输出
            torch.stack([r["input_ids"] for r in rows]).to(device),       # 堆叠成 batch → 设备
            torch.stack([r["attention_mask"] for r in rows]).to(device),  # 同上
            torch.stack([r["token_type_ids"] for r in rows]).to(device),   # 同上
        ).detach().cpu().numpy()                    # 脱离计算图 → CPU → numpy

    coss = [float(np.dot(a[i], b[i])) for i in range(n)]  # 逐样例点积（两侧向量都已 L2 归一化 → 点积即余弦）
    print("  样例余弦一致性: " + ", ".join(f"{c:.6f}" for c in coss))  # 逐条打印（应接近 1.000000）
    return min(coss)                                # 返回最差值（最坏情况的一致性）


def main():
    """导出主流程：加载 checkpoint → 导出 FP32 ONNX → 校验一致性 → （可选）INT8 量化。"""
    args = parse_args()                             # 解析参数
    import torch                                   # 加载权重用
    from transformers import BertTokenizer         # 分词器
    from model import build_biencoder              # 模型工厂

    ONNX_DIR.mkdir(parents=True, exist_ok=True)     # 确保输出目录存在
    device = torch.device("cpu")   # 导出与一致性校验统一 CPU（与 dummy 输入同设备）  # 固定 CPU：导出时设备和 dummy 输入必须一致
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)  # 加载 checkpoint
    saved = ckpt.get("args", {})                    # 训练超参
    bert_path = saved.get("bert_path") or args.bert_path  # 优先用 checkpoint 记录的路径（保证结构一致）
    max_length = saved.get("max_length", 64) or 64  # 编码长度同样以训练记录优先

    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 分词器
    model = build_biencoder(bert_path=bert_path, pool=saved.get("pool", "mean"),
                            num_hidden_layers=saved.get("num_hidden_layers")).to(device)  # 按超参重建模型
    model.load_state_dict(ckpt["state_dict"])       # 载入训练权重
    model.eval()                                    # 评估模式（关 Dropout，导出确定性图）
    print(f"checkpoint: {Path(args.ckpt).name} | pool={saved.get('pool')} "
          f"layers={saved.get('num_hidden_layers')} max_length={max_length} | 设备: {device}")  # 打印导出配置

    stem = Path(args.ckpt).stem                     # 权重文件去后缀名（命名 ONNX 用）
    fp32_path = ONNX_DIR / f"{stem}_fp32.onnx"      # FP32 模型路径
    print(f"导出 FP32 ONNX → {fp32_path}")           # 进度提示
    export_biencoder_onnx(model, tokenizer, fp32_path, max_length=max_length,
                          opset=args.opset)         # 执行导出（内部做图追踪）
    print("FP32 一致性:", end=" ")                    # 提示（不换行）
    verify_parity(fp32_path, model, tokenizer, device, max_length)  # FP32 应近乎 1.0

    target = fp32_path                              # 默认推荐产物是 FP32
    if args.precision == "int8":                    # 需要 int8 档位
        int8_path = ONNX_DIR / f"{stem}_int8.onnx"  # INT8 模型路径
        print(f"INT8 动态量化 → {int8_path}")         # 进度提示
        quantize_dynamic_int8(fp32_path, int8_path)  # 在 FP32 基础上量化（无需标定数据）
        print("INT8 一致性:", end=" ")                # 提示
        verify_parity(int8_path, model, tokenizer, device, max_length)  # INT8 会略低于 FP32，但也应 >0.9999
        target = int8_path                          # 推荐产物改为 INT8
    print(f"\n完成。线上推荐后端: ONNX-{args.precision} → {target}")  # 最终提示线上加载哪个文件


def parse_args():
    """解析 ONNX 导出与量化命令行的各项参数。"""
    p = argparse.ArgumentParser(description="BiEncoder ONNX 导出 + 量化")  # 解析器
    p.add_argument("--ckpt", default=str(ROOT / "outputs" / "checkpoints" /
                                        "biencoder_hardneg_best.pt"), type=str)  # 默认难负例权重
    p.add_argument("--bert_path", default=BERT_PATH, type=str)  # 预训练路径
    p.add_argument("--precision", default="int8", choices=["int8", "fp32"])  # 目标精度档
    p.add_argument("--opset", default=17, type=int)  # ONNX 算子集版本
    return p.parse_args()                           # 返回解析结果


if __name__ == "__main__":
    main()                                          # 直跑才执行导出
