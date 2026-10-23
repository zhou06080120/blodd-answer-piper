"""
编码延迟基准测试（需求 3 验证）：原始 PyTorch 动态图 vs ONNX 量化静态图

对比项（同一份 checkpoint、同一 tokenizer、同一批查询）：
  1. PyTorch fp32 CPU（原始动态图基线）
  2. ONNX INT8 动态量化（CPU，生产路径；缺失时自动用 ONNX FP32）
  3. PyTorch fp32 GPU（如有，仅作参考）
指标：单次编码（含 tokenize）p50 / p95 / mean，加速比与延迟压降百分比。

使用方式：
  python bench_latency.py --ckpt outputs/checkpoints/biencoder_hardneg_best.pt
产物：outputs/logs/latency_bench.json
"""
# ↑ 模块级 docstring：用数据证明"ONNX 化 + 量化"到底带来多少延迟收益

import argparse                                   # CLI 参数
import json                                       # 基准报告落盘 JSON
import statistics                                 # mean/median 统计
import sys                                        # 模块路径注入
import time                                       # perf_counter 高精度计时
from pathlib import Path                          # 路径处理

ROOT = Path(__file__).parent                      # 项目根目录
sys.path.insert(0, str(ROOT / "src"))             # 注入 src/ 到模块搜索路径

import torch  # noqa: E402                         # 设备管理（路径修改后才导入，E402 告警忽略）

from onnx_runtime import OnnxEncoder, load_dense_encoder  # noqa: E402  # 两个编码器后端 + 工厂

ONNX_DIR = ROOT / "outputs" / "onnx"               # ONNX 模型目录
BERT_PATH = r"F:\aaa123\pretrain_models\bert-base-chinese"  # 预训练路径
LOG_PATH = ROOT / "outputs" / "logs" / "latency_bench.json"  # 基准报告路径

SAMPLE_QUERIES = [                                # 基准用的真实查询池（覆盖长短句、口语化）
    "血压多少算正常？", "血压计怎么正确测量血压？", "血压机咋个用法？",
    "电子血压计需要多久校准一次？", "收缩压和舒张压是什么意思？",
    "腕式血压计准吗？", "高血压一定要吃药吗？", "血压计显示错误代码怎么办？",
    "饭后能马上测血压吗？", "左右手血压差多少正常？", "请问老人血压多少算高？",
    "袖带绑多紧合适？", "血压计清洗完能马上用吗？", "白大衣高血压严重吗？",
    "测血压前能喝咖啡吗？", "小孩血压怎么量？",
]


def measure(encode_fn, queries, warmup=10, iters=60):
    """预热后多次编码计时，返回 p50/p95/mean/min 延迟（毫秒）。"""
    for q in queries[:warmup]:                     # 预热阶段：触发懒加载/JIT/缓存，不计入统计
        encode_fn([q])                             # 逐条跑一次
    ts = []                                        # 采样时间列表
    for q in queries * max(1, iters // len(queries)):  # 查询池循环拼接，凑够约 iters 次测量
        t0 = time.perf_counter()                   # 计时开始
        encode_fn([q])                             # 单条编码（模拟线上单查请求）
        ts.append((time.perf_counter() - t0) * 1000)  # 记录毫秒耗时
    ts.sort()                                      # 排序后才能取分位数
    return {                                       # 返回四项延迟统计
        "mean_ms": round(statistics.mean(ts), 3),  # 平均值
        "p50_ms": round(statistics.median(ts), 3),  # 中位数（典型延迟）
        "p95_ms": round(ts[max(0, int(len(ts) * 0.95) - 1)], 3),  # P95（尾部延迟；-1 兼容越界）
        "min_ms": round(ts[0], 3),                 # 最小值（理论下限）
    }


def main():
    """基准主流程：分别测量 PyTorch-CPU、ONNX-INT8/FP32、PyTorch-GPU 三档延迟并生成报告。"""
    args = parse_args()                            # 解析参数
    ckpt = Path(args.ckpt)                         # 指定 checkpoint
    if not ckpt.exists():                          # 不存在则回退
        ckpt = ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt"  # 用 cosine 版兜底
    print(f"checkpoint: {ckpt.name} | 查询样本 {len(SAMPLE_QUERIES)} 条 | 每后端 {args.iters} 次")  # 打印基准配置

    # ── PyTorch CPU 基线 ───────────────────────────────────────────────────
    device_cpu = torch.device("cpu")               # CPU 设备
    torch_enc, meta = load_dense_encoder(ckpt, BERT_PATH, onnx_dir=None,
                                         device=device_cpu, prefer="torch")  # 强制 PyTorch 后端（onnx_dir=None 跳过 ONNX 档）
    res_torch_cpu = measure(torch_enc.encode, SAMPLE_QUERIES, iters=args.iters)  # 测 PyTorch 基线
    print(f"[PyTorch-CPU fp32 ] p50={res_torch_cpu['p50_ms']}ms "
          f"p95={res_torch_cpu['p95_ms']}ms mean={res_torch_cpu['mean_ms']}ms")  # 打印结果

    # ── ONNX INT8 / FP32 ───────────────────────────────────────────────────
    onnx_res = {}                                  # 两种 ONNX 档位的结果
    for kind in ("int8", "fp32"):                   # 从快到慢逐档测
        cand = sorted(ONNX_DIR.glob(f"*{ckpt.stem}*{kind}*.onnx")) if ONNX_DIR.exists() else []  # 先找带本 checkpoint 名的模型
        if not cand:                               # 没有专属命名的
            cand = sorted(ONNX_DIR.glob(f"*{kind}*.onnx")) if ONNX_DIR.exists() else []  # 退而求其次找任意该档模型
        if not cand:                               # 彻底没有
            print(f"[ONNX-{kind:>4}] 未找到模型，跳过")  # 提示跳过
            continue                              # 下一档
        from transformers import BertTokenizer     # 到用时分词器
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)  # 加载
        enc = OnnxEncoder(cand[0], tokenizer, max_length=meta["max_length"])  # ONNX 编码器（max_length 用 torch 后端的 meta 保持一致）
        r = measure(enc.encode, SAMPLE_QUERIES, iters=args.iters)  # 测该档延迟
        print(f"[ONNX-{kind:>4} CPU      ] p50={r['p50_ms']}ms p95={r['p95_ms']}ms "
              f"mean={r['mean_ms']}ms  模型={cand[0].name}")  # 打印（注明实际模型文件）
        onnx_res[kind] = r                         # 收集结果

    # ── GPU 参考 ───────────────────────────────────────────────────────────
    res_gpu = None                                 # GPU 结果（无则保持 None）
    if torch.cuda.is_available():                  # 有 GPU 才测
        device_gpu = torch.device("cuda")           # GPU 设备
        gpu_enc, _ = load_dense_encoder(ckpt, BERT_PATH, onnx_dir=None,
                                        device=device_gpu, prefer="torch")  # 强制 PyTorch GPU
        res_gpu = measure(gpu_enc.encode, SAMPLE_QUERIES, iters=args.iters)  # 测 GPU 延迟
        print(f"[PyTorch-GPU fp32 ] p50={res_gpu['p50_ms']}ms p95={res_gpu['p95_ms']}ms "
              f"mean={res_gpu['mean_ms']}ms（参考）")  # 打印（仅供参考）
    else:
        print("[PyTorch-GPU fp32 ] 本机无可用 GPU，跳过参考项")  # 无 GPU 提示

    # ── 结论 ───────────────────────────────────────────────────────────────
    report = {"checkpoint": ckpt.name, "torch_cpu_fp32": res_torch_cpu,
              "onnx": onnx_res, "torch_gpu_fp32": res_gpu}  # 汇总所有测量结果
    best_kind = min(onnx_res, key=lambda k: onnx_res[k]["p50_ms"]) if onnx_res else None  # 选 p50 最小的 ONNX 档
    if best_kind:                                  # 至少测到一个 ONNX 档
        base = res_torch_cpu["mean_ms"]             # 基线 = PyTorch CPU 平均延迟
        opt = onnx_res[best_kind]["mean_ms"]         # 最优 ONNX 平均延迟
        report["conclusion"] = {                    # 写结论块
            "best_backend": f"onnx-{best_kind}",    # 最快后端名
            "speedup_vs_torch_cpu": round(base / opt, 2),  # 加速倍数
            "latency_reduction_pct": round((1 - opt / base) * 100, 1),  # 延迟压降百分比
        }
        c = report["conclusion"]                    # 简化引用
        print(f"\n结论: ONNX-{best_kind} 相对 PyTorch-CPU 加速 {c['speedup_vs_torch_cpu']}x，"
              f"单次编码延迟压降 {c['latency_reduction_pct']}%")  # 打印结论
        if c["latency_reduction_pct"] < 60:         # 压降未达 60% 目标
            print("（未达 60% 目标：当前为 CPU 量化路径；GPU 环境建议 FP16/TensorRT，"
                  "见优化建议清单）")                 # 给出后续优化方向
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
    LOG_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")           # 报告落盘 JSON
    print(f"报告 → {LOG_PATH}")                     # 提示产物位置


def parse_args():
    """解析延迟基准命令行的各项参数。"""
    p = argparse.ArgumentParser()                   # 解析器（无 description，极简）
    p.add_argument("--ckpt", default=str(ROOT / "outputs" / "checkpoints" /
                                        "biencoder_hardneg_best.pt"))  # 默认难负例权重
    p.add_argument("--iters", default=60, type=int)  # 每后端测量次数
    return p.parse_args()                          # 返回解析结果


if __name__ == "__main__":
    main()                                         # 直跑才执行基准测试
