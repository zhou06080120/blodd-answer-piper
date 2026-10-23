"""
ONNX 运行时：模型导出（FP32/INT8 动态量化）+ 双后端编码器（需求 3 落地组件）

设计：
  1. export_biencoder_onnx —— 把 BiEncoder 的 encode 路径（BERT→池化→Dropout→L2 归一化）
     导出为静态计算图 ONNX（固定 seq=64，仅 batch 动态，利于 ORT 图优化）
  2. quantize_dynamic_int8 —— INT8 动态量化（CPU 友好，无需标定集，权重+激活动态量化）
  3. OnnxEncoder / TorchEncoder —— 统一 encode(texts)->[N,H] 接口；
     load_dense_encoder 按 "onnx-int8 → onnx-fp32 → pytorch" 顺序自动回退
     （对齐历史经验：任何一层加载失败都不能让服务启动崩溃）

说明：FP16/TensorRT 需要 GPU 加速包（onnxruntime-gpu / TensorRT），本机 ORT 仅 CPU
provider，故线上默认走 INT8 CPU 路径；GPU 服务器可在部署时换装 onnxruntime-gpu 并
重导出 FP16 模型（见 export_onnx.py --precision fp16）。
"""
# ↑ 模块级 docstring：本文件负责训练后模型的"部署形态"——导出 ONNX、量化压缩、双后端推理

import os                                       # 读取环境变量 BP_BERT_PATH（容器部署时覆盖 checkpoint 内记录的本机训练路径）
import numpy as np                              # 数值计算库：所有后端的向量张量都以 numpy 数组流转

try:                                            # 尝试导入 onnxruntime（部署机可能没装）
    import onnxruntime as ort                  # ONNX 推理引擎（比 PyTorch CPU 推理快得多）
    _HAS_ORT = True                             # 标记：ORT 可用
except Exception:  # onnxruntime 缺失时整体降级 pytorch 后端
    _HAS_ORT = False                            # 标记：ORT 不可用 → load_dense_encoder 自动回退 PyTorch


# ── 导出 ────────────────────────────────────────────────────────────────────

def export_biencoder_onnx(model, tokenizer, path, max_length=64, opset=17):
    """导出 BiEncoder 单路 encode 为 ONNX（输入 [B, max_length]，返回 L2 归一化向量）"""
    import torch                                # 函数内导入 torch，避免模块级依赖（纯 ORT 环境也可 import 本模块）
    from torch import nn                        # 用 nn.Module 写包装器

    if not str(path).endswith(".onnx"):          # 规范输出文件名
        path = str(path) + ".onnx"               # 缺后缀则自动补 .onnx

    class _Encoder(nn.Module):                  # 装饰器模式：把 encode() 三参接口包装成 forward
        def __init__(self, m):                  # m = 待导出的 BiEncoder
            super().__init__()                  # 父类初始化
            self.m = m                          # 持有被包装模型（nn.Module 嵌套注册）

        def forward(self, input_ids, attention_mask, token_type_ids):  # torch.onnx.export 只认 forward 签名
            return self.m.encode(input_ids, attention_mask, token_type_ids)  # 直接委托给 BiEncoder.encode（池化+归一化全在图内）

    wrapper = _Encoder(model).eval()            # 包装器切 eval 模式（关 Dropout——推理图里不能有随机性）
    enc = tokenizer("血压多少算正常？", max_length=max_length, truncation=True,  # 造一个真实业务例句作为 dummy 输入
                    padding="max_length", return_tensors="pt")  # 定长填充到 max_length → 导出静态 shape 图
    dummy = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])  # 转成元组，export 按此追踪计算图
    model.eval()                                # 原模型也切 eval，双保险
    torch.onnx.export(                         # 执行导出：追踪 forward 的算子流写成 ONNX 计算图
        wrapper, dummy, path,                   # 被导出模块、示例输入、输出文件
        input_names=["input_ids", "attention_mask", "token_type_ids"],  # 给输入张量命名，ORT 推理时按名喂参
        output_names=["embedding"],              # 输出命名为 embedding
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},  # 只放开第 0 维（batch）为动态，序列长固定 64 利于优化
                      "token_type_ids": {0: "batch"}},  # 三个输入的 batch 维共享同一符号名 → ORT 校验一致性
        opset_version=opset,                    # 指定 ONNX 算子集版本（17 支持较新的算子）
    )
    return path                                 # 返回最终落盘的 ONNX 文件路径


def quantize_dynamic_int8(src, dst):
    """INT8 动态量化（无需标定集）"""
    from onnxruntime.quantization import QuantType, quantize_dynamic  # ORT 官方量化工具
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,  # 权重量化成有符号 int8（存储减为约 1/4）
                     extra_options={"ActivationSymmetric": True})     # 激活对称量化 → 无需标定集，直接转换
    return dst                                  # 返回量化产物路径（线上 CPU 推理的默认形态）


# ── 编码器实现 ──────────────────────────────────────────────────────────────

def _tokenize_batch(tokenizer, texts, max_length):
    """逐条 tokenize 并堆叠为 numpy 数组，返回 (input_ids, attention_mask, token_type_ids)。"""
    ids, masks, ttids = [], [], []              # 分别收集三路输入
    for t in texts:                             # 逐句编码（tokenizer 支持列表输入，但逐条便于控制 dtype/维度）
        enc = tokenizer(t, max_length=max_length, truncation=True,  # 截断超长文本
                        padding="max_length", return_tensors="np")  # 直接产出 numpy 数组（ORT 原生吃 numpy）
        ids.append(enc["input_ids"][0])         # [0] 去掉批量维（tokenizer 会多套一层）
        masks.append(enc["attention_mask"][0])   # 收集掩码
        ttids.append(enc["token_type_ids"][0])   # 收集类型标记
    return (np.stack(ids).astype(np.int64), np.stack(masks).astype(np.int64),  # 堆叠成 [N, L] 并转 int64（ORT 要求）
            np.stack(ttids).astype(np.int64))   # 三路输入齐返回


def _l2_norm(x):
    denom = np.linalg.norm(x, axis=-1, keepdims=True)  # 沿向量维度求 L2 范数，keepdims 保留维度便于广播相除
    return x / np.maximum(denom, 1e-12)         # 除以范数归一化；maximum 防零向量除零（数值安全）


class OnnxEncoder:
    """ORT 推理编码器（INT8/FP32 ONNX），CPU 线程池可配"""

    backend = "onnx"                            # 类属性：标记后端类型，load_dense_encoder 写进 meta 供服务报告

    def __init__(self, model_path, tokenizer, max_length=64, batch_size=128,
                 intra_threads=0):
        if not _HAS_ORT:                        # 环境里没有 onnxruntime 就不该走到这（防御）
            raise RuntimeError("onnxruntime 不可用")  # 显式报错 → 工厂函数捕获后降级下一档
        opts = ort.SessionOptions()             # ORT 会话配置对象
        opts.intra_op_num_threads = intra_threads  # 0 = ORT 自动选择  # 算子内并行线程数；0 表示让 ORT 自行决定
        opts.inter_op_num_threads = 1           # 算子间并行度设 1（避免线程争抢，推理服务通常这样配）
        self.sess = ort.InferenceSession(str(model_path), sess_options=opts,  # 加载 ONNX 计算图建立推理会话（一次加载，多次复用）
                                         providers=["CPUExecutionProvider"])  # 显式指定 CPU 执行后端（本机无 GPU provider）
        self.tokenizer = tokenizer              # 保存 tokenizer 供 encode 用
        self.max_length = max_length            # 与导出时一致的定长
        self.batch_size = batch_size            # 默认批量切片大小
        self.kind = "int8" if "int8" in str(model_path).lower() else "fp32"  # 从文件名推断精度档位，写进 meta 展示

    def encode(self, texts, batch_size=None):
        bs = batch_size or self.batch_size      # 允许调用方临时覆盖批量（or 短路：None 时取默认值）
        out = []                                # 收集各批次的向量结果
        for i in range(0, len(texts), bs):      # 按步长 bs 切片迭代，i 为每个批次的起点
            ids, masks, ttids = _tokenize_batch(self.tokenizer, texts[i:i + bs],  # 本批文本转 numpy 三元组
                                                self.max_length)
            vec = self.sess.run(None, {"input_ids": ids, "attention_mask": masks,  # 喂给 ONNX 会话推理；None=取全部输出
                                       "token_type_ids": ttids})[0]  # [0] 取第一个输出张量（即 embedding）
            out.append(vec.astype(np.float32))   # 统一转 float32（int8 模型输出可能带量化后的类型）
        return _l2_norm(np.concatenate(out, axis=0))  # 沿 batch 维拼回 [N, H] 再做 L2 归一化（与训练口径一致）


class TorchEncoder:
    """PyTorch 编码器（GPU 可用时走 GPU；作为 ONNX 缺失/失败时的回退后端）"""

    backend = "pytorch"                         # 类属性：后端标记为 pytorch

    def __init__(self, model, tokenizer, device, max_length=64, batch_size=128):
        self.model = model                      # 已加载权重、eval 状态的 BiEncoder
        self.tokenizer = tokenizer              # 保存 tokenizer
        self.device = device                    # 计算设备（cuda / cpu）
        self.max_length = max_length            # 定长编码
        self.batch_size = batch_size            # 默认批量

    def encode(self, texts, batch_size=None):
        import torch                            # 函数内导入，保持与 OnnxEncoder 对称的轻量 import
        bs = batch_size or self.batch_size      # 批量取默认或调用方覆盖值
        out = []                                # 收集各批次向量
        with torch.no_grad():                   # 推理上下文：不建计算图、省显存并加速
            for i in range(0, len(texts), bs):  # 分批迭代
                ids, masks, ttids = _tokenize_batch(self.tokenizer, texts[i:i + bs],  # 先转 numpy（复用同一工具函数）
                                                    self.max_length)
                vec = self.model.encode(        # 调 PyTorch 模型的 encode（池化+归一化在模型内完成）
                    torch.from_numpy(ids).to(self.device),    # numpy → tensor → 搬到目标设备
                    torch.from_numpy(masks).to(self.device),  # 掩码同样搬运
                    torch.from_numpy(ttids).to(self.device),  # 类型标记搬运
                )
                out.append(vec.detach().cpu().numpy().astype(np.float32))  # 脱离计算图 → 搬回 CPU → 转 numpy
        return _l2_norm(np.concatenate(out, axis=0))  # 拼回 [N, H] 并归一化（与 OnnxEncoder 输出口径完全一致）


# ── 工厂：按优先级自动选择后端 ───────────────────────────────────────────────

def load_dense_encoder(biencoder_ckpt, bert_path, onnx_dir=None, device=None,
                       max_length=64, prefer="onnx"):
    """
    构建稠密编码器。顺序：
      prefer=="onnx": onnx-int8 → onnx-fp32 → pytorch（任一层失败自动降级，不抛异常）
      prefer=="torch": pytorch（GPU 优先）
    返回 (encoder, meta)，meta 含 backend/kind 等信息。
    """
    import torch                                # 加载 checkpoint 用
    from transformers import BertTokenizer      # 加载分词器用

    from model import build_biencoder           # 回退后端需要重建 PyTorch 模型
    from pathlib import Path                    # 检查 ONNX 目录用

    ckpt = torch.load(str(biencoder_ckpt), map_location="cpu", weights_only=False)  # 加载训练脚本保存的 checkpoint（含 state_dict + args），先映射到 CPU 避免设备冲突
    saved = ckpt.get("args", {})                 # 取出训练时的超参（bert_path/pool/层数等），保证部署与训练一致
    bert_path = os.getenv("BP_BERT_PATH") or saved.get("bert_path") or bert_path  # 优先级：环境变量（容器挂载）> checkpoint 记录 > 传参
    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 加载分词器（所有后端共用同一份）
    max_length = saved.get("max_length", max_length) or max_length  # max_length 同样以 checkpoint 记录优先

    meta = {"bert_path": bert_path, "max_length": max_length, "backend": None}  # meta 先建骨架，选定后端后补充字段

    if prefer == "onnx" and onnx_dir and Path(onnx_dir).exists():  # 只有明确要 ONNX 且目录存在才尝试
        for name in ("int8", "fp32"):           # 按优先级从快到慢逐档尝试（int8 最快，失败再试 fp32）
            for f in Path(onnx_dir).glob(f"*{name}*.onnx"):  # 按档位名模糊匹配目录下的 onnx 文件
                try:                             # 单文件加载失败不能拖垮整个服务 → 逐个 try
                    enc = OnnxEncoder(f, tokenizer, max_length=max_length)  # 尝试建 ORT 编码器
                    meta.update(backend=enc.backend, kind=enc.kind, onnx=str(f),  # 成功 → meta 记录后端/精度/文件
                                device="cpu")   # ONNX 路径固定 CPU
                    return enc, meta            # 命中即返回（最优先档位生效）
                except Exception as e:           # 本文件加载失败（损坏/缺算子等）
                    print(f"[onnx_runtime] {f.name} 加载失败（{e}），尝试下一档。")  # 打印原因并继续降级

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 回退 PyTorch：自动选择 GPU/CPU
    model = build_biencoder(bert_path=bert_path, pool=saved.get("pool", "mean"),  # 按 checkpoint 超参重建模型结构
                            num_hidden_layers=saved.get("num_hidden_layers"))
    model.load_state_dict(ckpt["state_dict"])   # 把训练好的权重载入模型
    model.to(device).eval()                     # 搬到设备并切 eval（关 Dropout）
    enc = TorchEncoder(model, tokenizer, device, max_length=max_length)  # 包成统一接口的 Torch 编码器
    meta.update(backend=enc.backend, device=str(device))  # meta 记录实际后端与设备
    return enc, meta                            # 返回编码器与元信息（服务层会打印 meta 告知线上走哪条路径）
