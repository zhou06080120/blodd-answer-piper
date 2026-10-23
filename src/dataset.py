"""
文本匹配数据集类

教学重点：
  1. PairDataset — 句对数据集，用于 CosineEmbeddingLoss 训练和评估
  2. TripletDataset — 三元组数据集，用于 TripletLoss 训练
     离线构建方式：从正样本对出发，为每个 anchor 查找负样本
  3. CrossEncoderDataset — 交互型数据集，句对拼接为单序列送入 BERT

使用方式：
  from dataset import PairDataset, TripletDataset, CrossEncoderDataset, build_pair_loaders

依赖：
  pip install torch transformers
"""
# ↑ 模块级 docstring：本文件负责把 JSONL 原始数据包装成 PyTorch 可迭代的数据集/DataLoader

import json                                     # 解析 JSONL 文件（每行一个 JSON 对象）
import random                                   # 负采样时的随机选择（已固定种子保证可复现）
from collections import defaultdict            # 缺失键自动创建默认值的字典，用于建倒排索引
from pathlib import Path                       # 面向对象路径操作，拼接数据文件路径

import torch                                    # 构造标签张量
from torch.utils.data import DataLoader, Dataset  # PyTorch 数据集基类与批量加载器

random.seed(42)                                 # 固定全局随机种子 → 每次运行三元组负采样结果一致（可复现性）


# ── 工具函数 ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []                                   # 收集解析后的所有样本行
    with open(path, encoding="utf-8") as f:     # 以 UTF-8 打开 JSONL 文件（中文数据必须显式指定编码）
        for line in f:                          # 逐行读取（JSONL = 每行一个独立 JSON）
            line = line.strip()                 # 去掉行首尾空白（含换行符）
            if line:                            # 跳过空行，避免解析报错
                rows.append(json.loads(line))  # 把该行 JSON 文本解析成 dict 并收集
    return rows                                 # 返回样本字典列表，如 [{"sentence1":..., "sentence2":..., "label":...}, ...]


def encode_single(tokenizer, text, max_length):
    enc = tokenizer(                            # 调用 HuggingFace tokenizer 把一句话转成模型输入张量
        text,                                    # 待编码的原始文本
        max_length=max_length,                  # 序列最大长度上限
        truncation=True,                        # 超长文本从尾部截断
        padding="max_length",                   # 不足 max_length 的补 [PAD] 到等长（便于批量堆叠）
        return_tensors="pt",                    # 返回 PyTorch 张量格式
    )
    return {                                    # 统一封装成 dict，squeeze(0) 去掉多余的第 0 维（单句无 batch 维）
        "input_ids":      enc["input_ids"].squeeze(0),        # token id 序列 [L]
        "attention_mask": enc["attention_mask"].squeeze(0),   # 真实位=1 / padding 位=0 [L]
        "token_type_ids": enc["token_type_ids"].squeeze(0),   # 句子归属标记（单句全 0）[L]
    }


# ── PairDataset ───────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：PairDataset 句对数据集
# 一句话概括：每条样本 = (句1, 句2, 是否匹配标签)，按行取出时现场 tokenize 成 BiEncoder 需要的双塔输入。
# 在流水线中的位置：BiEncoder 的 CosineEmbeddingLoss 训练与检索评估都吃这个数据集。
# ═════════════════════════════════════════════════════════════════════════
class PairDataset(Dataset):
    """
    句对数据集：每个样本是 (sentence1, sentence2, label)

    用途：
      - CosineEmbeddingLoss 训练（label 0/1 在训练脚本中转为 -1/+1）
      - 评估时计算余弦相似度分布

    参数：
      data_path  : JSONL 文件路径，字段 sentence1 / sentence2 / label
      tokenizer  : HuggingFace tokenizer
      max_length : 单句最大 token 数
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        self.rows      = load_jsonl(data_path)          # 把 JSONL 全量加载进内存（数据量小，直接 list 访问最简单）
        self.tokenizer = tokenizer                       # 保存 tokenizer，__getitem__ 里现场做编码
        self.max_length = max_length                     # 单句最大 token 数

    def __len__(self):
        return len(self.rows)                            # DataLoader 靠它知道数据集总条数来划分 batch

    def __getitem__(self, idx):
        r = self.rows[idx]                               # 取第 idx 条样本 dict
        enc_a = encode_single(self.tokenizer, r["sentence1"], self.max_length)  # 塔一输入：编码句子 1
        enc_b = encode_single(self.tokenizer, r["sentence2"], self.max_length)  # 塔二输入：编码句子 2
        return {                                         # 返回扁平化 dict；DataLoader 会自动按 key 堆叠成 batch 张量
            "input_ids_a":      enc_a["input_ids"],       # 句 1 的 token ids（_a 后缀对应塔 a）
            "attention_mask_a": enc_a["attention_mask"], # 句 1 的掩码
            "token_type_ids_a": enc_a["token_type_ids"], # 句 1 的类型标记
            "input_ids_b":      enc_b["input_ids"],      # 句 2 的 token ids（_b 后缀对应塔 b）
            "attention_mask_b": enc_b["attention_mask"], # 句 2 的掩码
            "token_type_ids_b": enc_b["token_type_ids"], # 句 2 的类型标记
            "label": torch.tensor(r["label"], dtype=torch.long),  # 匹配标签 0/1，long 类型配合交叉熵等损失
        }


# ── TripletDataset ────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：TripletDataset 三元组数据集
# 一句话概括：在 __init__ 阶段把句对数据离线重组为 (anchor, positive, negative) 三元组，供 TripletLoss 训练。
# 在流水线中的位置：BiEncoder 的 TripletLoss 训练分支（拉正推负，学出更有区分度的向量空间）。
# ═════════════════════════════════════════════════════════════════════════
class TripletDataset(Dataset):
    """
    三元组数据集：每个样本是 (anchor, positive, negative)

    构建逻辑（离线静态构建）：
      1. 扫描所有正样本对 label=1，得到 (anchor=s1, positive=s2)
      2. 为每个 anchor 收集其配对的负样本（同一 s1 出现在 label=0 对中的 s2）
      3. 若 anchor 无自身负样本，从全局负样本池随机选取
      → 最终 triplet 数量 ≈ 正样本对数量（AFQMC 约 10K 条）

    教学说明：
      这是"离线随机负采样"策略，简单但效果基础。
      进阶做法是"在线难负样本挖掘"（Online Hard Negative Mining）：
      同一 batch 内，选余弦相似度最高的非正例对作为负样本。

    参数：
      data_path  : JSONL 文件路径
      tokenizer  : HuggingFace tokenizer
      max_length : 单句最大 token 数
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        self.tokenizer  = tokenizer                     # 保存 tokenizer 供 __getitem__ 编码
        self.max_length = max_length                     # 单句最大 token 数
        self.triplets   = self._build_triplets(load_jsonl(data_path))  # 加载 JSONL 后立即离线构建三元组（一次性完成，训练期不再变）

    def _build_triplets(self, rows):
        # 为每个句子建立"其负样本列表"索引
        neg_by_sent = defaultdict(list)                  # 倒排索引：句子 → 与它标注为不匹配的句子列表
        all_sents   = set()                              # 全局句子去重集合（作为兜底负样本池）
        for r in rows:                                   # 第一遍扫描：建索引
            all_sents.add(r["sentence1"])                # 收集句子 1
            all_sents.add(r["sentence2"])                # 收集句子 2
            if r["label"] == 0:                          # 只有不匹配对才有资格进负样本索引
                neg_by_sent[r["sentence1"]].append(r["sentence2"])  # s1 的负样本 += s2
                neg_by_sent[r["sentence2"]].append(r["sentence1"])  # 匹配是对称关系，反向也登记

        global_pool = list(all_sents)                    # 全局句子池 → 列表形式供 random.choice 采样

        triplets = []                                    # 存放最终 (anchor, positive, negative) 三元组
        for r in rows:                                   # 第二遍扫描：从正样本对出发构造三元组
            if r["label"] != 1:                          # 跳过负样本对——三元组的骨架必须来自真实正例
                continue
            anchor   = r["sentence1"]                    # 锚点句：检索场景中的"用户问题"角色
            positive = r["sentence2"]                    # 正例句：与锚点语义等价的标准问法

            negs = neg_by_sent.get(anchor, [])           # 查倒排索引，看该锚点有没有现成的真实负样本
            if negs:                                     # 有 → 优先用数据集中真实标注的负例（比随机负例更有信息量）
                negative = random.choice(negs)           # 从候选负样本中随机挑一个（种子固定 → 可复现）
            else:
                # anchor 没有自身的负样本 → 从全局随机选
                negative = anchor                        # 先随便占位一个不可能的值
                while negative in (anchor, positive):    # 反复抽签直到负样本既不等于锚点也不等于正例
                    negative = random.choice(global_pool)  # 从全局句子池随机抽一句

            triplets.append((anchor, positive, negative))  # 三元组入列，供 __getitem__ 逐条取出

        print(f"  TripletDataset: 构建 {len(triplets):,} 个三元组")  # 打印构建结果规模，确认数据量符合预期
        return triplets                                   # 返回三元组列表

    def __len__(self):
        return len(self.triplets)                        # 数据集规模 = 三元组条数

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]   # 取出一条三元组的三句话
        enc_a = encode_single(self.tokenizer, anchor,   self.max_length)  # 编码锚点句（塔 a）
        enc_p = encode_single(self.tokenizer, positive, self.max_length)  # 编码正例句（塔 p）
        enc_n = encode_single(self.tokenizer, negative, self.max_length)  # 编码负例句（塔 n）
        return {                                          # 三路输入分别加 _a/_p/_n 后缀；DataLoader 自动按 key 拼 batch
            "input_ids_a":      enc_a["input_ids"],      # 锚点 token ids
            "attention_mask_a": enc_a["attention_mask"], # 锚点掩码
            "token_type_ids_a": enc_a["token_type_ids"], # 锚点类型标记
            "input_ids_p":      enc_p["input_ids"],      # 正例 token ids
            "attention_mask_p": enc_p["attention_mask"], # 正例掩码
            "token_type_ids_p": enc_p["token_type_ids"], # 正例类型标记
            "input_ids_n":      enc_n["input_ids"],      # 负例 token ids
            "attention_mask_n": enc_n["attention_mask"], # 负例掩码
            "token_type_ids_n": enc_n["token_type_ids"], # 负例类型标记
        }


# ── CrossEncoderDataset ───────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：CrossEncoderDataset 交互型数据集
# 一句话概括：用 tokenizer 的双参调用把两句拼成 "[CLS] s1 [SEP] s2 [SEP]" 单序列，直接作为 BERT 输入。
# 在流水线中的位置：CrossEncoder（Reranker）训练阶段的数据供给。
# ═════════════════════════════════════════════════════════════════════════
class CrossEncoderDataset(Dataset):
    """
    交互型数据集：sentence1 与 sentence2 拼接为单序列

    BERT tokenizer 自动生成：
      [CLS] sentence1 [SEP] sentence2 [SEP]
      token_type_ids: 0000...0  1111...1

    教学对比：
      相比 BiEncoder（两路独立编码），CrossEncoder 让两句在每一层都交互，
      表达能力更强，但无法预计算句向量，推理时每对都要过一次 BERT。

    参数：
      data_path  : JSONL 文件路径
      tokenizer  : HuggingFace tokenizer
      max_length : 句对总最大 token 数（两句拼接后）
    """

    def __init__(self, data_path, tokenizer, max_length=128):
        self.rows       = load_jsonl(data_path)          # 全量加载句对样本进内存
        self.tokenizer  = tokenizer                       # 保存 tokenizer
        self.max_length = max_length                      # 拼接后的总长上限（双句共用，故比单句的 64 更大）

    def __len__(self):
        return len(self.rows)                            # 返回样本总数

    def __getitem__(self, idx):
        r = self.rows[idx]                               # 取第 idx 条样本
        enc = self.tokenizer(                            # 关键差异：传两个位置参数 → tokenizer 自动拼接并生成 token_type_ids
            r["sentence1"],                               # 前句：token_type_ids 标记为 0
            r["sentence2"],                               # 后句：token_type_ids 标记为 1
            max_length=self.max_length,                   # 拼接序列总长上限
            truncation=True,                              # 超长截断
            padding="max_length",                         # 补齐到等长
            return_tensors="pt",                          # 返回 PyTorch 张量
        )
        return {                                          # 单序列 + 标签；无 _a/_b 后缀，因为两句已在序列内交互
            "input_ids":      enc["input_ids"].squeeze(0),        # 拼接后的 token ids [L]
            "attention_mask": enc["attention_mask"].squeeze(0),   # 掩码 [L]
            "token_type_ids": enc["token_type_ids"].squeeze(0),   # 0/1 句界标记 [L]
            "label": torch.tensor(r["label"], dtype=torch.long),  # 匹配标签 0/1
        }


# ── DataLoader 工厂函数 ───────────────────────────────────────────────────

def build_pair_loaders(data_dir, tokenizer, max_length=64, batch_size=32,
                       train_file="train.jsonl"):
    """
    为 BiEncoder（CosineEmbeddingLoss / 评估）构建 train/val/test DataLoader。
    注意：AFQMC test 集无正样本标签，实际评估用 val。
    train_file 可指定增强训练集（如 bien_train_chitchat.jsonl）。
    """
    data_dir = Path(data_dir)                            # 包装成 Path 对象，可用 / 拼路径
    train_ds = PairDataset(data_dir / train_file, tokenizer, max_length)  # 训练集（可替换成增强过的数据文件）
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)  # 验证集：训练中监控泛化、选 checkpoint
    test_ds  = PairDataset(data_dir / "test.jsonl",       tokenizer, max_length)  # 测试集：最终评估（AFQMC 专属说明见 docstring）

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)   # 训练打乱顺序（打散分布），单进程加载
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)   # 验证不打乱（保证指标计算顺序稳定）
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)   # 测试不打乱

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")  # 打印训练集规模与 batch 数（右对齐排版）
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")       # 打印验证集规模
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch  (AFQMC test 无正样本，仅供参考)")  # 打印测试集规模并附注意事项
    return train_loader, val_loader, test_loader           # 三个 loader 一并返回，训练脚本解包使用


def build_triplet_loader(data_dir, tokenizer, max_length=64, batch_size=32):
    """为 TripletLoss 训练构建 DataLoader，val/test 仍用 PairDataset。"""
    data_dir = Path(data_dir)                            # 转成 Path 以便拼接
    train_ds = TripletDataset(data_dir / "train.jsonl", tokenizer, max_length)  # 训练用三元组数据集（内部做离线负采样）
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)  # 验证仍用句对形式（与最终评估口径一致）

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)   # 三元组训练 loader，打乱
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)   # 句对验证 loader，不打乱

    print(f"  triplet train : {len(train_ds):>7,} 三元组, {len(train_loader):>5} batch")  # 打印三元组训练集规模
    print(f"  val (pair)    : {len(val_ds):>7,} 对,     {len(val_loader):>5} batch")       # 打印验证集规模（注明是句对形式）
    return train_loader, val_loader                       # 只返回两个 loader（三元组训练一般不再单独跑 test）


def build_crossencoder_loaders(data_dir, tokenizer, max_length=128, batch_size=32,
                               train_file="train.jsonl"):
    """为 CrossEncoder 构建 train/val/test DataLoader。

    train_file 可指定增强训练集（如 ce_train.jsonl），val/test 仍用标准切分。
    """
    data_dir = Path(data_dir)                            # 转成 Path 以便拼接
    train_ds = CrossEncoderDataset(data_dir / train_file, tokenizer, max_length)  # 交互型训练集（max_length 默认 128 容纳双句拼接）
    val_ds   = CrossEncoderDataset(data_dir / "validation.jsonl", tokenizer, max_length)  # 交互型验证集
    test_ds  = CrossEncoderDataset(data_dir / "test.jsonl",       tokenizer, max_length)  # 交互型测试集

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)   # 训练打乱
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)   # 验证不打乱
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)   # 测试不打乱

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")  # 打印训练集规模
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")      # 打印验证集规模
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch")    # 打印测试集规模
    return train_loader, val_loader, test_loader           # 返回三个 loader
