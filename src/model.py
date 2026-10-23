"""
文本匹配模型定义

教学重点：
  1. BiEncoder（表示型）— 共享 BERT 骨干，对两句分别编码，计算余弦相似度
     对应 Sentence-BERT 论文中的 Siamese 架构
  2. CrossEncoder（交互型）— 两句拼接后整体送入 BERT，直接输出匹配概率
  3. L2 归一化 — encode() 输出归一化向量后，余弦相似度等价于点积（更高效）
  4. num_hidden_layers — 限制 BERT 层数加速训练（4 层约为全量的 1/3 时间）
     原理：从完整 12 层权重中只加载前 N 层，其余丢弃

使用方式：
  from model import BiEncoder, CrossEncoder, build_biencoder, build_crossencoder

依赖：
  pip install torch transformers
"""
# ↑ 模块级 docstring：说明本文件定义了 RAG 检索阶段用的两种匹配模型

import torch.nn as nn                           # 神经网络模块（层、损失、容器）
import torch.nn.functional as F                 # 函数式接口（归一化、激活等）
import transformers                             # HuggingFace 库总入口，用于控制日志级别
from transformers import BertConfig, BertModel  # BERT 配置类与预训练模型类


# ── BiEncoder ─────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：BiEncoder 双塔（Siamese）匹配模型
# 一句话概括：两个句子各走一遍共享权重的 BERT，得到两个句向量后用余弦相似度衡量匹配度。
# 在流水线中的位置：训练后为 FAQ 库预计算句向量，线上把用户问题编码成向量做最近邻召回。
# ═════════════════════════════════════════════════════════════════════════
class BiEncoder(nn.Module):
    """
    表示型文本匹配：Siamese Bi-Encoder

    结构：
      shared BertModel → 池化 → Dropout → L2 归一化 → 句向量

    匹配方式：
      sim = cosine_similarity(encode(s1), encode(s2))
      sim ∈ [-1, 1]，越接近 1 越相似

    支持两种 Loss：
      CosineEmbeddingLoss — 直接用相似度与标签计算损失
      TripletLoss         — 拉近 (anchor, positive)，推远 (anchor, negative)

    参数：
      bert_path         : 预训练权重路径（本地目录或 HuggingFace 模型名）
      pool              : 向量提取策略，'cls' / 'mean' / 'max'
                          mean 在句子相似度任务上通常优于 cls（Sentence-BERT 结论）
      dropout           : Dropout 比例
      num_hidden_layers : BERT Transformer 层数；None = 全量 12 层，
                          建议课堂快速验证用 4 层，留 12 层给学生自行实验
    """

    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
        super().__init__()                       # 调用父类初始化，注册子模块/参数所必需
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"  # 防御式校验池化策略合法性，防止拼写错误静默失效

        config = BertConfig.from_pretrained(bert_path)  # 读取预训练模型的配置 JSON（层数、隐层维度等）
        if num_hidden_layers is not None:              # 若调用方指定了裁剪层数
            config.num_hidden_layers = num_hidden_layers  # 覆盖配置中的层数 → 加载时只取前 N 层权重，训练更快

        _prev = transformers.logging.get_verbosity()   # 记录当前日志级别，便于加载完毕后恢复
        transformers.logging.set_verbosity_error()     # 临时静默加载时的 warning（如"部分权重未使用"），保持控制台干净
        self.bert = BertModel.from_pretrained(bert_path, config=config)  # 按修改后的配置加载预训练 BERT 骨干（Siamese 结构的关键：a/b 两塔共享同一实例）
        transformers.logging.set_verbosity(_prev)       # 恢复原日志级别

        self.pool    = pool                        # 保存池化策略，_pool() 方法中会按它分支
        self.dropout = nn.Dropout(dropout)          # 训练时随机失活防过拟合；eval 模式下自动关闭

    def encode(self, input_ids, attention_mask, token_type_ids):
        """
        单句编码，返回 L2 归一化后的句向量 [B, H]

        L2 归一化后：cosine_sim(u, v) == dot(u, v)
        可用矩阵乘法批量计算所有两两相似度，适合向量检索场景（如 RAG）
        """
        out = self.bert(                          # 前向传播：把 token 序列交给 BERT
            input_ids=input_ids,                  # token id 序列 [B, L]
            attention_mask=attention_mask,        # 1=真实 token，0=padding（BERT 据此忽略填充位）
            token_type_ids=token_type_ids,        # 句子归属标记（单句全 0；双句前句 0 后句 1）
            return_dict=True,                     # 返回 ModelOutput 字典而非 tuple，取字段更直观
        )
        vec = self._pool(out.last_hidden_state, attention_mask)  # [B, H]  # 从 [B,L,H] 的 token 向量序列压成 [B,H] 句向量
        vec = self.dropout(vec)                   # 池化结果过 Dropout（仅训练时生效）
        return F.normalize(vec, p=2, dim=-1)      # L2 归一化 → 向量落单位球面，之后点积即余弦相似度

    def forward(self, batch_a, batch_b):
        """返回 (emb_a, emb_b)，各形状 [B, H]，可直接计算余弦相似度"""
        emb_a = self.encode(**batch_a)            # 塔一：编码句子 a（** 把字典解包成 encode 的三个参数）
        emb_b = self.encode(**batch_b)             # 塔二：编码句子 b（与 a 共享同一 BERT 权重 = Siamese）
        return emb_a, emb_b                       # 训练脚本拿两个向量算余弦损失；检索场景只调 encode

    def _pool(self, last_hidden, attention_mask):
        # 池化策略分发：决定如何把 token 级向量序列 [B,L,H] 压缩成句向量 [B,H]
        if self.pool == "cls":                    # 策略一：直接取 [CLS] 位置向量（BERT 的"句摘要"位）
            return last_hidden[:, 0, :]           # 切片取第 0 个 token → [B, H]

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]  # 升维成 [B,L,1] 便于与 [B,L,H] 逐元素相乘

        if self.pool == "mean":                   # 策略二：真实 token 向量取平均（忽略 padding）
            sum_h = (last_hidden * mask).sum(dim=1)    # padding 位乘 0 被屏蔽后，沿序列维求和 → [B,H]
            count = mask.sum(dim=1).clamp(min=1e-9)    # 每句真实 token 数，clamp 防止除零
            return sum_h / count                      # 求平均 → [B, H]

        if self.pool == "max":                    # 策略三：每个维度沿序列取最大值
            masked = last_hidden + (1 - mask) * (-1e9)  # padding 位加上极大负数，max 时必不可能被选中
            return masked.max(dim=1).values       # 沿序列维取 max → [B, H]


# ── CrossEncoder ──────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：CrossEncoder 交互式匹配模型
# 一句话概括：把"句对"拼接成一条序列整体送入 BERT，让两句话在每一层充分交互，输出 2 分类 logits。
# 在流水线中的位置：作为 Reranker，对 BiEncoder 召回的 Top-K 候选精排（精度高但慢，不能全库扫）。
# ═════════════════════════════════════════════════════════════════════════
class CrossEncoder(nn.Module):
    """
    交互型文本匹配：Cross-Encoder

    结构：
      BertModel([CLS] s1 [SEP] s2 [SEP]) → CLS 向量 → Dropout → Linear(H, 2) → logits

    对比 BiEncoder：
      优点：两句在每一层都交互，表达能力更强，精度更高
      缺点：无法预计算向量，每对句子都要完整过 BERT，不适合大规模检索
      典型用途：Reranker（对召回的 Top-K 候选精排），即 rag_annual_report 中的做法

    参数：
      bert_path         : 预训练权重路径
      dropout           : 分类头 Dropout 比例
      num_hidden_layers : 同 BiEncoder，限层数加速
    """

    def __init__(self, bert_path, dropout=0.1, num_hidden_layers=None):
        super().__init__()                       # 父类初始化，注册模块

        config = BertConfig.from_pretrained(bert_path)  # 读取预训练配置
        if num_hidden_layers is not None:              # 若指定层数
            config.num_hidden_layers = num_hidden_layers  # 覆盖层数以裁剪模型、加速训练

        _prev = transformers.logging.get_verbosity()   # 保存当前日志级别
        transformers.logging.set_verbosity_error()      # 临时静默加载告警
        self.bert = BertModel.from_pretrained(bert_path, config=config)  # 加载 BERT 骨干（输入是"句对拼接"的单序列）
        transformers.logging.set_verbosity(_prev)       # 恢复日志级别

        hidden_size  = self.bert.config.hidden_size    # 从配置读隐层维度 H（如 768），分类头输入维度由它决定
        self.dropout = nn.Dropout(dropout)             # 分类头前的 Dropout
        self.classifier = nn.Linear(hidden_size, 2)    # 2 分类头：输出 [不匹配, 匹配] 的原始分数 logits

    def forward(self, input_ids, attention_mask, token_type_ids):
        """返回 logits [B, 2]，未经 softmax（CrossEntropyLoss 内部处理）"""
        out = self.bert(                          # 句对整体过 BERT，两句在自注意力中逐层交互
            input_ids=input_ids,                  # 拼接序列 [B, L]：[CLS] s1 [SEP] s2 [SEP]
            attention_mask=attention_mask,        # padding 掩码
            token_type_ids=token_type_ids,        # 前句标记 0、后句标记 1，让 BERT 知道句边界
            return_dict=True,                     # 字典式输出
        )
        cls_vec = out.last_hidden_state[:, 0, :]  # [B, H]  # 取 [CLS] 位向量作为整对句子的语义摘要
        cls_vec = self.dropout(cls_vec)           # 过 Dropout 防过拟合
        return self.classifier(cls_vec)            # [B, 2]  # 映射到 2 维 logits；训练配 CrossEntropyLoss，推理配 softmax 得概率


# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    """构建 BiEncoder 并打印参数量。"""
    model = BiEncoder(bert_path, pool=pool, dropout=dropout,  # 按给定超参实例化双塔模型
                      num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")  # 打印参数量报告，直观感受裁剪层数的效果
    return model                                # 返回待训练/待加载权重的模型实例


def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    """构建 CrossEncoder 并打印参数量。"""
    model = CrossEncoder(bert_path, dropout=dropout,  # 实例化交叉编码器
                         num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")  # 打印参数量报告
    return model                                # 返回模型实例


def _print_param_info(model, name):
    """打印模型总参数量与 BERT 骨干参数量（单位：百万 M）。"""
    total = sum(p.numel() for p in model.parameters()) / 1e6    # 遍历所有参数张量求元素总数，除以 1e6 换算成"百万"
    bert  = sum(p.numel() for p in model.bert.parameters()) / 1e6  # 只统计 BERT 骨干参数量，用于对比分类头占比
    print(f"模型: {name}")                       # 打印模型名称与关键超参
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")  # 打印总参数量与骨干参数量（保留 1 位小数）
