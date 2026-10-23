"""
混合检索两阶段架构（需求 2 落地）：Dense(S-BERT) + Sparse(BM25) 多路召回 → RRF 融合 → CrossEncoder 精排

设计动机：
  - 医疗器械场景的专有名词（型号 HEM-7156、参数 140/90 mmHg、错误码 E1/E2、数字编号）
    在纯语义向量"平均池化"下极易被稀释，纯 Dense 检索首位命中率受损
  - BM25 稀疏检索按精确 token 召回，对型号/数字/错误码天然敏感，与 Dense 互补
  - RRF（Reciprocal Rank Fusion）无需调权重即可稳定合并双路排序
  - CrossEncoder（本项目 crossencoder_best.pt）对 Top-K 候选逐对精排，输出校准概率作为置信度

索引兼容灰度发布：KB 条目 status 为 active/gray 时参与检索（gray 权重打折），draft 不参与。

使用方式：
  from hybrid_retriever import HybridRetriever
  ret = HybridRetriever(faqs, dense_encode=encoder.encode, cross_score=ce_score)
  hits = ret.search("血压机咋用啊？", top_k=3, threshold=0.82)
"""
# ↑ 模块级 docstring：本文件是 RAG 检索的核心——双路召回 + 融合 + 精排的两阶段架构

import math                                     # BM25 的 IDF 公式用到 log
import re                                       # 分词正则
import time                                     # 各阶段延迟统计（perf_counter 高精度计时）

import numpy as np                              # 向量矩阵运算与 argsort 排名

from llm_synth import normalize                 # 复用 LLM 合成模块的文本归一化（去空白/统一全半角），用于 leave-one-out 泄漏屏蔽

_TOKEN_WORD = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")  # 匹配拉丁词/数字/带连字符或斜杠的型号（如 HEM-7156、140/90）
_TOKEN_CJK = re.compile(r"[\u4e00-\u9fff]")      # 匹配单个中文字符（CJK 统一表意文字区间）


def tokenize(text):
    """混合分词：保留型号/数字/错误码整词；中文 1-gram + 2-gram"""
    tokens = []                                  # 收集分词结果
    for m in _TOKEN_WORD.finditer(text):         # 逐个提取英文/数字整词（型号、错误码不拆散）
        tokens.append(m.group(0).lower())        # 统一转小写，做到大小写不敏感
    cjk = "".join(_TOKEN_CJK.findall(text))      # 抽出全部中文汉字拼成连续串（供下面切 gram）
    tokens.extend(list(cjk))                     # 中文 1-gram：每个字单独作为一个 token
    for i in range(len(cjk) - 1):                # 中文 2-gram：相邻两字组词（缓解单字歧义）
        tokens.append(cjk[i:i + 2])              # 切出第 i 个二元组
    return tokens                                # 返回 token 列表，供 BM25 建索引/查询共用


class BM25:
    """自研 BM25（兼容中文 char-gram，避免额外依赖）"""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b                  # BM25 两个超参：k1 控制 TF 饱和速度，b 控制文档长度归一化强度
        tokenized = [tokenize(t) for t in corpus]  # 预先把全部库文本分好词（建索引一次，查询多次复用）
        self.doc_len = np.array([len(t) or 1 for t in tokenized], dtype="float32")  # 每篇文档 token 数（or 1 防零长度除零）
        self.avgdl = float(self.doc_len.mean())   # 语料平均文档长度，BM25 长度归一化要用
        df = {}                                   # 文档频率：term → 含该 term 的文档篇数
        for toks in tokenized:                    # 统计 DF
            for term in set(toks):                # 每篇文档内 term 去重，一篇只计一次
                df[term] = df.get(term, 0) + 1     # 计数累加
        n = len(corpus) or 1                      # 语料总篇数（or 1 防空语料除零）
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}  # 平滑 IDF 公式：常见词权重低、稀有词权重高，恒为正
        self.tf_map = {t: {} for t in df}         # 倒排索引骨架：term → {文档下标: 出现次数}
        for i, toks in enumerate(tokenized):      # 第二遍扫描填 TF
            for term in toks:                     # 文档内每个 token（含重复）逐个登记
                self.tf_map[term][i] = self.tf_map[term].get(i, 0) + 1  # 该 term 在第 i 篇文档的词频 +1
        self.n_docs = n                           # 保存语料规模，scores() 要按它开数组


    def scores(self, query):
        out = np.zeros(self.n_docs, dtype="float32")  # 每篇文档的 BM25 得分初始化为 0
        for term in tokenize(query):              # 查询也用同一套分词，保证词表对齐
            idf = self.idf.get(term, 0.0)          # 查该 term 的 IDF；语料没见过的词权重为 0
            if idf == 0.0:                        # 语料中不存在的 term
                continue                          # 直接跳过，不贡献分数
            for i, tf in self.tf_map.get(term, {}).items():  # 沿倒排索引只遍历含该 term 的文档（稀疏加速）
                dl = self.doc_len[i]               # 该文档长度
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)  # 分母：TF 饱和项 × 长度归一化项
                out[i] += idf * tf * (self.k1 + 1) / denom  # 经典 BM25 打分公式逐项累加
        return out                                # 返回全部文档得分向量 [N]


class HybridRetriever:
    """Dense + Sparse 双路召回 → RRF 融合 → CrossEncoder 精排

    参数：
      faqs        : [{id, question, answer, variants, status}]（status ∈ active/gray/draft）
      dense_encode: callable(list[str]) -> np.ndarray [N, H]（已 L2 归一化）
      cross_score : 可选 callable(list[str], list[str]) -> list[float]（0~1 概率，
                    两个参数为等长列表，逐对打分；None 时退化为 RRF+余弦组合分）
      rrf_k       : RRF 平滑系数
    """

    def __init__(self, faqs, dense_encode, cross_score=None, rrf_k=60,
                 rerank_ratio=2, gray_weight=0.5):
        self.faqs = faqs                          # 原始 FAQ 列表（含答案，检索命中后取内容用）
        self.faq_by_id = {f["id"]: f for f in faqs}  # id → FAQ 条目的哈希表，O(1) 反查答案
        self.rrf_k = rrf_k                         # RRF 平滑常数（经验值 60，越大名次差异影响越平缓）
        self.rerank_ratio = rerank_ratio           # 精排候选扩比：召回 top_k*2 条送 CrossEncoder
        self.gray_weight = gray_weight             # 灰度条目的降权系数（0.5 = 融合分打对折）
        self._cross_score = cross_score            # CrossEncoder 逐对打分函数（None 则跳过精排）
        self._dense_encode_fn = dense_encode       # Dense 编码函数（ONNX/PyTorch 后端统一接口）

        self.texts, self.fid, self.status, self._norms = [], [], [], []  # 四条平行数组：索引文本/所属FAQ/状态/归一化文本
        for f in faqs:                             # 逐条 FAQ 展开成可检索文本
            if f.get("status", "active") not in ("active", "gray"):  # 只看 status（缺省视作 active）
                continue  # draft 不参与在线检索（灰度发布策略）  # draft 草稿不进索引
            for t in [f["question"]] + list(f.get("variants", [])):  # 标准问 + 变体问法都各自成一条索引项
                self.texts.append(t)               # 记录索引文本
                self.fid.append(f["id"])           # 记录该文本归属的 FAQ id（同 FAQ 多变体共享一个答案）
                self.status.append(f.get("status", "active"))  # 记录状态（gray 融合时降权）
                self._norms.append(normalize(t))   # 预计算归一化文本，评估 leave-one-out 时按它比对
        self.n = len(self.texts)                   # 索引条目总数
        self._dense = np.asarray(dense_encode(self.texts), dtype="float32")  # 一次性批量编码全部索引文本 → 矩阵 [N, H]（已归一化）
        assert self._dense.shape[0] == self.n, "dense_encode 返回行数与文本数不一致"  # 防御：编码器行数必须对齐（错位会导致后面全部错位）
        self.bm25 = BM25(self.texts)              # 在同一份文本上建 BM25 稀疏索引
        self._stats = {"searches": 0, "dense_only": 0, "sparse_only": 0, "joint": 0,
                       "low_conf": 0}              # 运行统计：搜索次数/单路命中/双路命中/低置信数
        self._timings = {"dense_ms": [], "sparse_ms": [], "ce_ms": []}  # 各阶段延迟采样列表，stats() 里算均值


    # ── 检索 ────────────────────────────────────────────────────────────
    def search(self, query, top_k=3, threshold=0.82, dense_threshold=0.82,
               exclude_texts=None):
        """返回 {"results": [...], "matched": bool, "confidence": float, "low_confidence": bool}

        exclude_texts: str 或 list[str]——检索时屏蔽与这些文本归一化相同的索引
        （评估时做 leave-one-out，避免查询问法在索引中直接命中自身造成泄漏）。
        """
        self._stats["searches"] += 1               # 搜索计数 +1（stats() 里的分母）
        skip = set()                               # 待屏蔽的索引下标集合
        if exclude_texts:                          # 需要做泄漏屏蔽时
            exs = [exclude_texts] if isinstance(exclude_texts, str) else exclude_texts  # 统一成列表（兼容传单个字符串）
            nq = {normalize(x) for x in exs}        # 把待屏蔽文本也归一化，保证与索引侧口径一致
            skip = {i for i, nt in enumerate(self._norms) if nt in nq}  # 归一化相同 → 命中自身 → 记入屏蔽集

        t0 = time.perf_counter()                   # Dense 阶段计时开始
        qv = np.asarray(self._dense_encode_one(query), dtype="float32").reshape(-1)  # 查询编码成向量并展平为 [H]
        dense_sims = self._dense @ qv                       # [N] 余弦（向量已归一化）  # 矩阵×向量一次算出与全部索引的余弦相似度（归一化后点积=余弦）
        self._timings["dense_ms"].append((time.perf_counter() - t0) * 1000)  # 记录 Dense 阶段耗时(ms)

        t0 = time.perf_counter()                   # Sparse 阶段计时开始
        bm25_scores = self.bm25.scores(query)      # BM25 对全库打分 [N]
        self._timings["sparse_ms"].append((time.perf_counter() - t0) * 1000)  # 记录 Sparse 阶段耗时

        if skip:                                   # 若存在屏蔽项
            for i in skip:                         # 逐个下标
                dense_sims[i] = -1e9               # Dense 分压成极小值 → 排名必落榜
                bm25_scores[i] = 0.0               # BM25 分清零 → 不进 sparse_rank

        # 双路 Top-k 排名
        k = min(max(top_k * self.rerank_ratio * 2, 8), self.n)  # 每路取多少名：top_k×4 且至少 8、至多全库（两路并集要够 CE 精排挑）
        dense_rank = {int(i): r for r, i in enumerate(np.argsort(-dense_sims)[:k], 1)
                      if int(i) not in skip}  # Dense 降序前 k 名 → {索引: 名次}（argsort 加负号实现降序；屏蔽项不参与排名，防止泄漏进候选尾部）
        order = np.argsort(-bm25_scores)[:k]       # BM25 降序前 k 名的下标序列
        sparse_rank = {int(i): r for r, i in enumerate(order, 1) if bm25_scores[i] > 0}  # 同样转成 {索引: 名次}，0 分（无共享词）的不给名次

        # RRF 融合（gray 状态降权）
        fused = {}                                 # 融合分字典：索引 → RRF 得分
        for idx, r in dense_rank.items():          # Dense 路名次贡献
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + r)  # RRF 公式：名次 r 的倒数贡献（rank1≈1/61）
        for idx, r in sparse_rank.items():         # Sparse 路名次贡献
            w = self.gray_weight if self.status[idx] == "gray" else 1.0  # 灰度条目贡献打折扣（谨慎放量）
            fused[idx] = fused.get(idx, 0.0) + w / (self.rrf_k + r)  # 加权 RRF 累加；双路都命中则两项相加

        pre = sorted(((fused[i] + 0.6 * float(dense_sims[i]), i) for i in fused),
                     reverse=True)                 # 粗排：RRF 分 + 0.6×余弦兜底排序（纯 RRF 分不出并列时可依赖语义相似度）
        cand = pre[:max(top_k * self.rerank_ratio, top_k)]  # 取前 top_k×2 个候选送精排（多召回一些让 CE 有挑选余地）

        # CrossEncoder 精排（可选）
        t0 = time.perf_counter()                   # 精排阶段计时开始
        ce_scores = None                           # 默认无 CE 分数（未注入 cross_score 时保持 None）
        if self._cross_score is not None:          # 注入了 CrossEncoder 才精排
            idxs = [i for _, i in cand]            # 解包候选下标列表
            ce_scores = self._cross_score([query] * len(idxs), [self.texts[i] for i in idxs])  # (query, 候选) 逐对批量打分，返回 0~1 概率
            scored = sorted(zip(ce_scores, idxs),  # 把 CE 分数与下标配对
                            key=lambda p: 0.8 * float(dense_sims[p[1]]) + 0.2 * p[0],  # 最终排序键 = 0.8×余弦 + 0.2×CE 概率（dense 为主、CE 微调）
                            reverse=True)          # 按组合分降序排列
            self._timings["ce_ms"].append((time.perf_counter() - t0) * 1000)  # 记录 CE 精排耗时
        else:
            scored = list(cand)                    # 无 CE：直接沿用粗排顺序
            self._timings["ce_ms"].append(0.0)     # CE 耗时记 0（保持各查询延迟数组等长）

        # 按 FAQ 去重取 Top-K
        results, seen = [], set()                  # 结果列表 + 已收录 FAQ id 集合
        for score, idx in scored:                  # 按精排顺序遍历
            if self.fid[idx] in seen:               # 同一 FAQ 的另一变体已入选
                continue                           # 跳过（一个 FAQ 只出一条答案）
            seen.add(self.fid[idx])                # 标记该 FAQ 已收录
            f = self.faq_by_id[self.fid[idx]]      # 反查 FAQ 条目拿标准问与答案
            # 领域闸门置信度（dense 主导，CE 双方一致判正才按 CE 尺度）：
            #  - CE 判负 或 dense 语义过低（离题问法）→ 置信度回落到 dense 尺度，低于阈值即拦截
            #  - 两者一致判正 → 0.75*CE + 0.25*dense，保持 CE 概率域的校准语义
            d_cos = min(max(float(dense_sims[idx]), 0.0), 1.0)  # 余弦裁剪到 [0,1]（负相似度在此场景视为 0）
            c_prob = float(score) if ce_scores is not None else None  # CE 概率；无 CE 时为 None
            if c_prob is not None and c_prob >= 0.5 and d_cos >= 0.45:  # 双闸门：CE 判正(≥0.5) 且 dense 不太低(≥0.45)
                conf = 0.75 * c_prob + 0.25 * d_cos  # 双方一致 → 以 CE 概率为主的混合置信度
            elif c_prob is not None:               # 有 CE 但任一闸门未过（CE 判负或语义离题）
                conf = 0.85 * d_cos                # 回落到纯 dense 尺度 → 难过阈值，实现"拦回答"效果
            else:
                conf = d_cos                       # 无 CE 时置信度就是余弦本身
            results.append({                       # 组装单条结果 dict
                "faq_id":     f["id"],              # 命中的 FAQ 编号
                "question":   f["question"],        # 库内标准问法
                "answer":     f["answer"],          # 标准答案（最终返回给用户的内容）
                "matched_text": self.texts[idx],    # 实际匹配上的索引文本（可能是变体）
                "status":     self.status[idx],      # 命中条目的发布状态
                "confidence": round(conf, 4),       # 置信度（保留 4 位小数）
                "score":      round(conf, 4),       # score 与 confidence 同值（兼容不同消费方字段名）
                "dense_sim":  round(float(dense_sims[idx]), 4),  # Dense 余弦相似度（诊断用）
                "bm25_score": round(float(bm25_scores[idx]), 4),  # BM25 原始分（诊断用）
                "dense_rank": dense_rank.get(idx),   # 在 Dense 路的名次（不在 Top-k 则为 None）
                "sparse_rank": sparse_rank.get(idx),  # 在 BM25 路的名次
                "i_dense": idx in dense_rank,       # 是否进入 Dense 路 Top-k
                "i_sparse": idx in sparse_rank,     # 是否进入 Sparse 路 Top-k
            })
            if len(results) >= top_k:              # 攒够 top_k 条
                break                             # 提前结束循环

        top = results[0] if results else {}        # 首条结果（无命中给空 dict）
        conf = top.get("confidence", 0.0)          # 首条置信度（无结果视作 0）
        use_ce = ce_scores is not None             # 本次检索是否用了 CE 尺度
        matched = conf >= (threshold if use_ce else dense_threshold)  # 阈值判匹配：CE/无CE 用不同阈值，低于阈值则建议拒答/转人工
        if top:                                    # 有命中时统计哪路贡献的 Top1
            if top["i_dense"] and top["i_sparse"]:  # 两路都在榜
                self._stats["joint"] += 1          # 双路共同命中（理想情况）
            elif top["i_dense"]:                   # 仅 Dense 在榜
                self._stats["dense_only"] += 1     # 语义路独立命中
            elif top["i_sparse"]:                  # 仅 BM25 在榜
                self._stats["sparse_only"] += 1    # 字面路独立命中（如型号精确匹配场景）
        if not matched:                            # 首条就低置信
            self._stats["low_conf"] += 1           # 低置信计数（衡量拒答率）
        return {"results": results, "matched": bool(matched),  # 返回结果列表与匹配标志
                "confidence": round(conf, 4),      # 首条置信度（服务层据此决定直接答/兜底话术）
                "low_confidence": not matched}     # 低置信标志（matched 的取反，方便调用方直接判）

    def _dense_encode_one(self, text):
        return self._dense_encode_fn([text])[0]   # 把单条查询包成列表送编码器（后端接口按批设计），再取回第 0 行

    def stats(self):
        """汇总本次运行统计：命中贡献分布、低置信率、分阶段平均延迟与索引规模。"""
        s = dict(self._stats)                      # 拷贝计数器（避免外部误改内部状态）
        t = s["searches"] or 1                     # 归一化分母（0 次搜索时用 1 防除零）
        return {                                   # 组装汇总报告 dict
            **s,                                   # 原样带上各项原始计数
            "hit_distribution": {                 # Top1 命中来源分布（占比）
                "dense_only": round(s["dense_only"] / t, 3),   # 仅 Dense 命中占比
                "sparse_only": round(s["sparse_only"] / t, 3),  # 仅 BM25 命中占比
                "joint": round(s["joint"] / t, 3),  # 双路命中占比
            },
            "low_conf_rate": round(s["low_conf"] / t, 3),  # 低置信（近拒答）比例
            "avg_latency_ms": {                   # 各阶段平均延迟（毫秒）
                "dense": round(float(np.mean(self._timings["dense_ms"] or [0])), 2),   # Dense 编码+打分均值；空列表兜底 0
                "sparse": round(float(np.mean(self._timings["sparse_ms"] or [0])), 2),  # BM25 打分均值
                "ce": round(float(np.mean(self._timings["ce_ms"] or [0])), 2),         # CE 精排均值
            },
            "index_size": self.n,                  # 索引条目数（线上容量参考）
        }
