"""hybrid_retriever 单测：tokenize / BM25 / HybridRetriever（Fake 编码器 + Fake 精排）"""

import math

import numpy as np
import pytest
from hybrid_retriever import BM25, HybridRetriever, tokenize
from llm_synth import normalize

FAQS = [
    {"id": 1, "question": "血压计怎么测量血压", "variants": ["如何用血压计测血压"],
     "answer": "静坐五分钟后测量。", "status": "active"},
    {"id": 2, "question": "错误码E1是什么意思", "variants": [],
     "answer": "E1 表示袖带漏气。", "status": "active"},
    {"id": 3, "question": "血压计怎么校准", "variants": [],
     "answer": "请联系售后。", "status": "gray"},
    {"id": 4, "question": "袖带怎么清洗", "variants": [],
     "answer": "清水擦拭。", "status": "draft"},   # draft 不应进入索引
]


def make_retriever(fake_dense, cross_score=None, **kw):
    return HybridRetriever(FAQS, dense_encode=fake_dense.encode,
                           cross_score=cross_score, **kw)


# ── 分词 ────────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_latin_model_tokens_kept_whole(self):
        toks = tokenize("HEM-7156 血压 140/90mmHg")
        assert "hem-7156" in toks          # 型号整词保留 + 小写
        assert "140/90mmhg" in toks        # 数值/单位整词保留
        assert "血" in toks and "血压" in toks  # 中文 1-gram + 2-gram

    def test_cjk_bigrams(self):
        toks = tokenize("怎么测量")
        for t in ["怎", "么", "测", "量", "怎么", "么测", "测量"]:
            assert t in toks

    def test_lowercase_latin(self):
        assert tokenize("ABC") == ["abc"]


# ── BM25 ────────────────────────────────────────────────────────────────────

class TestBM25:
    CORPUS = ["HEM-7156 血压计怎么测量", "错误码 E1 是什么意思", "血压计怎么校准"]

    def test_exact_token_ranks_first(self):
        bm = BM25(self.CORPUS)
        assert int(np.argmax(bm.scores("HEM-7156"))) == 0

    def test_query_hits_second_doc(self):
        bm = BM25(self.CORPUS)
        assert int(np.argmax(bm.scores("错误码 E1 什么意思"))) == 1

    def test_unknown_term_zero_scores(self):
        bm = BM25(self.CORPUS)
        assert float(np.sum(bm.scores("zzzqqq xxx"))) == 0.0

    def test_idf_always_positive(self):
        bm = BM25(self.CORPUS)
        assert all(v > 0 for v in bm.idf.values())


# ── 索引构建 ────────────────────────────────────────────────────────────────

class TestIndex:
    def test_draft_excluded_variants_indexed(self, fake_dense):
        r = make_retriever(fake_dense)
        # FAQ1（标准问+1 变体）+ FAQ2 + FAQ3(gray) = 4 条；draft FAQ4 不进索引
        assert r.n == 4
        assert all("清洗" not in t for t in r.texts)

    def test_faq_by_id_lookup(self, fake_dense):
        r = make_retriever(fake_dense)
        assert r.faq_by_id[2]["answer"] == "E1 表示袖带漏气。"


# ── 检索 ────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_exact_query_matched(self, fake_dense):
        r = make_retriever(fake_dense)
        ret = r.search("血压计怎么测量血压", top_k=3)
        assert ret["matched"] is True
        assert ret["low_confidence"] is False
        top = ret["results"][0]
        assert top["faq_id"] == 1
        assert top["question"] == "血压计怎么测量血压"
        assert top["answer"] == "静坐五分钟后测量。"
        assert top["confidence"] == pytest.approx(1.0, abs=1e-3)   # 完全同句 → 余弦≈1

    def test_faq_dedup_one_result_per_faq(self, fake_dense):
        r = make_retriever(fake_dense)
        ret = r.search("血压计怎么测量血压", top_k=5)
        ids = [x["faq_id"] for x in ret["results"]]
        assert len(ids) == len(set(ids))          # 一个 FAQ 只出一条
        assert set(ids) == {1, 2, 3}              # 全部可召回 FAQ

    def test_threshold_intercepts_off_topic(self, fake_dense):
        r = make_retriever(fake_dense)
        ret = r.search("今天股市行情怎么样", top_k=3)
        assert ret["matched"] is False
        assert ret["low_confidence"] is True
        assert ret["confidence"] < 0.82

    def test_gray_entry_searchable_with_status(self, fake_dense):
        r = make_retriever(fake_dense)
        ret = r.search("血压计怎么校准", top_k=3)
        top = ret["results"][0]
        assert top["faq_id"] == 3
        assert top["status"] == "gray"

    def test_exclude_texts_masks_self(self, fake_dense):
        """leave-one-out 泄漏屏蔽：排除标准问后应命中变体而非自身。"""
        r = make_retriever(fake_dense)
        ret = r.search("血压计怎么测量血压", top_k=3,
                       exclude_texts="血压计怎么测量血压")
        top = ret["results"][0]
        assert top["matched_text"] == "如何用血压计测血压"
        assert normalize(top["matched_text"]) != normalize("血压计怎么测量血压")
        assert top["faq_id"] == 1

    def test_exclude_texts_list_masks_faq_entirely(self, fake_dense):
        r = make_retriever(fake_dense)
        ret = r.search("血压计怎么测量血压", top_k=3,
                       exclude_texts=["血压计怎么测量血压", "如何用血压计测血压"])
        assert all(x["faq_id"] != 1 for x in ret["results"])


# ── CrossEncoder 精排与置信度闸门 ───────────────────────────────────────────

class TestCrossEncoder:
    def test_ce_confidence_formula(self, fake_dense):
        """双闸门通过（CE≥0.5 且 dense≥0.45）：conf = 0.75*CE + 0.25*cos。"""
        r = make_retriever(fake_dense, cross_score=lambda q, t: [0.9] * len(t))
        ret = r.search("血压计怎么测量血压", top_k=3)
        top = ret["results"][0]
        assert top["confidence"] == pytest.approx(0.75 * 0.9 + 0.25 * 1.0, abs=1e-3)

    def test_ce_negative_falls_back_to_dense_scale(self, fake_dense):
        """CE 判负：conf 回落 0.85*dense 尺度。"""
        r = make_retriever(fake_dense, cross_score=lambda q, t: [0.1] * len(t))
        ret = r.search("血压计怎么测量血压", top_k=3)
        assert ret["results"][0]["confidence"] == pytest.approx(0.85, abs=1e-3)

    def test_low_dense_gated_even_with_positive_ce(self, fake_dense):
        """CE 判正但 dense 语义过低（<0.45）：按 dense 尺度拦截。"""
        gate_faqs = [
            {"id": 1, "question": "甲乙丙丁戊", "variants": [], "answer": "A", "status": "active"},
            {"id": 2, "question": "子丑寅卯辰", "variants": [], "answer": "B", "status": "active"},
        ]
        r = HybridRetriever(gate_faqs, dense_encode=fake_dense.encode,
                            cross_score=lambda q, t: [0.9] * len(t))
        ret = r.search("甲", top_k=2)
        # cos(甲, 甲乙丙丁戊) = 1/√5 ≈ 0.447 < 0.45 闸门 → conf = 0.85*cos
        assert ret["results"][0]["faq_id"] == 1
        assert ret["results"][0]["confidence"] == pytest.approx(0.85 / math.sqrt(5), abs=1e-3)
        assert ret["low_confidence"] is True


# ── 运行统计 ────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_counters(self, fake_dense):
        r = make_retriever(fake_dense)
        r.search("血压计怎么测量血压")
        r.search("今天股市行情怎么样")
        s = r.stats()
        assert s["searches"] == 2
        assert s["low_conf"] == 1
        assert s["index_size"] == r.n
        assert set(s["hit_distribution"]) == {"dense_only", "sparse_only", "joint"}
        assert all(v >= 0 for v in s["avg_latency_ms"].values())
