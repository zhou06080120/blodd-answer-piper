"""active_learning 单测：AnnotationQueue 队列流转 / IntentClustering 聚类 / ActiveLearning 迭代闭环"""

import json

import numpy as np
import pytest
from active_learning import ActiveLearning, AnnotationQueue, IntentClustering


@pytest.fixture()
def queue(tmp_path):
    return AnnotationQueue(data_dir=tmp_path / "fb")


# ── AnnotationQueue 待标注队列 ──────────────────────────────────────────────

class TestAnnotationQueue:
    def test_append_creates_file_and_record(self, queue):
        rec = queue.append("血压计怎么用", 0.42123, [f"c{i}" for i in range(7)])
        assert queue.path.exists()
        assert rec["status"] == "pending"
        assert len(rec["id"]) == 12                    # 12 位十六进制 id
        assert rec["confidence"] == 0.4212             # 4 位小数
        assert len(rec["candidates"]) == 5             # 候选截前 5
        rows = queue.list()
        assert len(rows) == 1 and rows[0]["query"] == "血压计怎么用"

    def test_list_status_filter_and_limit(self, queue):
        ids = [queue.append(f"问题{i}", 0.3, [])["id"] for i in range(6)]
        queue.reject(ids[0])
        assert len(queue.list(status="pending")) == 5
        tail = queue.list(limit=2)
        assert [r["query"] for r in tail] == ["问题4", "问题5"]   # 最近优先

    def test_label_flows(self, queue):
        rid = queue.append("q", 0.2, [])["id"]
        queue.label_attach(rid, 8)
        r = queue.list()[0]
        assert r["status"] == "attached" and r["faq_id"] == 8

        queue.label_new_intent(rid, question="标准新问法")
        r = queue.list()[0]
        assert r["status"] == "new_intent" and r["query"] == "标准新问法"
        assert r["faq_id"] is None

        queue.reject(rid)
        assert queue.list()[0]["status"] == "rejected"

    def test_stats(self, queue):
        a = queue.append("q1", 0.1, [])["id"]
        queue.append("q2", 0.1, [])
        queue.reject(a)
        assert queue.stats() == {"rejected": 1, "pending": 1, "total": 2}

    def test_empty_queue_and_dir_selfheal(self, tmp_path):
        q = AnnotationQueue(data_dir=tmp_path / "not_exist_yet")   # 目录自愈
        assert q.list() == []
        assert q.stats()["total"] == 0

    def test_update_nonexistent_id_is_noop(self, queue):
        queue.append("q", 0.1, [])
        queue.update("no-such-id", status="rejected")
        assert queue.list()[0]["status"] == "pending"


# ── IntentClustering 意图聚类 ───────────────────────────────────────────────

class TestIntentClustering:
    def test_empty_and_single_record(self):
        clu = IntentClustering()
        empty = clu.cluster([], lambda ts: np.zeros((0, 2), dtype="float32"))
        assert empty[0]["size"] == 0 and empty[0]["avg_conf"] == 0

        single = clu.cluster([{"query": "q", "confidence": 0.3, "status": "pending"}],
                             lambda ts: np.array([[1.0, 0]], dtype="float32"))
        assert len(single) == 1 and single[0]["size"] == 1
        assert single[0]["avg_conf"] == 0.3

    def test_two_clusters_and_pending_filter(self):
        vecs = {"清洗袖带": [1, 0], "袖带咋清洗": [0.95, 0.05],
                "错误码E1": [0, 1], "E1啥意思": [0.05, 0.95]}
        recs = [{"query": q, "confidence": 0.3, "status": "pending"} for q in vecs]
        recs.append({"query": "已处理过的问题", "confidence": 0.9, "status": "attached"})

        def fake_encode(texts):
            return np.array([vecs[t] for t in texts], dtype="float32")

        summary = IntentClustering(dist_threshold=0.35).cluster(recs, fake_encode)
        assert len(summary) == 2                              # 两簇（attached 被过滤）
        assert sorted(s["size"] for s in summary) == [2, 2]
        assert all(len(s["samples"]) <= 3 for s in summary)
        assert all(s["avg_conf"] == 0.3 for s in summary)     # 簇内平均置信


# ── ActiveLearning 周级迭代 ─────────────────────────────────────────────────

@pytest.fixture()
def kb_file(tmp_path):
    kb = [
        {"id": 1, "question": "q1", "variants": [], "answer": "a1", "status": "active"},
        {"id": 2, "question": "q2", "variants": [], "answer": "a2", "status": "gray"},
        {"id": 3, "question": "q3", "variants": [], "answer": "a3", "status": "draft"},
    ]
    p = tmp_path / "kb.json"
    p.write_text(json.dumps(kb, ensure_ascii=False), encoding="utf-8")
    return p


class TestActiveLearning:
    def test_apply_clusters_creates_draft(self, kb_file, tmp_path):
        al = ActiveLearning(kb_path=kb_file, feedback_dir=tmp_path / "fb")
        summary = [{"cluster_id": 0, "size": 3, "samples": ["n1", "n2", "n3"], "avg_conf": 0.3}]
        created = al.apply_clusters(summary)
        assert len(created) == 1
        d = created[0]
        assert d["id"] == 4                              # 现有最大 id 顺延
        assert d["status"] == "draft"                    # 草稿态不进线上索引
        assert d["question"] == "n1" and d["variants"] == ["n2", "n3"]
        assert "人工补充审核" in d["answer"]               # 答案占位提示
        assert d["cluster_size"] == 3
        assert kb_file.with_suffix(".json.bak").exists()  # 写库前自动备份
        assert len(al.load_kb()) == 4

    def test_apply_clusters_min_size_filter(self, kb_file, tmp_path):
        al = ActiveLearning(kb_path=kb_file, feedback_dir=tmp_path / "fb")
        summary = [{"cluster_id": 0, "size": 2, "samples": ["n1", "n2"], "avg_conf": 0.3}]
        assert al.apply_clusters(summary, min_size=3) == []   # 不达标簇不入库

    def test_promote_draft_and_gray(self, kb_file, tmp_path):
        al = ActiveLearning(kb_path=kb_file, feedback_dir=tmp_path / "fb")
        assert al.promote([2, 3]) == [2, 3]
        faqs = al.load_kb()
        assert all(f["status"] == "active" for f in faqs if f["id"] in (2, 3))
        assert al.promote([2]) == []                     # 已 active 不重复转正
        assert al.promote([99]) == []                    # 不存在的 id
