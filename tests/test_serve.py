"""serve.py 在线服务端点测试

策略：TestClient 不进入 with 上下文 → 不触发 lifespan（避免加载真实模型），
手工填充全局状态 S（Fake 编码器 + 临时 KB + 临时标注队列），
真实跑通 HybridRetriever 检索链路与全部管理端点。
"""

import json

import pytest
from active_learning import AnnotationQueue
from fastapi.testclient import TestClient

KB = [
    {"id": 1, "question": "血压计怎么测量血压", "variants": ["如何用血压计测血压"],
     "answer": "静坐五分钟后测量。", "status": "active"},
    {"id": 2, "question": "错误码E1是什么意思", "variants": [],
     "answer": "E1 表示袖带漏气。", "status": "active"},
    {"id": 3, "question": "血压计怎么校准", "variants": [],
     "answer": "请联系售后。", "status": "gray"},
    {"id": 4, "question": "袖带怎么清洗", "variants": [],
     "answer": "清水擦拭。", "status": "draft"},
]


class NullLock:
    """跨请求安全的不加锁替身（TestClient 每个请求独立事件循环，asyncio.Lock 会跨循环绑定）。"""

    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


@pytest.fixture()
def client(tmp_path, monkeypatch, fake_dense):
    kb_file = tmp_path / "kb.json"
    kb_file.write_text(json.dumps(KB, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("BP_KB", str(kb_file))       # load_kb/rebuild_index 走临时库
    monkeypatch.setenv("BP_AUDIT_PATH", str(tmp_path / "audit.jsonl"))  # P4：管理端点审计重定向临时目录，测试零仓库污染

    import serve
    serve.S.update(
        dense_enc=fake_dense,
        cross_score=None,
        queue=AnnotationQueue(data_dir=tmp_path / "fb"),
        threshold=0.82,
        top_k=3,
        lock=NullLock(),
        lat=[],
    )
    serve.rebuild_index()                            # 用 Fake 编码器建混合检索索引
    yield TestClient(serve.app)                      # 不进上下文 → 不触发 lifespan
    serve.S.clear()


class TestQuery:
    def test_matched(self, client):
        r = client.post("/query", json={"query": "血压计怎么测量血压"})
        assert r.status_code == 200
        body = r.json()
        assert body["matched"] is True
        assert body["faq_id"] == 1
        assert body["answer"] == "静坐五分钟后测量。"
        assert body["threshold"] == 0.82
        assert body["elapsed_ms"] >= 0
        assert body["candidates"]

    def test_low_confidence_intercepts_and_queues(self, client):
        import serve
        r = client.post("/query", json={"query": "今天股市行情怎么样"})
        body = r.json()
        assert body["matched"] is False
        assert body["low_confidence"] is True
        assert "人工客服" in body["answer"]           # 兜底话术 + 医疗免责
        assert body["faq_id"] is None
        rows = serve.S["queue"].list("pending")       # 低置信已异步入队
        assert len(rows) == 1 and rows[0]["query"] == "今天股市行情怎么样"

    def test_empty_query_400(self, client):
        assert client.post("/query", json={"query": "   "}).status_code == 400


class TestObservability:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["dense_backend"] == "fake"
        assert body["kb_size"] == 4
        assert body["index_size"] == 4                # draft 不进索引
        assert body["threshold"] == 0.82

    def test_stats(self, client):
        client.post("/query", json={"query": "血压计怎么测量血压"})
        client.post("/query", json={"query": "今天股市行情怎么样"})
        body = client.get("/stats").json()
        assert body["searches"] == 2
        assert body["low_conf_rate"] == 0.5
        assert body["e2e_latency_ms"]["recent"] == 2
        assert body["queue"]["total"] == 1


class TestAnnotations:
    def test_list_and_attach_merges_variant_into_kb(self, client):
        import serve
        rid = serve.S["queue"].append("袖带如何弄干净", 0.3, [])["id"]
        assert client.get("/annotations").json()["total"] == 1

        r = client.post(f"/annotations/{rid}", json={"action": "attach", "faq_id": 1})
        body = r.json()
        assert body == {"ok": True, "merged": True, "faq_id": 1}
        assert serve.S["queue"].list()[0]["status"] == "attached"
        kb = json.loads(serve.S["kb_path"].read_text(encoding="utf-8"))
        faq1 = next(f for f in kb if f["id"] == 1)
        assert "袖带如何弄干净" in faq1["variants"]   # 问法并回 KB 落盘

    def test_attach_requires_faq_id(self, client):
        import serve
        rid = serve.S["queue"].append("q", 0.2, [])["id"]
        r = client.post(f"/annotations/{rid}", json={"action": "attach"})
        assert r.status_code == 400

    def test_reject_flow(self, client):
        import serve
        rid = serve.S["queue"].append("乱七八糟", 0.1, [])["id"]
        r = client.post(f"/annotations/{rid}", json={"action": "reject"})
        assert r.json() == {"ok": True}
        assert serve.S["queue"].list()[0]["status"] == "rejected"

    def test_unknown_action_400(self, client):
        import serve
        rid = serve.S["queue"].append("q", 0.2, [])["id"]
        assert client.post(f"/annotations/{rid}", json={"action": "boom"}).status_code == 400


class TestIterate:
    def test_empty_pending(self, client):
        assert client.get("/iterate").json() == {"clusters": [], "note": "队列为空"}

    def test_cluster_preview_and_apply_draft(self, client):
        import serve
        for q in ("袖带怎么清洗", "袖带如何清洗", "怎么清洗袖带"):
            serve.S["queue"].append(q, 0.3, [])

        body = client.get("/iterate").json()
        assert body["pending"] == 3
        assert body["qualify_draft"] == 1             # 三条同义问法聚成一簇

        r = client.post("/iterate/apply")
        drafts = r.json()["created_drafts"]
        assert len(drafts) == 1 and drafts[0]["id"] == 5
        kb = json.loads(serve.S["kb_path"].read_text(encoding="utf-8"))
        assert kb[-1]["status"] == "draft"            # 新条目为 draft，不影响线上


class TestAdmin:
    def test_promote_rebuilds_index(self, client):
        import serve
        before = serve.S["retriever"].n
        r = client.post("/admin/promote", json={"faq_ids": [4]})
        body = r.json()
        assert body["promoted"] == [4]                # draft → active
        assert body["index_rebuilt"] is True
        assert body["index_size"] == before + 1       # 转正后进入在线索引
        assert client.get("/health").json()["kb_size"] == 4

    def test_refresh(self, client):
        body = client.post("/admin/refresh").json()
        assert body["ok"] is True
        assert body["kb_size"] == 4
        assert body["index_size"] == 4
