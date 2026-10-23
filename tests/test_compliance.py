"""compliance 医疗合规模块测试（P3）

策略：
  - validate_query 纯函数单测：正常透传 / 空输入拒绝 / 控制字符剥离 / 长度边界；
  - API 集成测试沿用 test_serve.py 手法：TestClient 不进 with 上下文（不触发
    lifespan 加载真实模型），手工填充全局状态 S（Fake 编码器 + 临时 KB +
    临时标注队列），真实跑通 /query 的 400 校验分支与免责声明字段；
  - 检索服务无 LLM 生成、天然免疫 Prompt 注入，故不设注入拦截用例。
"""

import json

import pytest
from active_learning import AnnotationQueue
from compliance import MEDICAL_DISCLAIMER, ComplianceError, validate_query
from fastapi.testclient import TestClient

KB = [  # 与 test_serve.py 同款 KB：保证低置信拦截用例行为可复现
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


class TestValidateQuery:
    """validate_query 单测：正常文本 / 空拒 / 控制字符剥离 / 长度边界。"""

    def test_normal_text_passthrough(self):
        assert validate_query("  血压计怎么测量血压  ") == "血压计怎么测量血压"  # strip 首尾空白后原样返回

    def test_empty_rejected(self):
        for bad in (None, "", "   ", "\x00\x07"):    # 缺字段 / 空串 / 纯空白 / 纯控制字符
            with pytest.raises(ComplianceError) as e:  # 一律拒绝
                validate_query(bad)
            assert e.value.code == "invalid_input"   # 错误码：无效输入

    def test_control_chars_stripped(self):
        assert validate_query("血\x00压\x0b计\x07怎么用") == "血压计怎么用"  # 控制字符剥离（\t\n\r 保留）

    def test_length_boundary(self):
        assert validate_query("压" * 500) == "压" * 500  # 恰好 500 字（默认上限）：放行
        with pytest.raises(ComplianceError) as e:     # 501 字：超长拒绝
            validate_query("压" * 501)
        assert e.value.code == "input_too_long"       # 错误码：输入超长
        with pytest.raises(ComplianceError):          # 自定义上限覆盖同样生效
            validate_query("压" * 11, max_length=10)


class TestQueryCompliance:
    """/query 合规集成：空/超长 400 校验分支 + 命中/拦截响应均带免责声明。"""

    def test_empty_query_400(self, client):
        r = client.post("/query", json={"query": "   "})
        assert r.status_code == 400                   # 空白问句：校验拦截（不进检索）
        assert "detail" in r.json()                   # 返回具体原因文案

    def test_too_long_query_400(self, client):
        r = client.post("/query", json={"query": "压" * 501})
        assert r.status_code == 400                   # 超长问句：校验拦截
        assert "过长" in r.json()["detail"]            # 错误码 input_too_long 的文案

    def test_matched_query_has_disclaimer(self, client):
        body = client.post("/query", json={"query": "血压计怎么测量血压"}).json()
        assert body["matched"] is True                # 正常命中
        assert body["disclaimer"] == MEDICAL_DISCLAIMER  # 命中响应也统一附带免责声明

    def test_low_confidence_has_disclaimer(self, client):
        body = client.post("/query", json={"query": "今天股市行情怎么样"}).json()
        assert body["matched"] is False               # 低置信拦截（兜底话术内嵌免责不变）
        assert body["disclaimer"] == MEDICAL_DISCLAIMER  # 拦截响应同样附带免责声明
