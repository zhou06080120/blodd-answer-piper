"""observability 可观测性模块测试（P2）

策略：
  - Prometheus 指标是全进程全局单例（observability.py 模块级 Counter/Histogram）：
    所有断言一律用「增量」而非绝对值，避免与其他用例/app 实例互相污染；
  - /metrics 与 /health 端点不依赖 serve 的全局状态 S：直接 TestClient(serve.app)
    请求即可（不进 with 上下文 → 不触发 lifespan，与 test_serve.py 同款用法）；
  - 纯单元部分（JsonFormatter / setup_logging / llm_call_timer）直接调用，不经 HTTP；
  - 限流 429 计数：monkeypatch 设 RATE_LIMIT_RPM=1 自建独立 app，
    并复现生产注册顺序（先 install_rate_limit 再 install_observability）——
    后者在外层，限流器返回的 429 才会流经可观测中间件被统计。
"""

import json     # JsonFormatter 输出行的反序列化校验
import logging  # 构造 LogRecord / 检查根 logger handler
import re       # request_id 的 12 位十六进制格式校验

import pytest   # raises 断言异常上抛
from fastapi import FastAPI  # 限流计数用例自建独立应用
from fastapi.testclient import TestClient  # 不进 with：不触发 lifespan
from observability import (JsonFormatter, install_observability, llm_call_timer,  # noqa: E402 (pythonpath 注入 src)
                           request_id_var, setup_logging)
from prometheus_client import generate_latest  # 直接抓进程内指标文本（与 /metrics 同源）
from rate_limit import install_rate_limit  # noqa: E402

import serve  # noqa: E402  # serve.app：/health /metrics /annotations/{rid} 三条路由


def _metrics_text():
    """抓取当前进程全部 Prometheus 指标的文本表示（与 GET /metrics 响应体同源）。"""
    return generate_latest().decode("utf-8")


def _counter_value(text, name, label=None):
    """从 /metrics 文本累加以 name 开头的样本行尾数值。

    精确前缀匹配：只认 name+"{" 或 name+" "（后带空格）两种行首，
    自动排除 _created/_bucket/_sum 等衍生序列与 HELP/TYPE 注释行；
    label 可选：进一步过滤含特定标签串的行（如 'status="ok"'、'path="/health"'）。
    """
    total = 0.0                                             # 累加器
    for line in text.splitlines():                          # 逐行扫描
        if not (line.startswith(name + "{") or line.startswith(name + " ")):  # 精确前缀
            continue                                        # 注释行 / 衍生序列 / 其他指标全部跳过
        if label and label not in line:                     # 需要按标签过滤而未命中
            continue                                        # 跳过
        total += float(line.rsplit(" ", 1)[1])              # 行尾数值（空格分隔的最后一段）
    return total                                            # 返回该指标（可选标签）累计值


class TestMetricsEndpoint:
    """/metrics 抓取端点：Prometheus 文本协议 + HTTP/限流两计数器序列齐备。"""

    def test_metrics_exposes_http_and_rate_limit_counters(self):
        client = TestClient(serve.app)                      # 不进上下文 → 不触发 lifespan
        r = client.get("/metrics")                          # 抓取端点自身限流豁免（EXEMPT_PATHS）
        assert r.status_code == 200                         # 探针必须常绿
        assert "text/plain" in r.headers["content-type"]     # Prometheus 文本协议
        assert "http_requests_total" in r.text               # HTTP 请求计数器已暴露
        assert "rate_limit_rejected_total" in r.text          # 限流拒绝计数器已暴露


class TestRequestId:
    """X-Request-ID：自动生成 12 位十六进制；网关透传时原样回写响应头。"""

    def test_health_auto_generates_request_id(self):
        client = TestClient(serve.app)
        rid = client.get("/health").headers["x-request-id"]  # 未带请求头 → 中间件自动生成
        assert re.fullmatch(r"[0-9a-f]{12}", rid)            # uuid4().hex[:12]：12 位小写十六进制

    def test_request_id_passthrough(self):
        client = TestClient(serve.app)
        r = client.get("/health", headers={"X-Request-ID": "abc123"})  # 网关已生成 ID
        assert r.headers["x-request-id"] == "abc123"          # 响应头原样回传，全链路日志按它串联


class TestCounters:
    """计数器解析辅助 + 请求计数增量（全局单例指标一律断言增量）。"""

    def test_counter_value_helper_and_health_increment(self):
        # 先用伪造文本验证 _counter_value 的精确前缀口径：排除衍生序列与注释行
        fake = (
            "# HELP http_requests_total HTTP 请求总数\n"        # 注释行：前缀是 "#"，不匹配
            "# TYPE http_requests_total counter\n"             # 同上
            'http_requests_total{method="GET",path="/health",status="200"} 2.0\n'  # 正常样本行
            'http_requests_total{method="GET",path="/stats",status="200"} 3.5\n'  # 第二条样本行（应累加）
            "http_requests_total_created 123.0\n"             # _created 衍生：第 21 字符是 "_"，精确前缀排除
            "http_request_duration_seconds_sum 9.9\n"          # 直方图衍生：名字不同，不匹配
            "rate_limit_rejected_total 1.0\n"                 # 其他指标：不匹配 http 前缀
        )
        assert _counter_value(fake, "http_requests_total") == 5.5      # 2.0 + 3.5，衍生序列不计入
        assert _counter_value(fake, "rate_limit_rejected_total") == 1.0  # 换个指标名同样可解析

        # 再打一次真实 /health，按 path 标签断言该行数值精确 +1
        client = TestClient(serve.app)
        before = _counter_value(client.get("/metrics").text, "http_requests_total",
                                'path="/health"')            # 读取前的 /health 行累计值
        client.get("/health")                                 # 本次被统计的请求
        after = _counter_value(client.get("/metrics").text, "http_requests_total",
                               'path="/health"')             # 读取后的累计值
        assert after == before + 1                            # 恰好 +1（/metrics 自身计入 path="/metrics" 行，互不干扰）


class TestRateLimitMetrics:
    """限流 429 拒绝计数：限流中间件在内层，429 响应流经外层可观测中间件被统计。"""

    def test_429_rejection_counted_and_metrics_exempt(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RPM", "1")            # 每分钟仅放行 1 次
        app = FastAPI()                                       # 自建独立应用，不碰 serve.app

        @app.get("/ping")
        def ping():
            return {"ok": True}

        install_rate_limit(app)                               # 先注册限流（内层）
        install_observability(app)                            # 后注册可观测（外层）→ 429 也被统计
        client = TestClient(app)                              # 不进 with：同款用法

        assert client.get("/ping").status_code == 200         # 第 1 次放行（占满配额）
        before = _counter_value(client.get("/metrics").text, "rate_limit_rejected_total")  # /metrics 限流豁免，不占配额
        assert client.get("/ping").status_code == 429         # 第 2 次超阈值 → 拒绝
        after = _counter_value(client.get("/metrics").text, "rate_limit_rejected_total")
        assert after >= before + 1                            # 429 计入 rate_limit_rejected_total
        for _ in range(3):                                    # /metrics 豁免：连打多次仍全 200
            assert client.get("/metrics").status_code == 200
        assert client.get("/ping").status_code == 429         # 配额未释放：/ping 依旧 429（豁免只对 /metrics 生效）


class TestLogging:
    """结构化日志：JsonFormatter 的 request_id 注入与 extra 字段透传；text/json 格式切换。"""

    def test_json_formatter_request_id_and_extra(self):
        token = request_id_var.set("rid-abc123")              # 模拟请求中间件注入的上下文
        try:
            rec = logging.LogRecord("serve", logging.INFO, __file__, 1, "服务已就绪", None, None)  # 手工构造日志记录
            rec.faq_count = 30                                # 模拟 extra={...} 业务字段（LogRecord 动态属性）
            payload = json.loads(JsonFormatter().format(rec))  # 序列化成一行 JSON 后反解析校验
            assert payload["msg"] == "服务已就绪"               # 消息体原样保留
            assert payload["request_id"] == "rid-abc123"       # 上下文 request_id 自动注入
            assert payload["faq_count"] == 30                  # extra 字段透传
        finally:
            request_id_var.reset(token)                       # 清理上下文，防串号污染后续用例

    def test_setup_logging_text_mode_then_restore_json(self):
        try:
            setup_logging(fmt="text")                          # 切人读单行文本格式，不应抛异常
            root = logging.getLogger()
            assert len(root.handlers) == 1                    # 替换式初始化：恰一个 handler
            assert not isinstance(root.handlers[0].formatter, JsonFormatter)  # text 模式用普通 Formatter
        finally:
            setup_logging(fmt="json")                          # 测试末尾恢复 JSON 行式（serve 进程级默认格式）
            assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)  # 已还原


class TestLlmCallTimer:
    """LLM 调用埋点：ok/error 计数增量 + 异常原样上抛（与姊妹项目同款口径）。"""

    def test_llm_call_timer_counts_and_reraises(self):
        ok0 = _counter_value(_metrics_text(), "llm_calls_total", 'status="ok"')      # ok 基线
        err0 = _counter_value(_metrics_text(), "llm_calls_total", 'status="error"')  # error 基线
        with llm_call_timer():                                # 正常结束 → status=ok
            pass
        assert _counter_value(_metrics_text(), "llm_calls_total", 'status="ok"') == ok0 + 1  # ok 增量恰 1
        with pytest.raises(ValueError):                       # 异常必须原样上抛（不吞错）
            with llm_call_timer():                            # 抛异常 → status=error
                raise ValueError("boom")
        assert _counter_value(_metrics_text(), "llm_calls_total", 'status="error"') == err0 + 1  # error 增量恰 1


class TestRouteTemplate:
    """路由模板标签：参数路由按模板记账，防高基数标签爆炸。"""

    def test_param_route_recorded_as_template(self):
        client = TestClient(serve.app)
        # action=boom 是未知动作 → 400（不触碰 S["queue"]，无需手工填充全局状态）
        assert client.post("/annotations/xyz", json={"action": "boom"}).status_code == 400
        text = client.get("/metrics").text                     # 抓取指标文本
        assert 'path="/annotations/{rid}"' in text              # 记录的是路由模板而非具体 rid
