"""rate_limit 限流器测试

策略：
  - 纯类用例全部用 now 参数注入时间，绝不真实 sleep（测试毫秒级跑完）；
  - 中间件集成用例自建独立 FastAPI 应用（monkeypatch 设 RATE_LIMIT_RPM=3），
    不依赖 serve.app——conftest 已把测试环境默认限流关闭（=0），互不干扰；
  - TestClient 用法与 test_serve.py 同款：不进 with 上下文，直接发请求。
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rate_limit import RateLimiter, install_rate_limit


class TestRateLimiter:
    """纯算法层：滑动窗口计数 / 窗口滑动恢复 / 关闭态 / retry_after / 过期自愈。"""

    def test_rpm3_allows_three_then_rejects(self):
        limiter = RateLimiter(rpm=3)                  # 每分钟 3 次
        assert limiter.allow("ip", now=0) is True     # 第 1 次放行
        assert limiter.allow("ip", now=0) is True     # 第 2 次放行
        assert limiter.allow("ip", now=0) is True     # 第 3 次放行
        assert limiter.allow("ip", now=0) is False     # 第 4 次拒绝

    def test_sliding_window_recovery(self):
        limiter = RateLimiter(rpm=2)                  # 每分钟 2 次
        assert limiter.allow("ip", now=0) is True     # 窗口内第 1 次
        assert limiter.allow("ip", now=0) is True     # 窗口内第 2 次
        assert limiter.allow("ip", now=0) is False    # 窗口已满 → 拒绝
        assert limiter.allow("ip", now=61) is True    # 旧时间戳滑出 60s 窗口 → 恢复放行

    def test_rpm0_disables_limit(self):
        limiter = RateLimiter(rpm=0)                  # 0 = 关闭限流
        assert all(limiter.allow("ip") for _ in range(100))  # 连续 100 次全放行

    def test_retry_after(self):
        limiter = RateLimiter(rpm=1)                  # 每分钟 1 次
        assert limiter.allow("ip", now=100) is True   # t=100 放行一次
        assert limiter.retry_after("ip", now=100) == 60  # 下一个可放行时刻 = 100+60

    def test_evict_stale(self):
        limiter = RateLimiter(rpm=10)                 # 阈值不重要，只验过期清理
        limiter._hits["stale"].extend([1.0, 2.0])     # 两条时间戳全部早于 cutoff
        limiter._hits["fresh"].append(100.0)          # 最近一条仍在窗口内
        limiter._evict_stale(now=120.0)               # cutoff = 120 - 60 = 60
        assert "stale" not in limiter._hits           # 全过期队列被清理
        assert "fresh" in limiter._hits               # 活跃队列保留


class TestMiddleware:
    """中间件集成：超阈值 429 + Retry-After 响应头；/health 探针豁免。"""

    def test_429_with_retry_after_and_health_exempt(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RPM", "3")     # 每分钟 3 次
        app = FastAPI()                               # 自建独立应用，不碰 serve.app

        @app.get("/ping")
        def ping():
            return {"ok": True}

        @app.get("/health")
        def health():
            return {"status": "ok"}

        install_rate_limit(app)                       # 注册限流中间件
        client = TestClient(app)                      # 不进 with：与 test_serve.py 同款用法

        for _ in range(3):                            # 阈值内连打 3 次
            assert client.get("/ping").status_code == 200
        r = client.get("/ping")                        # 第 4 次超阈值
        assert r.status_code == 429                    # Too Many Requests
        assert r.headers["Retry-After"].isdigit()     # 必须携带 Retry-After 响应头
        assert "请求过于频繁" in r.json()["detail"]    # 中文拒绝话术
        for _ in range(5):                            # /health 豁免：连打多次仍全 200
            assert client.get("/health").status_code == 200
