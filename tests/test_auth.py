"""auth 双级 Key 鉴权测试（P4 平台安全）

策略（与 test_rate_limit.py 同款）：
  - 全部用例自建独立迷你 FastAPI 应用（不碰 serve.app，不拉起模型/索引）；
  - monkeypatch.setenv 配 BP_API_KEYS / BP_ADMIN_KEYS 后调用 install_auth，
    复现 serve.py 模块级安装入口的同款行为；
  - 管理路由守卫与 serve.py 同款：带 Request 注解的薄包装调用 auth.require_admin
    （其 request 形参无类型注解，直接作 Depends 会被 FastAPI 解析成 query 参数）；
  - auth._auth_enabled 是模块级开关，配 Key 后置 True，跨用例会互相污染：
    autouse fixture 在每个用例前后都还原 False，保证 test_serve 等模块零影响。
"""

import hashlib

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

import auth


@pytest.fixture(autouse=True)
def _reset_auth_flag():
    """前后双还原模块级鉴权开关（install_auth 配 Key 后置 True，防跨用例/跨模块泄漏）。"""
    auth._auth_enabled = False                       # 前置清理：防上个用例遗留 True
    yield
    auth._auth_enabled = False                       # 后置还原：其他测试模块（如 test_serve）零影响


def _build_app():
    """构造与 serve.py 同口径的迷你应用：业务路由 + 指纹回显 + 管理路由（守卫适配层）。"""

    async def require_admin_dep(request: Request) -> None:   # 与 serve.py 同款注入适配层
        auth.require_admin(request)

    app = FastAPI()

    @app.get("/")                                    # 豁免路径：前端首页（与 auth 豁免清单对齐）
    def home():
        return {"app": "mini"}

    @app.get("/health")                              # 豁免路径：探针
    def health():
        return {"status": "ok"}

    @app.get("/metrics")                             # 豁免路径：指标抓取
    def metrics():
        return {"metrics": []}

    @app.get("/query")                               # 业务端点：不加路由依赖，中间件统一校验普通级
    def query():
        return {"ok": True}

    @app.get("/whoami")                              # 回显鉴权中间件写入的 Key 指纹（验 actor_id 注入）
    def whoami(request: Request):
        return {"actor_id": getattr(request.state, "actor_id", "-")}

    @app.post("/admin/promote", dependencies=[Depends(require_admin_dep)])  # 管理端点：403 守卫
    def promote():
        return {"ok": True}

    return app


def _install(monkeypatch, user="", admin=""):
    """注入 Key 配置 → 自建 app → 安装鉴权，返回 (app, client)。"""
    monkeypatch.setenv("BP_API_KEYS", user)          # 空串 = 未配置该级 Key
    monkeypatch.setenv("BP_ADMIN_KEYS", admin)
    app = _build_app()
    auth.install_auth(app)                           # 与 serve.py 模块级安装同款入口
    return app, TestClient(app)                     # 不进 with：与 test_serve.py 同款用法


class TestKeyAuth:
    def test_missing_key_401(self, monkeypatch):
        _, client = _install(monkeypatch, user="user-key", admin="admin-key")
        assert client.get("/query").status_code == 401   # 缺 X-API-Key 头被拦截

    def test_wrong_key_401(self, monkeypatch):
        _, client = _install(monkeypatch, user="user-key", admin="admin-key")
        r = client.get("/query", headers={"X-API-Key": "bad-key"})
        assert r.status_code == 401                      # 错 Key 与缺 Key 同码（不区分，防探测）

    def test_user_key_business_200_with_actor_fingerprint(self, monkeypatch):
        _, client = _install(monkeypatch, user="user-key", admin="admin-key")
        assert client.get("/query", headers={"X-API-Key": "user-key"}).status_code == 200
        expect = hashlib.sha256(b"user-key").hexdigest()[:8]   # 指纹 = SHA-256 前 8 位（审计脱敏用）
        r = client.get("/whoami", headers={"X-API-Key": "user-key"})
        assert r.json()["actor_id"] == expect           # request.state.actor_id 已由中间件注入

    def test_user_key_admin_403(self, monkeypatch):
        _, client = _install(monkeypatch, user="user-key", admin="admin-key")
        r = client.post("/admin/promote", headers={"X-API-Key": "user-key"})
        assert r.status_code == 403                    # 普通角色访问管理端点被 require_admin 拦截

    def test_admin_key_admin_200_and_business_200(self, monkeypatch):
        _, client = _install(monkeypatch, user="user-key", admin="admin-key")
        h = {"X-API-Key": "admin-key"}
        assert client.post("/admin/promote", headers=h).status_code == 200   # 管理角色放行管理端点
        assert client.get("/query", headers=h).status_code == 200            # 管理 Key 隐含普通权限

    def test_exempt_paths_bypass_auth(self, monkeypatch):
        _, client = _install(monkeypatch, user="user-key", admin="admin-key")
        for path in ("/", "/health", "/metrics"):       # 豁免清单：探针与前端不带 Key 也放行
            assert client.get(path).status_code == 200

    def test_no_keys_configured_passthrough(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "")           # 一级 Key 都不配 = 本地开发模式
        monkeypatch.setenv("BP_ADMIN_KEYS", "")
        app = _build_app()
        assert auth.install_auth(app) is None           # 未配置任何 Key：不注册中间件（返回 None，仅 warning）
        client = TestClient(app)
        assert client.get("/query").status_code == 200            # 业务端点放行
        assert client.post("/admin/promote").status_code == 200   # require_admin 未启用也放行（向后兼容）
