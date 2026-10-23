"""audit 管理操作审计留痕测试（P4 平台合规）

策略：
  - monkeypatch.setenv BP_AUDIT_PATH 指向 tmp_path 临时文件，绝不写仓库目录；
  - 直接调 record_audit 验证 JSONL 追加写语义：行数、逐行可解析、字段齐全、
    中文 ensure_ascii=False 原样落盘、默认参数口径。
"""

import json

import pytest

import audit


@pytest.fixture()
def audit_file(tmp_path, monkeypatch):
    """审计日志定向到临时文件，返回该 Path。"""
    p = tmp_path / "audit.jsonl"
    monkeypatch.setenv("BP_AUDIT_PATH", str(p))
    return p


class TestRecordAudit:
    def test_two_records_append_two_lines(self, audit_file):
        """连续两次追加 = 恰好两行，每行独立 JSON 可解析，六个字段齐全。"""
        audit.record_audit("annotate", target="r1", detail={"action": "attach"}, actor="ab12cd34")
        audit.record_audit("promote_gray", target="4", detail={"faq_ids": [4]}, actor="ff0011aa")
        lines = audit_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2                          # 两条记录恰好两行（只增追加写）
        rows = [json.loads(x) for x in lines]          # 每行独立可解析
        for row in rows:
            assert set(row) == {"ts", "actor", "action", "target", "detail", "request_id"}  # 字段齐全
            assert row["ts"]                            # 时间戳非空
        assert rows[0]["action"] == "annotate" and rows[0]["target"] == "r1"
        assert rows[1]["action"] == "promote_gray" and rows[1]["actor"] == "ff0011aa"

    def test_chinese_detail_not_escaped(self, audit_file):
        """中文 detail 原样 UTF-8 落盘（ensure_ascii=False），不转 \\uXXXX。"""
        audit.record_audit("annotate", target="r9", detail={"query": "血压计怎么校准"}, actor="ab12cd34")
        raw = audit_file.read_text(encoding="utf-8")   # 读原始字节流验转义行为
        assert "血压计怎么校准" in raw                   # 中文原样可见（未被 \uXXXX 转义）
        row = json.loads(raw.splitlines()[0])           # 单行仍可反序列化回对象
        assert row["detail"]["query"] == "血压计怎么校准"

    def test_default_arguments(self, audit_file):
        """缺省口径：actor="-"、target=""、detail=None、request_id="-"（无请求上下文）。"""
        entry = audit.record_audit("admin_refresh")    # 全部默认参数
        assert entry["actor"] == "-"                    # 未鉴权操作者占位
        assert entry["target"] == ""                    # 无操作对象
        assert entry["detail"] is None                  # 无附加信息
        assert entry["request_id"] == "-"              # 请求外调用无请求编号
        assert audit_file.exists()                      # 单条也已落盘（目录缺失自愈）
