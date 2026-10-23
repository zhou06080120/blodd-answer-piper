"""llm_synth 单测：normalize / dedupe / _extract_json / 离线模板合成 / LLM 合成器（Fake 客户端回放）"""

import json

import llm_synth
from llm_synth import (
    OfflineTemplateSynth,
    LLMSynthesizer,
    _extract_json,
    build_synthesizer,
    dedupe,
    normalize,
)

DEMO = {
    "id": 1,
    "question": "血压计怎么测量血压？",
    "variants": ["如何用血压计测血压？"],
    "answer": "静坐五分钟后测量。",
}


# ── 文本归一化 / 去重 ────────────────────────────────────────────────────────

class TestNormalize:
    def test_strips_punct_and_lowercases(self):
        assert normalize("Hello, World!？") == "helloworld"
        assert normalize("血压计，怎么用？") == "血压计怎么用"
        assert normalize("  A B\tC ") == "abc"

    def test_non_string_coerced(self):
        assert normalize(123) == "123"


class TestDedupe:
    def test_removes_empty_and_question_only(self):
        assert dedupe(["", "   ", "？", "?"]) == []

    def test_dedupe_by_normalized_key(self):
        assert dedupe(["血压计怎么用", "血压计怎么用？", "血压计，怎么用!"]) == ["血压计怎么用"]

    def test_excludes_existing(self):
        out = dedupe(["新问题", "血压计怎么用"], existing=["血压计怎么用？"])
        assert out == ["新问题"]

    def test_preserves_first_occurrence_order(self):
        assert dedupe(["b", "a", "B"]) == ["b", "a"]


# ── JSON 容错抽取 ───────────────────────────────────────────────────────────

class TestExtractJson:
    def test_direct_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        text = '说明文字\n```json\n{"paraphrases": ["x"]}\n```\n结束'
        assert _extract_json(text) == {"paraphrases": ["x"]}

    def test_embedded_json(self):
        assert _extract_json('前言 {"a": [1, 2]} 后记') == {"a": [1, 2]}

    def test_invalid_returns_empty(self):
        assert _extract_json("not json at all") == {}
        assert _extract_json("") == {}


# ── 离线模板合成 ────────────────────────────────────────────────────────────

class TestOfflineTemplateSynth:
    def test_deterministic_with_seed(self):
        a = OfflineTemplateSynth(seed=7).generate_paraphrases(DEMO, need=6)
        b = OfflineTemplateSynth(seed=7).generate_paraphrases(DEMO, need=6)
        assert a == b and len(a) > 0

    def test_paraphrases_valid(self):
        out = OfflineTemplateSynth(seed=42).generate_paraphrases(DEMO, need=8)
        assert 0 < len(out) <= 8
        existing = {normalize(x) for x in [DEMO["question"]] + DEMO["variants"]}
        for p in out:
            assert normalize(p) not in existing   # 不与库内问法重复
            assert p.endswith("？")               # 统一补问号

    def test_negatives_swap_intent_word(self):
        out = OfflineTemplateSynth(seed=42).generate_negatives(DEMO, need=10)
        assert 0 < len(out) <= 4                  # “测量”有 4 个跨意图替换词
        for n in out:
            assert "测量" not in n                # 动作词已被替换成相邻意图
            assert normalize(n) != normalize(DEMO["question"])


# ── LLM 合成器 ──────────────────────────────────────────────────────────────

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeOpenAI:
    """按脚本回放 LLM 响应的假客户端；脚本元素为 JSON 字符串或 Exception。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                if not outer.script:
                    raise AssertionError("LLM 调用次数超出 FakeOpenAI 脚本预设")
                item = outer.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return _Resp(item)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class TestLLMSynthesizer:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        s = LLMSynthesizer()
        assert s.available is False
        assert s._client is None

    def test_no_client_returns_offline(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        s = LLMSynthesizer()
        r = s.synthesize_faq(DEMO)
        assert r == {"paraphrases": [], "boundary_negatives": [], "source": "offline"}

    def test_llm_success_path(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        s = LLMSynthesizer(api_key="sk-fake")
        assert s.available is True
        payload = json.dumps({
            "paraphrases": ["怎么量血压", "血压咋测"],
            "boundary_negatives": ["血压计怎么校准"],
        })
        s._client = FakeOpenAI([payload])
        r = s.synthesize_faq(DEMO)
        assert r["source"] == "llm"
        assert "怎么量血压" in r["paraphrases"]
        assert r["boundary_negatives"] == ["血压计怎么校准"]
        assert len(s._client.calls) == 1

    def test_retry_then_success(self, monkeypatch):
        monkeypatch.setattr(llm_synth.time, "sleep", lambda s: None)
        s = LLMSynthesizer(api_key="sk-fake")
        payload = json.dumps({"paraphrases": ["问法一"], "boundary_negatives": []})
        s._client = FakeOpenAI([RuntimeError("boom"), payload])
        r = s.synthesize_faq(DEMO)
        assert r["source"] == "llm"
        assert len(s._client.calls) == 2

    def test_empty_paraphrases_triggers_retry(self, monkeypatch):
        monkeypatch.setattr(llm_synth.time, "sleep", lambda s: None)
        s = LLMSynthesizer(api_key="sk-fake")
        bad = json.dumps({"paraphrases": [], "boundary_negatives": []})
        good = json.dumps({"paraphrases": ["有效问法"], "boundary_negatives": []})
        s._client = FakeOpenAI([bad, good])
        r = s.synthesize_faq(DEMO)
        assert r["source"] == "llm"
        assert r["paraphrases"] == ["有效问法"]

    def test_all_fail_falls_offline(self, monkeypatch):
        monkeypatch.setattr(llm_synth.time, "sleep", lambda s: None)
        s = LLMSynthesizer(api_key="sk-fake")
        s._client = FakeOpenAI([RuntimeError("net down")] * 3)
        r = s.synthesize_faq(DEMO)
        assert r["source"] == "offline"
        assert r["paraphrases"] == []


class TestBuildSynthesizer:
    def test_force_offline(self):
        assert isinstance(build_synthesizer(force_offline=True), OfflineTemplateSynth)

    def test_no_key_returns_offline(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert isinstance(build_synthesizer(), OfflineTemplateSynth)

    def test_force_llm_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        s = build_synthesizer(force_llm=True)
        assert isinstance(s, LLMSynthesizer)
        assert s.available is False
