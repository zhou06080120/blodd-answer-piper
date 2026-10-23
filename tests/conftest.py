"""
blodd_answer_piper 测试全局配置

设计要点（离线可复现，本地与 CI 行为一致）：
  1. KMP_DUPLICATE_LIB_OK：torch / scikit-learn / onnxruntime 三方在 Windows 上
     各自带一份 OpenMP 运行库，同进程共存会访问违例——项目源码（train_biencoder.py）
     已有同款约定，测试侧在导入任何重库之前先设好。
  2. 不预置 DEEPSEEK_API_KEY：llm_synth 无 Key 时自动走离线模板，
     保证 LLMSynthesizer 相关测试永不触网。
  3. RATE_LIMIT_RPM=0：测试默认关闭限流——serve 测试连打多个请求不触发 429，
     限流专项测试（test_rate_limit.py）自建独立 app 验证中间件行为。
  4. FakeDenseEncoder：字符袋 + L2 归一化的假编码器——共享字符越多余弦越高，
     让 HybridRetriever / serve 的检索逻辑可以零模型依赖地跑全流程。
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("RATE_LIMIT_RPM", "0")  # 测试默认关闭限流（限流专项测试自建 app）

import numpy as np  # noqa: E402  (须在环境变量设置之后再导入 numpy/torch 生态)
import pytest  # noqa: E402


class FakeDenseEncoder:
    """假 Dense 编码器：字符袋 + L2 归一化，共享字符越多余弦越高。

    接口对齐 onnx_runtime.OnnxEncoder / TorchEncoder 的 encode(texts)->[N,H]。
    """

    backend = "fake"   # serve /health 会读取 backend 字段
    kind = "char-bow"

    def __init__(self, dim=512):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for ch in str(text):
            v[ord(ch) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-9)

    def encode(self, texts, batch_size=None):
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture()
def fake_dense():
    return FakeDenseEncoder()
