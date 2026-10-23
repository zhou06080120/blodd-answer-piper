"""onnx_runtime 单测：_l2_norm / _tokenize_batch / TorchEncoder（Fake 模型，零 BERT 依赖）"""

import numpy as np
import torch
from onnx_runtime import TorchEncoder, _l2_norm, _tokenize_batch


class FakeTokNP:
    """伪 HuggingFace tokenizer：单句调用协议，返回 numpy 数组（return_tensors='np'）。"""

    def __call__(self, text, max_length=8, truncation=True,
                 padding="max_length", return_tensors="np"):
        ids = [ord(c) % 1000 + 2 for c in str(text)][:max_length]
        ids += [0] * (max_length - len(ids))
        mask = [1 if i else 0 for i in ids]
        return {"input_ids": np.array([ids]),
                "attention_mask": np.array([mask]),
                "token_type_ids": np.zeros((1, max_length))}


class TestL2Norm:
    def test_normalizes(self):
        out = _l2_norm(np.array([[3.0, 4.0]]))
        assert np.allclose(out, [[0.6, 0.8]])

    def test_zero_vector_safe(self):
        out = _l2_norm(np.zeros((1, 3)))
        assert np.isfinite(out).all()          # 防零除保护，不产生 nan


class TestTokenizeBatch:
    def test_stack_shape_and_dtype(self):
        ids, masks, ttids = _tokenize_batch(FakeTokNP(), ["甲乙", "丙"], 4)
        assert ids.shape == (2, 4) and ids.dtype == np.int64     # ORT 要求 int64
        assert masks.shape == (2, 4) and ttids.shape == (2, 4)
        assert int(masks[1].sum()) == 1        # “丙”仅 1 个真实 token，其余为 pad


class FakeBiencoder:
    """伪 BiEncoder：encode(三路输入) → [B, 4] 张量（one-hot 可区分 batch 内样本）。"""

    def encode(self, ids, masks, ttids):
        return torch.nn.functional.one_hot(ids[:, 0] % 4, num_classes=4).float()


class TestTorchEncoder:
    def test_encode_batches_and_l2_normalizes(self):
        enc = TorchEncoder(FakeBiencoder(), FakeTokNP(), torch.device("cpu"),
                           max_length=4, batch_size=2)
        out = enc.encode(["甲乙", "丙", "丁"])     # batch_size=2 → 2+1 两个批次
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)  # 输出已归一化
