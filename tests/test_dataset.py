"""dataset 单测：load_jsonl / PairDataset / TripletDataset（离线负采样）/ CrossEncoderDataset / DataLoader 工厂

全部使用 FakeTok（伪 HuggingFace tokenizer），不依赖真实 BERT 权重。
"""

import json

import torch
from dataset import (
    CrossEncoderDataset,
    PairDataset,
    TripletDataset,
    build_pair_loaders,
    load_jsonl,
)


class FakeTok:
    """伪 HuggingFace tokenizer：支持单句/句对两种调用，返回 [1, L] torch 张量。"""

    def _pad(self, vals, L):
        vals = list(vals[:L]) + [0] * (L - len(vals))
        return torch.tensor([vals])

    def __call__(self, *texts, max_length=8, truncation=True,
                 padding="max_length", return_tensors="pt"):
        L = max_length
        if len(texts) == 1:                     # 单句：token_type 全 0
            ids = [ord(c) % 1000 + 2 for c in texts[0]]
            mask = [1 if i else 0 for i in ids]
            return {"input_ids": self._pad(ids, L),
                    "attention_mask": self._pad(mask, L),
                    "token_type_ids": torch.zeros(1, L, dtype=torch.long)}
        a = [ord(c) % 1000 + 2 for c in texts[0]]    # 句对：拼接 + 句界标记
        b = [ord(c) % 1000 + 2 for c in texts[1]]
        return {"input_ids": self._pad(a + b, L),
                "attention_mask": self._pad([1] * (len(a) + len(b)), L),
                "token_type_ids": self._pad([0] * len(a) + [1] * len(b), L)}


def write_jsonl(path, rows):
    # 行间多写一个空行，顺带覆盖"跳过空行"分支
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n",
                    encoding="utf-8")


class TestLoadJsonl:
    def test_load_skips_blank_lines(self, tmp_path):
        p = tmp_path / "x.jsonl"
        write_jsonl(p, [{"sentence1": "a", "sentence2": "b", "label": 1}])
        assert load_jsonl(p) == [{"sentence1": "a", "sentence2": "b", "label": 1}]


class TestPairDataset:
    def test_getitem_shapes_and_label(self, tmp_path):
        p = tmp_path / "p.jsonl"
        write_jsonl(p, [
            {"sentence1": "血压计怎么用", "sentence2": "血压计如何使用", "label": 1},
            {"sentence1": "血压计怎么用", "sentence2": "今天天气如何", "label": 0},
        ])
        ds = PairDataset(p, FakeTok(), max_length=8)
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids_a"].shape == (8,)
        assert item["input_ids_b"].shape == (8,)
        assert item["label"].dtype == torch.long
        assert item["label"].item() == 1


class TestTripletDataset:
    ROWS = [
        {"sentence1": "A句", "sentence2": "B句", "label": 1},
        {"sentence1": "C句", "sentence2": "D句", "label": 1},
        {"sentence1": "A句", "sentence2": "X句", "label": 0},
        {"sentence1": "B句", "sentence2": "Y句", "label": 0},
    ]

    def test_build_triplets(self, tmp_path):
        p = tmp_path / "t.jsonl"
        write_jsonl(p, self.ROWS)
        ds = TripletDataset(p, FakeTok(), max_length=8)
        assert len(ds) == 2                                   # 三元组数 = 正样本对数
        assert ds.triplets[0] == ("A句", "B句", "X句")         # anchor/positive/标注负例
        anchor2, pos2, neg2 = ds.triplets[1]
        assert (anchor2, pos2) == ("C句", "D句")
        assert neg2 not in ("C句", "D句")                      # 无自身负例 → 全局兜底采样

    def test_getitem_keys(self, tmp_path):
        p = tmp_path / "t.jsonl"
        write_jsonl(p, self.ROWS)
        item = TripletDataset(p, FakeTok(), max_length=8)[0]
        for suffix in ("a", "p", "n"):
            assert item[f"input_ids_{suffix}"].shape == (8,)


class TestCrossEncoderDataset:
    def test_pair_concat_with_segment_ids(self, tmp_path):
        p = tmp_path / "c.jsonl"
        write_jsonl(p, [{"sentence1": "甲乙", "sentence2": "丙丁", "label": 0}])
        item = CrossEncoderDataset(p, FakeTok(), max_length=8)[0]
        assert item["input_ids"].shape == (8,)
        assert item["token_type_ids"].sum().item() == 2       # 第二句 2 个 token 标记为 1
        assert item["label"].item() == 0


class TestBuildPairLoaders:
    def test_loaders_batching(self, tmp_path):
        rows = [{"sentence1": "a", "sentence2": "b", "label": 1}] * 4
        write_jsonl(tmp_path / "train.jsonl", rows)
        write_jsonl(tmp_path / "validation.jsonl", rows[:2])
        write_jsonl(tmp_path / "test.jsonl", rows[:2])
        tr, va, te = build_pair_loaders(tmp_path, FakeTok(), max_length=8, batch_size=2)
        batch = next(iter(tr))
        assert batch["input_ids_a"].shape == (2, 8)           # DataLoader 自动堆叠
        assert batch["label"].shape == (2,)
        assert len(va) == 1 and len(te) == 1
