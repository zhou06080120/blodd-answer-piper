"""
CrossEncoder 训练数据增强（prepare_ce_data.py）

动机（实测诊断结论）：
  - synthesize_data.py 切分的正例均为 (FAQ标准问, 变体)，导致 train 中 sentence1 位置
    永远是标准问句式（「血压机」一词只出现在 sentence2，sentence1 出现 0 次）
  - CrossEncoder 对 sentence1 位置过拟合标准问句式，真实用户口语（如「血压机怎么使用？」）
    放在 sentence1 时灾难性误判；领域外句子（「今天天气怎么样？」）无训练先例也会乱判高分
  - 本脚本对 train.jsonl 做三类增强，产出 ce_train.jsonl（不动原 train.jsonl，
    BiEncoder 训练数据的可复现性不受影响）：
    1. 位置翻转：每对 (a, b, label) 补充 (b, a, label)  → 打破 sentence1 只出标准问的偏置
    2. 同义口语替换：按保守词典随机替换生成变体对      → 覆盖「血压机/咋用/咋测」等口语组合
    3. 领域外 chitchat 负例：正例问句配无关句（双向位置）→ 教会 CE 对无关内容判低分

使用方式：
  python prepare_ce_data.py
产物：data/blood_pressure_expanded/ce_train.jsonl
随后重训：python src/train_crossencoder.py --data_dir data/blood_pressure_expanded \
            --train_file ce_train.jsonl --bert_path F:\\aaa123\\pretrain_models\\bert-base-chinese --epochs 3
"""
# ↑ 模块级 docstring：修复 CrossEncoder 位置偏置与领域外误判的数据增强脚本

import json                                       # 读写 JSONL
import random                                      # 可复现随机替换
from pathlib import Path                          # 路径处理

ROOT = Path(__file__).parent                       # 项目根目录
DATA_DIR = ROOT / "data" / "blood_pressure_expanded"  # 增强数据所在目录
SRC_TRAIN = DATA_DIR / "train.jsonl"               # 输入：原始训练对
OUT_TRAIN = DATA_DIR / "ce_train.jsonl"            # 输出：CE 专用增强训练对

# ── 保守同义替换词典（只做语义严格等价的替换）──────────────────────────────
SYNONYMS = [                                      # 只收录语义完全等价的替换对（保守策略）
    ("血压计", "血压机"),                           # 常见错别字/口语叫法
    ("怎么使用", "怎么用"),                          # 动词口语化
    ("如何", "怎么"),                              # 疑问词替换
    ("测量", "测"),                                # 动词简化
    ("校准", "校正"),                              # 同义术语
]

# ── 领域外 chitchat / 跨域负例句（双向位置都放）─────────────────────────────
CHITCHAT = [                                       # 与血压计完全无关的日常问句
    "今天天气怎么样？", "明天会下雨吗？", "你好", "你是谁？", "谢谢你了",
    "帮我查一下快递到哪了", "附近有什么好吃的？", "手机怎么开机？",
    "空调遥控器坏了怎么办？", "给我讲个笑话", "股票今天涨了吗？",
    "推荐一部电影", "英语四级怎么报名？", "红烧肉怎么做？",
    "帮我订个外卖", "这个软件怎么卸载？", "WiFi 密码是多少？",
    "洗衣机不转了怎么回事？", "感冒药怎么吃？", "小孩发烧怎么办？",
    "高铁票怎么改签？", "今天限行尾号是多少？",
]


def apply_replacements(text, rng, max_subs=1):
    """随机做至多 max_subs 处同义替换；无可替换时返回 None"""
    out, done = text, 0                            # out=替换结果，done=已替换次数
    for src, dst in rng.sample(SYNONYMS, len(SYNONYMS)):  # 随机打乱词典顺序（多样性）
        if done >= max_subs:                      # 达到替换上限
            break                                # 提前结束
        if src in out:                            # 文本含该词
            out = out.replace(src, dst, 1)       # 只替换第一处
            done += 1                             # 计数
    return out if done else None                  # 一处都没换到就返回 None（外层跳过）


def main():
    """对 train.jsonl 做位置翻转 / 同义替换 / 领域外负例三类增强，产出 ce_train.jsonl。"""
    rng = random.Random(20261013)                  # 固定种子（日期即种子，可复现）
    rows = [json.loads(line) for line in SRC_TRAIN.read_text(encoding="utf-8").splitlines()
            if line.strip()]                          # 读原始训练对（跳过空行）
    print(f"原始 train 对数: {len(rows)}")          # 打印输入规模

    out, seen = [], set()                          # 输出列表 + 去重集合

    def add(a, b, label):                          # 统一入口：去重后追加样本
        key = (a, b, label)                        # 三元组作键
        if key in seen:                            # 重复样本
            return                                # 丢弃
        seen.add(key)                              # 登记防重
        out.append({"sentence1": a, "sentence2": b, "label": label})  # 按 CE 数据格式落列

    for r in rows:                                 # 遍历原始样本
        a, b, label = r["sentence1"], r["sentence2"], r["label"]  # 解包三要素
        add(a, b, label)            # 原对        # 增强一（0）：原样保留
        add(b, a, label)            # 翻转：sentence1 也见口语变体  # 增强一：位置翻转打破位置偏置

        va = apply_replacements(a, rng)            # 对句 a 做同义替换
        vb = apply_replacements(b, rng)            # 对句 b 做同义替换
        if va:                                     # a 换出了新句
            add(va, b, label)                      # (a变体, b)
            add(b, va, label)                      # (b, a变体)（同样双向）
        if vb:                                     # b 换出了新句
            add(a, vb, label)                      # (a, b变体)
            add(vb, a, label)                      # (b变体, a)

        if label == 1:                             # 只给正样本配 chitchat 负例
            # 领域外负例：正例问句 vs chitchat（双向位置）
            for c in rng.sample(CHITCHAT, 2):      # 随机挑 2 条闲聊句
                add(a, c, 0)                       # 正问在前、闲聊在后 → 标 0
                add(c, a, 0)                       # 闲聊在前、正问在后 → 标 0（双向覆盖）

    pos = sum(1 for r in out if r["label"] == 1)   # 统计正例数
    neg = len(out) - pos                           # 负例数 = 总数 - 正例
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:  # 打开输出文件
        for r in out:                              # 逐行写
            f.write(json.dumps(r, ensure_ascii=False) + "\n")  # JSONL 一行一对
    print(f"增强后 CE train: {len(out)} 对（正 {pos} / 负 {neg}，正负比 1:{neg / max(pos, 1):.2f}）")  # 打印产物统计（正负比 max 防除零）
    print(f"产物 → {OUT_TRAIN}")                    # 提示产物路径


if __name__ == "__main__":
    main()                                         # 直跑才执行增强
