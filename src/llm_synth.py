"""
LLM 自动化数据合成：FAQ 同义改写 + 边界负例生成（含离线模板回退）

设计目标（需求 1：扩充训练数据、提升泛化鲁棒性）：
  1. LLMSynthesizer —— 调 DeepSeek（OpenAI 兼容协议）为每条 FAQ 批量生成：
       - paraphrases        : 同义改写（口语/方言/省略/礼貌语/错别字容忍等风格）
       - boundary_negatives : 边界负例（词面高度相似但意图不同的混淆问题）
     所有请求带 60s 超时 + 重试，解析失败逐级降级（JSON 解析 → 正则抽取 → 离线回退）；
     无 API Key 或全部失败时自动落入离线模板合成，保证流水线离线可冒烟、可复现。
  2. OfflineTemplateSynth —— 无网络兜底：
       前缀/语气词/疑问词/领域同义词 多策略组合改写 + 跨意图关键词替换的边界负例，
       候选语义正确性不在此层保证，由外层（synthesize_data.py）用模型相似度过滤兜底。

使用方式：
  from llm_synth import build_synthesizer
  synth = build_synthesizer()          # 有 Key 用 LLM，否则离线模板
  out   = synth.synthesize_faq(faq, n_paraphrase=30, n_negative=10)

依赖：
  openai（仅联网模式需要）；离线模式仅标准库
"""
# ↑ 模块级 docstring：数据合成双引擎——联网 LLM 改写 + 断网模板兜底

import json                                       # 解析 LLM 返回的 JSON
import os                                         # 读环境变量里的 API Key
import random                                     # 离线模板的随机选择
import re                                         # 文本规范化与 JSON 容错抽取
import time                                       # 重试退避的 sleep

DEFAULT_BASE_URL = "https://api.deepseek.com"      # DeepSeek 的 OpenAI 兼容端点
DEFAULT_MODEL = "deepseek-chat"                    # 默认对话模型名
TIMEOUT_S = 60                                    # 单次请求超时 60 秒
RETRIES = 2                                       # 失败后最多重试 2 次


# ── 文本规范化 / 去重 ───────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[\s，。！？、,.!?~～·「」『』（）()\"'“”‘’：:；;]+")  # 匹配所有空白与中英文标点


def normalize(text):
    """仅保留字母数字与中文，用于去重比较（不影响原始文本存储）"""
    return _PUNCT_RE.sub("", str(text)).lower()     # 删掉全部标点空白后转小写 → "吗??" 与 "吗" 视为相同


def dedupe(texts, existing=()):
    """按 normalize 去重，且排除与已有集合（标准问/已入库变体）重复的句子"""
    seen = {normalize(t) for t in existing if t}    # 把已有句集先归一化进 seen（不允许生成与库内重复的句子）
    out = []                                        # 存放去重后的输出
    for t in texts:                                 # 逐条过滤候选
        t = str(t).strip()                         # 转字符串并去首尾空白
        if not t or not t.rstrip("？?"):           # 两种无效情况：空串 / 去掉问号后没内容
            continue                               # 直接丢弃
        key = normalize(t)                          # 算归一化键
        if key in seen:                             # 与已有或已收集的重复
            continue                               # 丢弃
        seen.add(key)                               # 登记防后续重复
        out.append(t)                               # 保留该句
    return out                                      # 返回去重结果（保持原顺序）


# ── 离线模板合成（无 Key / 断网兜底）────────────────────────────────────────

_OFFLINE_PREFIXES = ["请问", "麻烦问下", "我想问一下", "你好，想咨询下"]  # 口语前缀池（改写时加在句首）

_OFFLINE_SUFFIXES = ["呀", "啊", "呢", ""]          # 语气助词后缀池（含空串表示不变）

# 领域同义词表：长的在前，避免子串先替换
_SYNONYMS = [                                      # (原词, 替换词列表)；长词在前防止"电子血压计"被"血压计"抢先替换
    ("电子血压计", ["电子血压仪", "电子测压计", "电子量压器"]),  # 器械全名同义
    ("水银血压计", ["水银式血压计", "汞柱血压计"]),             # 器械全名同义
    ("血压计", ["血压仪", "测压计", "量血压的仪器"]),           # 通用简称同义
    ("测量", ["测", "量", "量一下", "测一下"]),                # 动词口语化
    ("怎么", ["如何", "怎样", "咋", "怎么个"]),                # 疑问词变化
    ("什么时候", ["啥时候", "什么时候"]),                      # 时间疑问词
    ("多久", ["多长时间", "多少时间"]),                        # 时长疑问词
    ("为什么", ["为啥", "是什么原因"]),                        # 原因疑问词
    ("哪里", ["哪儿", "什么地方"]),                          # 地点疑问词
    ("可以吗", ["行不行", "可不可以", "行吗"]),                # 征求意见问法
    ("应该", ["要", "该"]),                                # 情态动词替换
    ("需要", ["要", "得"]),                                # 情态动词替换
    ("使用", ["用", "操作"]),                              # 动词替换
    ("正确", ["准确", "规范"]),                            # 形容词替换
    ("清洁", ["清洗", "清洁"]),                            # 动词替换（注意有自我重复项，替换后同串会被 dedupe 掉）
    ("保养", ["维护", "养护"]),                            # 动词替换
    ("不一样", ["不相同", "不同"]),                          # 否定短语替换
]

# 跨意图关键词替换：生成"词面相似、意图不同"的边界负例
_INTENT_SWAPS = {                                  # 核心动作词 → 相邻但不同意图的动作词列表
    "测量": ["校准", "清洗", "选购", "维修"],        # 把"怎么测量"改成"怎么校准/清洗/选购/维修"
    "校准": ["测量", "清洗", "选购"],                # 反向亦然，双向制造边界对
    "绑": ["清洗", "更换", "选购"],                 # 袖带相关动作
    "选购": ["校准", "清洁", "维修"],                # 购买意图换成售后意图
    "清洗": ["校准", "更换"],                       # 清洗意图的边界对
    "电池": ["袖带", "校准", "清洗"],                # 部件话题替换
}


# ═════════════════════════════════════════════════════════════════════════
# 功能说明：OfflineTemplateSynth 离线模板合成器
# 一句话概括：不联网、纯规则，用前缀/后缀/去"吗"/同义词替换组合出同义改写，用跨意图替换造边界负例。
# 在流水线中的位置：LLM 不可用时的兜底引擎，保证数据合成流水线在任何环境都能跑通。
# ═════════════════════════════════════════════════════════════════════════
class OfflineTemplateSynth:
    """离线模板改写器：确定性随机（seed 可复现），无任何外部依赖"""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)              # 独立 Random 实例（不污染全局种子），同 seed 产同样结果

    def _rewrite_one(self, sent):
        """对单个句子生成一批改写候选"""
        out = []                                    # 候选列表
        s = str(sent).strip().rstrip("？？")          # 预处理：去首尾空白、去句尾问号（后面统一补回）
        if not s:                                   # 空句直接返回
            return out                              # 返回空列表
        # 1) 前缀变体
        for p in _OFFLINE_PREFIXES[:2]:             # 只取前两个前缀，控制候选数量
            out.append(p + "，" + s + "？")          # 前缀 + 逗号 + 原句 + 问号
        # 2) 语气后缀
        for suf in _OFFLINE_SUFFIXES:                # 遍历语气词（含空串=不加后缀）
            out.append(s + suf + "？")               # 原句 + 语气词 + 问号
        # 3) 去除"吗"的省略口语
        if s.endswith("吗") or s.endswith("么"):      # 句尾是"吗/么"的疑问句
            out.append(s[:-1] + "？")                # 去掉最后一个字 + 问号 → "能测吗" → "能测？"
        # 4) 领域同义词单点替换（每条最多 1 处，避免语义漂移叠加）
        for word, alts in _SYNONYMS:                 # 按表序（长词优先）找第一个命中的原词
            if word in s:                           # 句中含该词
                alt = self.rng.choice(alts)          # 随机挑一个替换词（种子固定 → 可复现）
                out.append(s.replace(word, alt, 1) + "？")  # 只替换第一处，防多处叠加语义漂移
                break                               # 每句只做一次同义词替换
        return [t for t in out if t and t != s + "？"]  # 过滤空串与"原样句"（改写等于没改就没意义）

    def generate_paraphrases(self, faq, need):
        """生成 need 条改写候选（question 与 variants 轮流产生）"""
        pool = []                                   # 改写池
        seeds = [faq["question"]] + list(faq.get("variants", []))  # 种子句 = 标准问 + 已有变体
        idx = 0                                     # 轮询下标
        guard = 0                                   # 防死循环的保险计数
        while len(pool) < need and guard < need * 4 + 60:  # 攒够 need 条或超过保险上限就停
            cands = self._rewrite_one(seeds[idx % len(seeds)])  # 轮流对每个种子做改写（idx 取模循环）
            pool.extend(cands)                      # 候选入池
            idx += 1                                # 下标推进
            guard += 1                              # 保险计数推进
        pool = dedupe(pool, existing=[faq["question"]] + list(faq.get("variants", [])))  # 归一化去重并排除与库内重复
        self.rng.shuffle(pool)                      # 洗牌打散（种子固定仍可复现）
        return pool[:need]                          # 截取前 need 条

    def generate_negatives(self, faq, need):
        """跨意图关键词替换：边界负例（意图不同、词面相似）"""
        q = faq["question"]                          # 只在标准问上做意图替换
        out = []                                    # 负例候选
        for word, swaps in _INTENT_SWAPS.items():    # 遍历每个可替换的动作词
            if word in q:                           # 标准问中含该词
                for alt in swaps:                   # 换成每个相邻意图词
                    out.append(q.replace(word, alt, 1) + "？")  # 生成一条边界负例
        out = dedupe(out)                           # 去重
        self.rng.shuffle(out)                       # 洗牌
        return out[:need]                           # 取前 need 条


# ── LLM 合成（DeepSeek / OpenAI 兼容协议）──────────────────────────────────

# 系统提示词：设定 LLM 的角色与硬约束（内容必须是真实用户会问的问题，不得编造医学结论）
_SYS_PROMPT = (
    "你是一名医疗器械智能客服的数据合成专家，负责为血压计 FAQ 扩充高质量训练数据。"
    "你输出的每一句话都必须是真实用户可能输入的问题，忠于给出的事实与边界，不得编造医学结论。"
)

# 用户提示词模板：要求 LLM 严格按 JSON 返回两类数据——同义改写（多风格）与边界负例（近似但不同意图）
# 模板占位符 {question}/{variants}/{answer}/{n_para}/{n_neg} 由 synthesize_faq 用 format 填充
_USER_PROMPT_TMPL = """请为下面这条血压计 FAQ 生成两类训练数据，严格只输出一个 JSON 对象（不要任何解释文字）：
{{"paraphrases": [...], "boundary_negatives": [...]}}

【标准问题】{question}
【已知同义问法】{variants}
【标准答案】{answer}

1. paraphrases：生成 {n_para} 条与标准问题"语义完全等价、意图一致"的同义改写，模拟真实用户口语提问，风格需覆盖：
   省略主语/语气助词（如"血压怎么量"）、方言口语（"咋""啥""么"）、礼貌用语（"请问""麻烦问一下"）、
   倒装与句式变换、同义词替换（"测量→量/测"）、常见错别字（"血压计→血压机"）、老年人口吻。
   每条不超过 30 字，不得与已有问法重复，不得引入原答案没有的限定条件（如型号、品牌、人群）。

2. boundary_negatives：生成 {n_neg} 条"边界负例"——表面上与标准问题词面高度相似（共享大量字词），
   但用户真实意图并不相同、且无法用本条答案正确回答的问题（例如把"测量方法"问成"校准""清洗""电池""选购"等相邻话题，
   或提问对象/条件被悄悄替换）。这些句子将作为困难负样本训练模型，帮助模型学会区分近似但不同意图。"""


# ═════════════════════════════════════════════════════════════════════════
# 功能说明：LLMSynthesizer 联网合成器
# 一句话概括：把 FAQ 塞进提示词调 DeepSeek，JSON 解析失败逐级降级，全部失败由外层切离线模板。
# 在流水线中的位置：有 API Key 时的首选合成引擎，产出语义多样性远超模板。
# ═════════════════════════════════════════════════════════════════════════
class LLMSynthesizer:
    """DeepSeek 批量改写。构造不抛异常：无 Key 时 available=False，由外层切换离线模式。"""

    def __init__(self, api_key=None, base_url=DEFAULT_BASE_URL,
                 model=DEFAULT_MODEL, timeout=TIMEOUT_S, retries=RETRIES):
        self.api_key = (api_key if api_key is not None
                        else os.getenv("DEEPSEEK_API_KEY", ""))  # 优先显式传参，否则读环境变量（找不到给空串）
        self.base_url = base_url                    # API 端点
        self.model = model                          # 模型名
        self.timeout = timeout                      # 超时秒数
        self.retries = retries                      # 重试次数
        self.available = False                      # 默认不可用，成功建客户端才置 True
        self._client = None                         # OpenAI 客户端占位
        if not self.api_key:                        # 没有 Key
            return                                  # 直接返回（available=False，外层切离线）
        try:
            from openai import OpenAI               # 延迟导入：无 Key 的机器不需要装 openai
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout, max_retries=0)  # max_retries=0：重试逻辑由本类自己控制（带退避）
            self.available = True                   # 客户端就绪
        except Exception as e:  # 占位 Key/网络异常：延迟到调用时再报错
            print(f"[llm_synth] OpenAI 客户端初始化失败（{e}），回退离线模板。")  # 打印原因，available 保持 False

    def synthesize_faq(self, faq, n_paraphrase=30, n_negative=10):
        """返回 {"paraphrases": [...], "boundary_negatives": [...], "source": "llm"}"""
        variants = list(faq.get("variants", []))    # 已有变体（提示词里告知 LLM 避免重复）
        prompt = _USER_PROMPT_TMPL.format(          # 用 FAQ 内容填充提示词模板
            question=faq["question"],               # 标准问题
            variants=json.dumps(variants, ensure_ascii=False),  # 已知问法列表转 JSON 字符串
            answer=faq["answer"],                  # 标准答案
            n_para=n_paraphrase, n_neg=n_negative,  # 数量要求
        )
        last_err = None                             # 记录最后一次失败原因
        for attempt in range(self.retries + 1):     # 首次 + 最多 retries 次重试
            if not self._client:                    # 客户端没建成（无 Key 等）
                last_err = RuntimeError("no api key")  # 记录原因
                break                              # 无需重试，直接出循环走离线
            try:
                resp = self._client.chat.completions.create(  # 发起对话补全请求
                    model=self.model,               # 模型名
                    messages=[{"role": "system", "content": _SYS_PROMPT},  # 系统角色设定
                              {"role": "user", "content": prompt}],         # 用户任务提示词
                    temperature=0.9,                # 高温度 → 改写多样性更好
                    max_tokens=4000,                # 输出上限（30+10 条句子足够）
                    response_format={"type": "json_object"},  # 强制 JSON 模式（DeepSeek 支持，降低解析失败率）
                )
                data = _extract_json(resp.choices[0].message.content)  # 容错解析回复文本为 dict
                paras = dedupe(data.get("paraphrases", []), existing=[faq["question"]] + variants)  # 改写去重（排除库内已有）
                negs = dedupe(data.get("boundary_negatives", []))  # 负例去重
                if paras:                           # 拿到至少一条有效改写即算成功
                    return {"paraphrases": paras[:n_paraphrase],   # 截断到要求数量
                            "boundary_negatives": negs[:n_negative],  # 同上
                            "source": "llm"}        # 标记数据来源是 LLM
                last_err = RuntimeError("empty paraphrases")  # 改写全被过滤成空 → 视作失败可重试
            except Exception as e:                  # 请求/解析任意环节抛异常
                last_err = e                       # 记录异常
            time.sleep(1.5 * (attempt + 1))  # 退避后重试  # 线性递增退避：1.5s → 3s → 4.5s，避免打爆 API
        print(f"  [llm_synth] FAQ#{faq.get('id')} 生成失败（{last_err}），回退离线模板。")  # 全部尝试失败，说明将走离线
        return {"paraphrases": [], "boundary_negatives": [], "source": "offline"}  # 空结果 + offline 标记（外层据此切模板合成）


def _extract_json(text):
    """容错解析：直接 json.loads → 抽 ```json 围栏 → 抽第一个 {...} 块"""
    if not text:                                    # 空输入
        return {}                                   # 返回空 dict
    for candidate in (text.strip(),                 # 候选 1：整段文本直接当 JSON 解析
                      re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S),  # 候选 2：markdown 代码围栏里的 JSON
                      re.search(r"(\{.*\})", text, re.S)):  # 候选 3：贪心抽最外层 {...}（前一个失败时兜底；须带捕获组，否则 group(1) 抛 IndexError 使本分支失效）
        try:
            if candidate is None:                  # 正则没匹配上
                continue                           # 试下一个候选
            raw = candidate if isinstance(candidate, str) else candidate.group(1)  # 字符串直接用，Match 取第 1 组
            return json.loads(raw)                  # 解析成功立即返回
        except Exception:                           # 本候选解析失败
            continue                                # 降级到下一个候选
    return {}                                       # 全部失败 → 空 dict（上游按空数据走重试/离线）


# ── 工厂：按可用性自动选择 ─────────────────────────────────────────────────

def build_synthesizer(force_llm=False, force_offline=False, **kwargs):
    """force_llm=True 时即使无 Key 也返回 LLM 实例（调用时逐条降级离线）"""
    seed = kwargs.pop("seed", 42)                   # 取出（并从 kwargs 移除）seed 参数，离线合成器专用
    if force_offline:                               # 显式要求离线
        return OfflineTemplateSynth(seed=seed)      # 直接给模板合成器
    synth = LLMSynthesizer(**{k: v for k, v in kwargs.items()
                              if k not in ("seed",)})  # 用其余参数建 LLM 合成器（过滤 seed 防传错）
    if synth.available or force_llm:                # 客户端就绪，或强制要 LLM
        return synth                                # 返回 LLM 实例
    print("[llm_synth] 未检测到 DEEPSEEK_API_KEY，使用离线模板合成。")  # 提示降级原因
    return OfflineTemplateSynth(seed=seed)           # 无 Key 且不强制 → 离线模板


if __name__ == "__main__":
    # 冒烟：直接产物预览（离线/联网均可）
    demo_faq = {"id": 1, "question": "血压计怎么正确测量血压？",  # 演示 FAQ：标准问
                "variants": ["如何用血压计测血压？"],           # 一条已知变体
                "answer": "静坐休息5分钟，袖带绑于上臂肘窝上方2~3厘米处，测量时保持安静。"}  # 标准答案
    s = build_synthesizer()                          # 按环境自动选合成器
    r = s.synthesize_faq(demo_faq, n_paraphrase=8, n_negative=4)  # 小批量跑一次
    print("source:", r["source"])                   # 打印数据来源（llm / offline）
    print("paraphrases:", r["paraphrases"])         # 预览同义改写
    print("boundary_negatives:", r["boundary_negatives"])  # 预览边界负例
