"""
FAQ 知识库扩容 + 硬负例挖掘 + 训练数据重建主流水线（需求 1 落地脚本）

流程：
  1. 读 data/blood_pressure/faq.json（30 条种子 FAQ）
  2. 逐条合成：LLM（DeepSeek）生成同义改写 + 边界负例；失败/不足自动用离线模板补齐；
     结果缓存到 out_dir/raw_synth/<faq_id>.json（重跑不重复消耗 Token）
  3. 语义质量过滤：用现有 BiEncoder 对候选改写打分，
     与标准问相似度 >= filter_sim 才入库（自动剔除 LLM 语义漂移样本）；
     边界负例仅保留"词面相似但意图不同"（sim 处于 [0.3, max_boundary_sim]）的样本
  4. 知识库扩容至千级：data/blood_pressure_expanded/faq_extended.json
     （每 FAQ = 标准问 + <= target_variants 条同义改写，含 status 字段供灰度发布）
  5. 硬负例挖掘：全库编码后，为每条 FAQ 选出"其他 FAQ 中与其最相似"的 top-hard_neg 句
     作为困难负样本，与边界负例、随机负例一起支撑难负例微调
  6. 按 FAQ 维度 8:1:1 切分，重建 train/validation/test.jsonl（正例 cap + 三类负例）

使用方式：
  # 离线冒烟（无 Key/断网可用）
  python synthesize_data.py --no-llm --target_variants 24
  # 完整规模（默认千级 KB + 万级训练对）
  python synthesize_data.py
  # LLM 合成小规模联网验证
  python synthesize_data.py --target_variants 45 --pos_cap 400 --llm_faqs 3

依赖：torch / transformers（过滤与挖掘复用 src/model.py、src/dataset.py）
"""
# ↑ 模块级 docstring：需求 1 的落地流水线。以 30 条种子 FAQ 为起点，走
#   "LLM 合成（断网自动降级离线模板）→ 语义过滤去漂移 → 知识库扩容到千级 →
#   硬负例挖掘 → 重建训练集"五步，产出扩容 KB（faq_extended.json）与万级
#   句对数据集。核心风控点：① raw_synth 缓存防止重跑烧 Token；② 相似度窗口
#   过滤保证正例足够像、负例足够难但不同义，避免训练目标自相矛盾。

import argparse                                   # CLI 参数解析
import json                                       # KB/缓存/报告的 JSON 读写
import random                                     # 固定种子随机（切分/负例采样可复现）
import sys                                       # 运行时模块路径注入
import time                                       # 全流程计时
from pathlib import Path                           # 路径对象

import torch                                       # 张量与设备管理

ROOT = Path(__file__).parent                       # 项目根目录
SRC = ROOT / "src"                                 # 源码目录
sys.path.insert(0, str(SRC))                      # 注入 src/ 到模块搜索路径最前

from dataset import encode_single          # noqa: E402  # 单句编码工具（改路径后导入，豁免 E402）
from model import build_biencoder          # noqa: E402  # 双塔模型工厂
from llm_synth import (OfflineTemplateSynth, build_synthesizer,  # noqa: E402  # 合成器三件套
                       dedupe, normalize)                    # 离线模板合成器 / LLM+离线合成器工厂 / 去重与归一化工具

FAQ_PATH = ROOT / "data" / "blood_pressure" / "faq.json"          # 种子知识库（30 条）
OUT_DIR = ROOT / "data" / "blood_pressure_expanded"                # 扩容数据输出目录（KB+jsonl+缓存+报告）
CKPT_PATH = ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt"  # 基线双塔权重（过滤/挖掘用打分器）
BERT_PATH = r"F:\aaa123\pretrain_models\bert-base-chinese"        # 中文 BERT 基座目录


# ── 编码工具 ────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 函数：encode_texts
# 功能：把一批文本批量编码成 [N, H] 的 L2 归一化句向量矩阵（分批防 OOM）
# 流水线位置：语义过滤与硬负例挖掘的统一编码入口
# 输入：texts=文本列表；batch_size=每批条数　输出：CPU 上的 [N, H] 张量
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, max_length, batch_size=64):
    """批量编码 → [N, H] L2 归一化句向量（CPU tensor）"""
    out = []                                         # 各 batch 结果收集器
    for i in range(0, len(texts), batch_size):      # 按 batch_size 切片遍历
        batch = texts[i:i + batch_size]              # 当前批文本
        ids, masks, ttids = [], [], []               # 三种输入张量的收集器
        for t in batch:                              # 批内逐句编码
            enc = encode_single(tokenizer, t, max_length)  # 分词 → 1D 张量三件套
            ids.append(enc["input_ids"])             # 收集 token ids
            masks.append(enc["attention_mask"])      # 收集注意力掩码
            ttids.append(enc["token_type_ids"])     # 收集句对分段
        vec = model.encode(
            torch.stack(ids).to(device),             # list[1D] 沿新维度堆叠成 [B, L] → 设备
            torch.stack(masks).to(device),            # 同上：掩码
            torch.stack(ttids).to(device),            # 同上：分段
        )                                            # 批量前向 → 池化 → L2 归一化 → [B, H]
        out.append(vec.detach().cpu())               # 断梯度搬回 CPU（后续只做矩阵乘/比较）
    return torch.cat(out, dim=0)                    # 各批沿 batch 维拼接 → [N, H]


# ═══════════════════════════════════════════════════════════════════════════
# 函数：load_biencoder
# 功能：加载 BiEncoder checkpoint 并沿用其训练时结构（pool/层数/max_length），
#       作为后续语义过滤与硬负例挖掘的"打分器"
# 输入：ckpt_path/bert_path/device　输出：(model, tokenizer, max_length)
# ═══════════════════════════════════════════════════════════════════════════
def load_biencoder(ckpt_path, bert_path, device):
    """加载 BiEncoder checkpoint（沿用其 pool / 层数 / max_length 结构），返回 (model, tokenizer, max_length)。"""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)  # 加载 checkpoint（含权重+超参）
    saved = ckpt.get("args", {})                     # 训练时保存的超参
    from transformers import BertTokenizer           # 延迟导入分词器
    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 分词器
    model = build_biencoder(bert_path=bert_path,
                            pool=saved.get("pool", "mean"),
                            num_hidden_layers=saved.get("num_hidden_layers")).to(device)  # 与训练一致的结构重建并搬上设备
    model.load_state_dict(ckpt["state_dict"])       # 载入训练权重
    model.eval()                                     # 推理模式
    return model, tokenizer, saved.get("max_length", 64)  # 返回模型/分词器/训练时的编码长度


# ── 合成 + 过滤 ──────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 函数：collect_candidates
# 功能：为单条 FAQ 收集合成候选——LLM 优先（同义改写+边界负例），数量不足或
#       无 Key 时用离线模板补齐；最后对合并结果去重
# 流水线位置：主流程第 1 步（逐 FAQ 合成）的单条处理函数
# 输入：synth=合成器；faq=单条 FAQ；目标改写数/LLM 改写数/LLM 负例数
# 输出：(paras 候选改写列表, negs 候选边界负例列表, source "llm"/"offline")
# ═══════════════════════════════════════════════════════════════════════════
def collect_candidates(synth, faq, target_variants, n_llm_para, n_llm_neg):
    """LLM 优先 + 离线模板补齐，产出同义候选与边界负例候选"""
    llm_paras, llm_negs, source = [], [], "offline"  # LLM 结果收集器 + 来源标记（默认离线）
    if hasattr(synth, "synthesize_faq"):            # 有 synthesize_faq 方法 → 是 LLM 合成器
        r = synth.synthesize_faq(faq, n_paraphrase=n_llm_para, n_negative=n_llm_neg)  # 一次调用要改写+负例
        llm_paras, llm_negs, source = r["paraphrases"], r["boundary_negatives"], r["source"]  # 拆结果；source 标明 llm/offline（失败内部已降级）
    offline = OfflineTemplateSynth(seed=faq["id"] * 7 + 42)  # 离线模板合成器（种子与 FAQ 编号绑定，可复现）
    # 模板补齐到目标规模（离线兜底）
    need = max(0, target_variants - len(llm_paras))  # LLM 改写不足目标数的缺口
    tpl = offline.generate_paraphrases(faq, need) if need else []  # 有缺口才生成模板改写
    paras = dedupe(llm_paras + tpl, existing=[faq["question"]] + list(faq.get("variants", [])))  # 合并去重（不得与原问/已有改写重复）
    negs = dedupe(llm_negs + offline.generate_negatives(faq, n_llm_neg))  # 边界负例：LLM + 模板合并去重
    return paras, negs, source                       # 返回两类候选与来源标记


# ── 主流程 ──────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 函数：main
# 功能：扩容总入口——四步串行：① 逐 FAQ 合成（带磁盘缓存）② 语义过滤建扩容 KB
#       ③ 全库硬负例挖掘 ④ 按 FAQ 维度切分重建训练集，最后落盘报告
# 输入：命令行参数　输出：faq_extended.json + train/val/test.jsonl + synthesis_report.json
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """扩容主流程：逐 FAQ 合成（带缓存）→ 语义过滤 → 硬负例挖掘 → 切分重建训练集。"""
    args = parse_args()                               # 解析 CLI 超参
    t0 = time.time()                                  # 记录流程起点（报告要算总耗时）
    OUT_DIR.mkdir(parents=True, exist_ok=True)        # 确保输出目录存在
    rng = random.Random(args.seed)                    # 全局随机源（种子固定可复现）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选设备
    print(f"设备: {device}")                          # 打印设备

    faqs = json.loads(Path(args.faq).read_text(encoding="utf-8"))  # 读种子 FAQ 库
    print(f"加载种子 FAQ {len(faqs)} 条")             # 打印种子规模

    synth = build_synthesizer(force_llm=args.force_llm, force_offline=args.no_llm,
                              seed=args.seed)         # 构建合成器：默认自动探测（有 Key 走 LLM，无 Key 离线）
    model, tokenizer, max_length = load_biencoder(args.ckpt, args.bert_path, device)  # 加载打分用双塔
    print("过滤/挖掘用 BiEncoder 已加载")              # 打印就绪

    # ── 第 1 步：逐 FAQ 合成（带缓存）──────────────────────────────────────
    raw_dir = OUT_DIR / "raw_synth"                   # 原始合成缓存目录
    raw_dir.mkdir(parents=True, exist_ok=True)       # 建缓存目录
    paras_by_faq, negs_by_faq = {}, {}                # fid → 改写候选 / 边界负例候选
    src_stat = {"llm": 0, "offline": 0, "cache": 0}   # 来源统计（llm/离线/缓存命中各多少条 FAQ）
    for f in faqs:                                   # 逐条种子 FAQ 处理
        fid = f["id"]                                 # 当前 FAQ 编号
        cache = raw_dir / f"{fid}.json"               # 该 FAQ 的缓存文件路径
        if cache.exists() and not args.fresh:         # 缓存存在且未强制刷新
            raw = json.loads(cache.read_text(encoding="utf-8"))  # 直接读缓存
            paras_by_faq[fid], negs_by_faq[fid] = raw["paraphrases"], raw["boundary_negatives"]  # 还原两类候选
            src_stat["cache"] += 1                    # 统计缓存命中
            continue                                  # 跳过昂贵的合成调用
        paras, negs, source = collect_candidates(     # 现算：LLM 优先 + 模板兜底
            synth, f, args.target_variants, args.llm_para, args.llm_neg)
        cache.write_text(json.dumps(                  # 结果写回缓存（下次重跑免 Token/免耗时）
            {"id": fid, "source": source,
             "paraphrases": paras, "boundary_negatives": negs},
            ensure_ascii=False, indent=1), encoding="utf-8")
        paras_by_faq[fid], negs_by_faq[fid] = paras, negs  # 存入内存总表
        src_stat["llm" if source == "llm" else "offline"] += 1  # 按实际来源计数
        print(f"  FAQ#{fid:>2}: 改写候选 {len(paras):>3} | 边界负例 {len(negs):>3} | {source}")  # 打印单条进展

    # ── 第 2 步：语义过滤（LLM/模板漂移样本剔除）──────────────────────────
    print("\n语义质量过滤中...")                       # 打印阶段标题
    kb_faqs = []                                      # 扩容 KB 收集器
    kept_stat = []                                    # 每 FAQ 的保留统计
    for f in faqs:                                   # 逐条 FAQ 过滤
        fid = f["id"]                                 # 当前编号
        orig = [f["question"]] + list(f.get("variants", []))  # 原有问法集合（归一化比对基准）
        anchor_vec = encode_texts(model, tokenizer, [f["question"]], device, max_length)  # 标准问编码成锚向量 [1, H]
        paras = paras_by_faq.get(fid, [])             # 该 FAQ 的改写候选
        vecs = encode_texts(model, tokenizer, paras, device, max_length) if paras else None  # 候选批量编码（无候选则跳过）
        sims = (vecs @ anchor_vec.t()).squeeze(-1).tolist() if vecs is not None else []  # L2 归一化后点积=余弦相似度 → [N]
        accepted = [t for t, s in zip(paras, sims)
                    if s >= args.filter_sim and normalize(t) not in {normalize(o) for o in orig}]  # 双条件：相似度达标 + 与原问不同
        variants = list(f.get("variants", [])) + accepted  # 新 variants = 原有 + 过滤通过的候选
        variants = dedupe(variants, existing=[f["question"]])[:args.target_variants]  # 去重并截断到目标规模
        # 边界负例过滤：sim 落在 [neg_min_sim, neg_max_sim] 才保留（真正"词面近、意图远"）
        negs = negs_by_faq.get(fid, [])               # LLM/模板产出的边界负例候选
        nvecs = encode_texts(model, tokenizer, negs, device, max_length) if negs else None  # 批量编码
        nsims = (nvecs @ anchor_vec.t()).squeeze(-1).tolist() if nvecs is not None else []  # 与锚的相似度
        kept_negs = [t for t, s in zip(negs, nsims)
                     if args.neg_min_sim <= s <= args.neg_max_sim]  # 只留相似度窗口内的（太不像=简单负例，太像=疑似同义）
        tpl_neg = OfflineTemplateSynth(seed=fid * 13 + 7).generate_negatives(f, args.llm_neg)  # 另一套种子的模板负例补充
        tpl_vecs = encode_texts(model, tokenizer, tpl_neg, device, max_length) if tpl_neg else None  # 编码模板负例
        tpl_sims = (tpl_vecs @ anchor_vec.t()).squeeze(-1).tolist() if tpl_vecs is not None else []  # 模板负例相似度
        kept_negs += [t for t, s in zip(tpl_neg, tpl_sims)
                      if args.neg_min_sim <= s <= args.neg_max_sim]  # 窗口内的模板负例并入
        kept_negs = dedupe(kept_negs)[:args.llm_neg]  # 去重并截断到每 FAQ 上限
        negs_by_faq[fid] = kept_negs                  # 回写过滤后的边界负例

        kb_faqs.append({"id": f["id"], "question": f["question"],
                        "answer": f["answer"], "variants": variants, "status": "active"})  # 组装扩容 KB 条目（status 供灰度发布）
        kept_stat.append({"id": fid, "accepted": len(variants), "boundary_neg": len(kept_negs)})  # 记录该 FAQ 保留规模

    kb_path = OUT_DIR / "faq_extended.json"            # 扩容 KB 路径
    kb_path.write_text(json.dumps(kb_faqs, ensure_ascii=False, indent=2), encoding="utf-8")  # 落盘 KB
    n_q = sum(1 + len(f["variants"]) for f in kb_faqs)  # 候选问法总数 = Σ(1 标准问 + N 改写)
    print(f"知识库已扩容 → {kb_path}（{len(kb_faqs)} 条 FAQ / {n_q} 个候选问法）")  # 打印扩容成果

    # ── 第 3 步：硬负例挖掘 ─────────────────────────────────────────────────
    # 教训：硬负例若与锚文本语义几乎相同（sim>hard_neg_max，多为跨意图近同义句），
    # 会与正例目标直接矛盾，配合正负例严重失衡，训练会出现"全库向量坍缩为同一方向"。
    # 过滤规则：相似度需落在 [hard_neg_min, hard_neg_max] 窗口——足够难但又不是同义句。
    print("硬负例挖掘中...")                            # 打印阶段标题
    texts, mapping = [], []          # mapping: (text, faq_id)    # 全库问法文本列表 + 对齐的 FAQ 编号列表
    faq_by_id = {f["id"]: f for f in kb_faqs}          # 编号 → 扩容 FAQ 查表
    for f in kb_faqs:                                 # 展开扩容 KB
        for t in [f["question"]] + f["variants"]:     # 标准问 + 全部改写
            texts.append(t)                           # 收集文本
            mapping.append(f["id"])                   # 收集归属编号（注意：此处 mapping 只存 fid）
    all_vecs = encode_texts(model, tokenizer, texts, device, max_length)   # [N, H] 全库编码（后续反复复用，只算一次）
    hard_by_faq = {}                                  # fid → 硬负例文本列表
    hard_sim_stat = []                                # 全部硬负例的相似度（打印分布用）
    sim_map = {}    # fid -> {其他FAQ文本: 与锚标准问的相似度}，供随机负例窗口过滤复用  # 复用挖掘期相似度，避免第 4 步重算
    faq_idx = {fid: [i for i, m in enumerate(mapping) if m == fid] for fid in faq_by_id}  # fid → 该 FAQ 全部问法在 texts 中的下标列表
    for f in kb_faqs:                                 # 逐 FAQ 挖掘
        q_idx = faq_idx[f["id"]][0]                       # 标准问在库中的下标（构造顺序保证 [0] 即标准问）
        qv = all_vecs[q_idx]                          # 锚向量 [H]
        others = [i for i, m in enumerate(mapping) if m != f["id"]]  # 其他所有 FAQ 的问法下标
        sims = all_vecs[others] @ qv                  # 锚与其余全库的相似度 [M]
        sim_map[f["id"]] = {texts[others[k]]: float(sims[k]) for k in range(len(others))}  # 文本→相似度字典（第 4 步复用）
        norm_q = normalize(f["question"])              # 锚问法归一化（去重比对）
        top = sims.topk(min(args.hard_neg * 6, len(others))).indices.tolist()  # 取 6 倍候选下标（过滤后仍够 hard_neg 条）
        picked, seen = [], set()                      # 入选硬负例 + 归一化去重集
        for k in top:                                 # 相似度从高到低遍历
            t = texts[others[k]]                      # 候选负例文本
            s = float(sims[k])                        # 其与锚的相似度
            nt = normalize(t)                         # 归一化文本
            if nt in seen or nt == norm_q:            # 重复或与锚同句 → 跳过
                continue
            if not (args.hard_neg_min <= s <= args.hard_neg_max):  # 相似度不在"够难又非同义"窗口 → 跳过
                continue
            seen.add(nt)                              # 登记去重
            picked.append((t, round(s, 3)))            # 入选（保留 3 位小数的相似度）
            if len(picked) >= args.hard_neg:          # 已挖满目标条数
                break                                 # 提前结束
        hard_by_faq[f["id"]] = [t for t, _ in picked]  # 存该 FAQ 的硬负例文本
        hard_sim_stat.extend(s for _, s in picked)    # 收集相似度样本
    if hard_sim_stat:                                 # 确实挖到了硬负例
        print(f"  硬负例 sim 范围: [{min(hard_sim_stat):.2f}, {max(hard_sim_stat):.2f}]"
              f"（挖掘 {len(hard_sim_stat)} 条，窗口 [{args.hard_neg_min}, {args.hard_neg_max}]）")  # 打印相似度分布与窗口，人工复核风控

    # ── 第 4 步：切分 + 构建训练集 ─────────────────────────────────────────
    ids = list(faq_by_id.keys())                      # 全部 FAQ 编号
    other_faq_pool = {fid: [t for x in kb_faqs if x["id"] != fid
                            for t in [x["question"]] + x["variants"]]
                      for fid in ids}                 # fid → 其他 FAQ 的全部问法池（随机负例采样源）
    rng.shuffle(ids)                                  # 打乱编号（决定 split 成员）
    n_train, n_val = int(len(ids) * 0.8), int(len(ids) * 0.1)  # 8:1:1 条数切分
    splits = {"train": ids[:n_train], "validation": ids[n_train:n_train + n_val],
              "test": ids[n_train + n_val:]}           # 三个 split 的编号列表（按 FAQ 维度防泄漏）
    stats = {}                                        # 各 split 的规模统计
    for split, sids in splits.items():                # 逐 split 构建
        rows = []                                     # 该 split 的句对收集器
        for fid in sids:                              # 逐 FAQ 生成正负例
            f = faq_by_id[fid]                        # 当前 FAQ
            # 正例：标准问 × 变体 + 变体间两两采样（复用挖掘步的向量，按相似度地板过滤
            # LLM/模板漂移产生的"弱正例"——正例过弱同样会与负例目标冲突）
            idxs = faq_idx[fid]                       # 该 FAQ 全部问法在 texts 中的下标
            sents = [texts[i] for i in idxs]          # 对应文本列表（sents[0] 恒为标准问）
            vecs = all_vecs[idxs]                     # 对应向量 [K, H]（复用第 3 步全库编码，零额外计算）
            sim_mat = (vecs @ vecs.t()).cpu().numpy()  # 句间两两相似度矩阵 [K, K]（numpy 便于下标索引）
            q_ok = [j for j in range(1, len(sents)) if sim_mat[0, j] >= args.pos_min_sim]  # 标准问 × 达标变体的正例（相似度地板过滤）
            pos = [{"sentence1": sents[0], "sentence2": sents[j], "label": 1} for j in q_ok]  # 生成"标准问×变体"正例
            vv = [(a, b) for a in range(1, len(sents)) for b in range(a + 1, len(sents))
                  if sim_mat[a, b] >= args.pos_min_sim][:max(0, args.pos_cap - len(pos))]  # 变体×变体正例补足 cap 缺口（同样过地板）
            pos += [{"sentence1": sents[a], "sentence2": sents[b], "label": 1} for a, b in vv]  # 并入"变体×变体"正例
            # 负例：硬负例挖掘 + 边界负例 + 随机跨 FAQ（扩容语料池，改善正负比）
            neg = [{"sentence1": f["question"], "sentence2": t, "label": 0}
                   for t in hard_by_faq[fid]]        # 第一类：硬负例（全库最近邻的跨意图句）
            neg += [{"sentence1": f["question"], "sentence2": t, "label": 0}
                    for t in negs_by_faq[fid][:args.boundary_neg]]  # 第二类：边界负例（词面近、意图远）
            used = {normalize(r["sentence2"]) for r in neg} | {normalize(f["question"])}  # 已用负例文本（归一化）∪ 锚自身
            s_map = sim_map.get(fid, {})              # 第 3 步存好的"他库问法→相似度"字典
            pool = [q for q in other_faq_pool[fid]
                    if normalize(q) not in used and s_map.get(q, 1.0) <= args.hard_neg_max]  # 随机负例池：排重 + 排除疑似同义句（sim 超上界者不入选，同硬负例风控）
            neg += [{"sentence1": f["question"], "sentence2": q, "label": 0}
                    for q in rng.sample(pool, min(args.random_neg, len(pool)))]  # 第三类：随机负例（池不足时全取）
            rows += pos + neg                         # 该 FAQ 的正负例并入 split
        rng.shuffle(rows)                              # 打乱句对顺序
        out_path = OUT_DIR / f"{split}.jsonl"          # 该 split 输出文件
        with open(out_path, "w", encoding="utf-8") as fh:  # 覆盖写打开
            for r in rows:                            # 逐行写出
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")  # 单行 JSON + 换行
        pos_n = sum(1 for r in rows if r["label"] == 1)  # 统计正例条数
        stats[split] = {"pairs": len(rows), "pos": pos_n, "neg": len(rows) - pos_n,
                        "faqs": len(sids)}            # 记录该 split 的规模
        print(f"  {split:10s}: {len(rows):>6} 对（正 {pos_n:>5} / 负 {len(rows) - pos_n:>4}）→ {out_path.name}")  # 打印 split 概览

    report = {                                        # 汇总运行报告（复盘/汇报用）
        "seed_faqs": len(faqs),                        # 种子 FAQ 数
        "kb_questions": n_q,                          # 扩容后候选问法总数
        "avg_variants_per_faq": round(sum(len(f["variants"]) for f in kb_faqs) / len(kb_faqs), 1),  # 平均每 FAQ 改写数
        "source": src_stat,                           # 合成来源统计（llm/offline/cache）
        "splits": stats,                              # 各 split 规模
        "elapsed_s": round(time.time() - t0, 1),      # 总耗时（秒）
        "args": vars(args),                           # 本次全量超参（保证可追溯）
    }
    (OUT_DIR / "synthesis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")  # 报告落盘
    print(f"\n完成（{report['elapsed_s']}s）。报告 → {OUT_DIR / 'synthesis_report.json'}")  # 打印完成信息与报告路径


# ═══════════════════════════════════════════════════════════════════════════
# 函数：parse_args
# 功能：解析扩容流水线全部超参（合成规模/各类相似度窗口/三类负例配比）
# 输入：无　输出：argparse.Namespace（main 中消费）
# 关键参数设计：所有相似度窗口都是"防训练目标矛盾"的风控手段——
#   filter_sim 管正例质量、[neg_min_sim, neg_max_sim] 管边界负例、
#   [hard_neg_min, hard_neg_max] 管硬负例、pos_min_sim 管正例地板
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    """解析知识库扩容与硬负例挖掘命令行的各项参数。"""
    p = argparse.ArgumentParser(description="FAQ 知识库扩容 + 硬负例挖掘 + 数据重建")  # 参数解析器
    p.add_argument("--faq", default=str(FAQ_PATH), type=str)  # 种子 FAQ 库路径
    p.add_argument("--out_dir", default=str(OUT_DIR), type=str)  # 输出目录
    p.add_argument("--ckpt", default=str(CKPT_PATH), type=str)  # 过滤/挖掘用双塔权重
    p.add_argument("--bert_path", default=BERT_PATH, type=str)  # BERT 基座目录
    p.add_argument("--target_variants", default=35, type=int,
                   help="每 FAQ 最终同义改写数（KB 扩容目标 = 30×(1+N)）")  # KB 扩容目标：每 FAQ 保留的改写上限
    p.add_argument("--llm_para", default=30, type=int, help="LLM 每条 FAQ 生成改写数")  # 向 LLM 请求的改写数
    p.add_argument("--llm_neg", default=10, type=int, help="LLM 每条 FAQ 生成边界负例数")  # 向 LLM 请求的边界负例数（兼作保留上限）
    p.add_argument("--filter_sim", default=0.78, type=float, help="改写入库最低相似度")  # 改写入库地板：低于视为语义漂移
    p.add_argument("--neg_min_sim", default=0.30, type=float)  # 边界负例相似度下界（低于=简单负例交给随机负例）
    p.add_argument("--neg_max_sim", default=0.85, type=float)  # 边界负例相似度上界（高于=疑似同义，剔除）
    p.add_argument("--pos_cap", default=150, type=int, help="每 FAQ 训练正例对数上限")  # 正例对上限（防正例淹没负例）
    p.add_argument("--pos_min_sim", default=0.70, type=float,
                   help="正例对最低相似度地板（低于此值的 LLM/模板漂移正例不参与训练）")  # 正例地板：弱正例会与负例目标冲突
    p.add_argument("--hard_neg", default=10, type=int, help="每 FAQ 硬负例挖掘数")  # 硬负例目标条数
    p.add_argument("--hard_neg_min", default=0.45, type=float,
                   help="硬负例相似度下界（低于视为简单负例，交给随机负例）")  # 硬负例窗口下界
    p.add_argument("--hard_neg_max", default=0.92, type=float,
                   help="硬负例相似度上界（高于视为跨意图同义句，必须剔除以免训练目标矛盾）")  # 硬负例窗口上界（防坍缩关键）
    p.add_argument("--boundary_neg", default=10, type=int)  # 训练集里每 FAQ 并入的边界负例条数
    p.add_argument("--random_neg", default=40, type=int, help="每 FAQ 随机跨意图负例数")  # 随机负例条数（撑正负比）
    p.add_argument("--seed", default=42, type=int)   # 全局随机种子
    p.add_argument("--no-llm", action="store_true", help="强制离线模板合成")  # 开关：断网/无 Key 冒烟
    p.add_argument("--force-llm", action="store_true", help="强制使用 LLM（无 Key 时逐条降级）")  # 开关：强制 LLM（内部逐条降级兜底）
    p.add_argument("--fresh", action="store_true", help="忽略 raw_synth 缓存重新生成")  # 开关：忽略缓存重合成
    return p.parse_args()                             # 解析并返回参数命名空间


if __name__ == "__main__":                           # 直接运行本脚本时才执行（被导入时不触发）
    main()                                           # 调用扩容主流程
