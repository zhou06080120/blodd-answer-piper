"""
BM25 词面检索回归门禁 —— P2 评估门禁（CI 轻量版）

设计动机：
  - 全量检索评测（eval_retrieval.py）依赖 BERT 基座权重 + ONNX 后端 + CrossEncoder
    checkpoint，在 CI（无 GPU / 无模型权重）环境无法确定性复现，属于本地/集成环境；
  - BM25 词面检索是纯 numpy 自研实现（hybrid_retriever.BM25，导入链仅 numpy + 标准库），
    且完全确定性（无随机数、无 GPU 浮点误差），天然适合做 CI 回归门禁——零模型权重可跑；
  - 门禁口径：BM25-only 变体留出（hold-out）检索——每个 FAQ 从 variants 末尾留出
    N 条作查询（不在语料中，无泄漏），其余变体 + 标准问作语料建 BM25 索引；
    命中判定：Top1 = 排名第一的语料属于目标 FAQ；Top3 = 前三名语料中任一属于目标 FAQ。

阈值管理（evals/thresholds.json）：
  - {"top1", "top3", "holdout_per_faq", "queries"}：top1/top3 为真实运行实测值
    向下取整（两位小数）并留 3~5 个百分点安全余量后写入；
  - holdout_per_faq / queries 与构造口径绑定：任何人改留出条数或查询规模导致与
    阈值文件不一致时，门禁直接 FAIL（防止悄悄稀释评测集）；
  - gate 断言实测 ≥ 阈值；任一 FAIL 退出码 1（CI 据此拦截合入）。

用法：
  python evals/gate.py     # 首次运行（无 thresholds.json）打印实测值供写入阈值；此后做门禁断言
"""

import json             # 读知识库与阈值文件
import sys              # 运行时模块路径注入
from pathlib import Path  # 项目根定位（不依赖 cwd）

import numpy as np      # BM25 得分向量的 argsort 降序排名

ROOT = Path(__file__).resolve().parents[1]       # 项目根（evals/ 的上一级）
sys.path.insert(0, str(ROOT / "src"))            # 注入 src/（平铺模块架构，BM25 在 src/hybrid_retriever.py）

from hybrid_retriever import BM25  # noqa: E402  # 自研 BM25（中文 char-gram；导入链无 torch，CI 只需 numpy）

KB_PATH = ROOT / "data" / "blood_pressure_expanded" / "faq_extended.json"  # 线上默认知识库（30 FAQ / 1045 变体）
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"               # 门禁阈值文件（实测值取整写入）


# ═══════════════════════════════════════════════════════════════════════════
# 函数：build_holdout
# 功能：构造 BM25-only 留出评测集（确定性，无随机数）
# 流水线位置：gate.py 主流程第一步
# 输入：faqs=知识库列表；holdout_per_faq=每个 FAQ 留出的查询变体数
# 输出：(queries, corpus, fids) —— queries=[(查询文本, 目标FAQ id)]，
#       corpus/fids 为平行列表：语料文本与每条语料所属的 FAQ id
# 口径说明：status 缺省视为 active，只收 active/gray（与 HybridRetriever 线上索引一致）；
#       变体总数不足 3 条的只留最后 1 条（保底语料规模）
# ═══════════════════════════════════════════════════════════════════════════
def build_holdout(faqs, holdout_per_faq=2):
    """构造留出评测集：每 FAQ 取 variants 末尾 N 条作查询，其余变体 + 标准问作语料。"""
    queries, corpus, fids = [], [], []               # 查询对 / 语料文本 / 语料所属 FAQ id
    for f in faqs:                                  # 逐条 FAQ 展开
        if f.get("status", "active") not in ("active", "gray"):  # 缺省视作 active；draft 不参与
            continue                                # 与线上检索索引口径保持一致
        variants = list(f.get("variants", []))      # 该 FAQ 的变体列表
        if not variants:                            # 无变体：无查询可留出（语料仍有标准问）
            hold = []                               # 空留出
        elif len(variants) < 3:                     # 变体太少：只留最后 1 条
            hold = variants[-1:]                     # 保底 1 条，语料不至于枯竭
        else:                                       # 正常：末尾留 N 条
            hold = variants[-holdout_per_faq:]      # 尾部切片（确定性，无随机采样）
        keep = variants[:len(variants) - len(hold)] if hold else variants  # 其余变体进语料
        for t in [f["question"]] + keep:            # 标准问 + 留存变体都各自成一条语料
            corpus.append(t)                        # 记录语料文本
            fids.append(f["id"])                    # 记录该语料归属的 FAQ id
        for t in hold:                              # 留出的查询（不出现在语料中，防泄漏）
            queries.append((t, f["id"]))            # (查询文本, 目标 FAQ id)
    return queries, corpus, fids                    # 返回三件套


# ═══════════════════════════════════════════════════════════════════════════
# 函数：evaluate
# 功能：BM25-only 留出检索评测——全语料建索引，逐查询打分，统计 Top1/Top3 命中
# 输入：queries/corpus/fids（build_holdout 产物）　输出：(top1 命中数, top3 命中数, 查询总数)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(queries, corpus, fids):
    """BM25(corpus) 逐查询打分，argsort 降序取前三名，按 FAQ id 归属判命中。"""
    bm25 = BM25(corpus)                             # 在留出语料上建 BM25 索引（中文 char-gram 分词）
    top1 = top3 = 0                                # 两档命中计数
    for text, fid in queries:                       # 逐条留出查询
        scores = bm25.scores(text)                  # 全语料 BM25 得分向量 [N]
        order = np.argsort(-scores)[:3]             # 降序前三名语料下标（argsort 加负号实现降序）
        if fids[order[0]] == fid:                  # Top1 语料属于目标 FAQ
            top1 += 1                              # Top1 命中
        if fid in {fids[i] for i in order}:         # Top3 内任一语料属于目标 FAQ
            top3 += 1                              # Top3 命中
    return top1, top3, len(queries)                 # 返回命中数与查询规模


def main():
    """跑门禁：读 KB → 构造留出集 → 实测 Top1/Top3 → 与 thresholds.json 阈值比对。"""
    faqs = json.loads(KB_PATH.read_text(encoding="utf-8"))    # 读线上默认知识库
    if not THRESHOLDS_PATH.exists():                          # 首次运行：阈值文件尚未写入
        queries, corpus, fids = build_holdout(faqs)           # 用默认留出口径构造
        top1, top3, n_q = evaluate(queries, corpus, fids)     # 实测
        print("thresholds.json 不存在（首次运行）。实测结果（holdout_per_faq=2）：")
        print(f"  queries={n_q}  top1={top1 / n_q:.4f} ({top1}/{n_q})  top3={top3 / n_q:.4f} ({top3}/{n_q})")
        print(f"  请按实测值留 3~5 个百分点安全余量、向下取整后写入 {THRESHOLDS_PATH}")
        return 0                                              # 引导模式不算失败

    gates = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))  # 读入既定阈值
    queries, corpus, fids = build_holdout(faqs, holdout_per_faq=gates["holdout_per_faq"])  # 按阈值文件绑定的口径构造
    top1, top3, n_q = evaluate(queries, corpus, fids)         # 实测命中数
    top1_rate, top3_rate = top1 / n_q, top3 / n_q             # 实测命中率

    checks = [                                                # 逐项门禁：口径一致性 + 两档阈值
        ("queries", n_q, gates["queries"], "=="),             # 查询规模须与阈值文件记录一致
        ("top1", top1_rate, gates["top1"], ">="),             # 实测 Top1 ≥ 阈值
        ("top3", top3_rate, gates["top3"], ">="),             # 实测 Top3 ≥ 阈值
    ]
    print(f"BM25 词面检索门禁（hold-out：{n_q} 查询 × {len(corpus)} 语料，{len(faqs)} FAQ）")  # 汇总头
    failed = False                                            # 任一 FAIL 即置位
    for name, actual, expect, op in checks:                   # 逐项判定并打印
        ok = (actual == expect) if op == "==" else (actual >= expect)  # 按算符比较
        failed = failed or not ok                             # 累计失败标志
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: 实测 {actual:.4f} {op} 阈值 {expect}")  # 单项结果
    print(f"汇总: Top1 {top1}/{n_q} = {top1_rate:.4f}，Top3 {top3}/{n_q} = {top3_rate:.4f}")  # 实测数字汇总
    if failed:                                                # 任一项失败
        print("门禁未通过：BM25 检索质量或评测口径发生回归，请检查 KB 扩容与分词质量。")
        sys.exit(1)                                           # 退出码 1 → CI 拦截
    print("门禁全部通过。")                                    # 全 PASS
    return 0                                                  # 退出码 0


if __name__ == "__main__":
    main()                                                    # 直跑才执行门禁
