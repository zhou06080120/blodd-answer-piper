"""
低置信度拦截与主动学习冷启动闭环（需求 4 落地）

组成：
  1. AnnotationQueue   —— 低置信度 Bad Case 异步落盘队列（JSONL，目录自愈），
                          支持 附加到已有 FAQ / 标记新意图 / 拒绝 三类流转
  2. IntentClustering  —— 对队列中未知意图问题做向量聚类（余弦距离 + 层次聚类，
                          无需预设簇数），挖掘高频未覆盖问题
  3. ActiveLearning    —— 周级迭代：聚类摘要 → 生成 draft FAQ（灰度）→ promote 转正；
                          draft 不参与在线检索（见 hybrid_retriever，status 过滤）

使用方式：
  python active_learning.py queue-list --limit 20
  python active_learning.py queue-label --id <uuid> --faq_id 8
  python active_learning.py iterate --preview          # 聚类挖掘高频未知意图（只读）
  python active_learning.py apply                      # 把达标簇转成 draft FAQ 写入 KB
  python active_learning.py promote --ids 31 32        # 灰度转正
"""
# ↑ 模块级 docstring：线上"接不住的问题"如何沉淀→聚类→入库→转正的运营闭环

import argparse                                   # CLI 子命令解析
import json                                       # 读写 KB 与队列的 JSON
import uuid                                       # 生成队列记录唯一 id
from datetime import datetime                    # 记录时间戳
from pathlib import Path                          # 路径处理

ROOT = Path(__file__).parent                      # 本文件所在目录（= 项目根，注意此脚本在根目录）
KB_PATH = ROOT / "data" / "blood_pressure_expanded" / "faq_extended.json"  # 扩展知识库路径（draft/gray/active 条目都存这）
FEEDBACK_DIR = ROOT / "data" / "feedback"         # 待标注队列落盘目录


# ── 待标注队列 ──────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：AnnotationQueue 待标注队列
# 一句话概括：把线上低置信问题 append-only 追加写入 JSONL，提供按 id 改状态（attach/new_intent/reject）的流转能力。
# 在流水线中的位置：serve.py 拦截低置信问题后异步入队，本脚本提供人工标注与统计入口。
# ═════════════════════════════════════════════════════════════════════════
class AnnotationQueue:
    """低置信度 Bad Case 异步落盘队列（append-only JSONL + 状态流转）"""

    def __init__(self, data_dir=None):
        self.dir = Path(data_dir) if data_dir else FEEDBACK_DIR  # 指定目录或默认 feedback 目录
        self.dir.mkdir(parents=True, exist_ok=True)        # 目录自愈  # 目录不存在则创建（服务首次启动/目录被清都能恢复）
        self.path = self.dir / "queue.jsonl"              # 队列文件固定名 queue.jsonl

    def append(self, query, confidence, candidates, source="low_confidence"):
        """低置信度问题入队（服务端用 asyncio.to_thread 调用，不阻塞事件循环）"""
        record = {                                  # 组装入队记录
            "id": uuid.uuid4().hex[:12],             # 12 位十六进制唯一 id（人工标注时 --id 引用它）
            "ts": datetime.now().isoformat(timespec="seconds"),  # 秒级时间戳
            "query": query,                          # 用户原话
            "confidence": round(float(confidence), 4),  # 当时的置信度（4 位小数）
            "candidates": candidates[:5],            # Top 候选截前 5 条（辅助人工判断挂到哪个 FAQ）
            "source": source,                        # 入队来源（默认低置信拦截）
            "status": "pending",                     # 初始状态：待处理
            "faq_id": None,                         # 若 attach 流转，这里记录挂载的 FAQ 编号
            "cluster_id": None,                      # 预留：聚类编号回写
        }
        with open(self.path, "a", encoding="utf-8") as f:  # 追加模式打开（append-only，不锁不竞争）
            f.write(json.dumps(record, ensure_ascii=False) + "\n")  # 一行一条 JSON（JSONL 格式）
        return record                                # 返回记录（服务层可能要回显给日志）

    def list(self, status=None, limit=50):
        """读取队列（可选按状态过滤），返回最近的若干条记录（按写入顺序）。"""
        if not self.path.exists():                  # 队列文件还没生成过
            return []                               # 视作空队列
        rows = [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]  # 全量读入解析每行
        if status:                                  # 指定状态过滤
            rows = [r for r in rows if r.get("status") == status]  # 只留该状态
        return rows[-limit:]                        # 取末尾 limit 条（最近的记录）

    def _rewrite(self, rows):
        self.path.write_text(                       # 全量覆写队列文件（update 用）
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),  # 每行重新序列化
            encoding="utf-8")                        # UTF-8 编码

    def update(self, record_id, **fields):
        """按 id 更新队列记录字段并刷新时间戳（全量重写 JSONL，量小可接受）。"""
        rows = [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]  # 读出全部记录
        for r in rows:                               # 逐条找目标
            if r["id"] == record_id:                # id 匹配
                r.update(fields)                     # 合并更新字段
                r["ts"] = datetime.now().isoformat(timespec="seconds")  # 刷新时间戳（体现最近处理时间）
        self._rewrite(rows)                          # 全量写回（队列小，代价可接受）

    def label_attach(self, record_id, faq_id):
        self.update(record_id, status="attached", faq_id=int(faq_id))  # 流转一：这条问题其实属于已有 FAQ

    def label_new_intent(self, record_id, question=None):
        self.update(record_id, status="new_intent", faq_id=None,
                    query=question or None)          # 流转二：新意图（可顺手改写规范问法）

    def reject(self, record_id):
        self.update(record_id, status="rejected")   # 流转三：无效/捣乱问题，直接拒绝

    def stats(self):
        """统计各状态记录数，供 /stats 接口展示队列水位。"""
        rows = [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()] \
            if self.path.exists() else []            # 读全量（文件不存在给空表）
        by = {}                                      # 状态 → 数量
        for r in rows:                               # 遍历计数
            by[r["status"]] = by.get(r["status"], 0) + 1  # 累加
        by["total"] = len(rows)                      # 附上总数（队列水位）
        return by                                    # 返回统计 dict


# ── 意图聚类（高频未知问题挖掘）──────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：IntentClustering 意图聚类
# 一句话概括：把 pending 队列里的问题编码成向量，用层次聚类按余弦距离自动分组，找出高频未覆盖意图。
# 在流水线中的位置：iterate/apply 子命令的核心；结果决定哪些簇值得生成新 FAQ。
# ═════════════════════════════════════════════════════════════════════════
class IntentClustering:
    """未知意图聚类：余弦距离 + 层次聚类（distance_threshold 自动定簇数）"""

    def __init__(self, dist_threshold=0.35, min_size=3):
        self.dist_threshold = dist_threshold        # 距离阈值：两问余弦距离小于它才并簇
        self.min_size = min_size                     # 最小簇规模：小于它的簇不值得建 FAQ

    def cluster(self, records, dense_encode):
        """records: [{query, confidence}]；返回簇摘要（按规模降序）"""
        recs = [r for r in records if r.get("status", "pending") == "pending"]  # 只聚类还没处理过的
        if len(recs) < 2:                           # 少于 2 条没法聚
            return [{"cluster_id": 0, "size": len(recs),
                     "samples": [r["query"] for r in recs],
                     "avg_conf": round(sum(r["confidence"] for r in recs) /
                                       len(recs), 3) if recs else 0}]  # 退化为"单簇"摘要（空时 conf 记 0）
        import numpy as np                           # 函数内导入：仅聚类时需要
        from sklearn.cluster import AgglomerativeClustering  # 层次聚类器

        texts = [r["query"] for r in recs]          # 抽出问题文本
        vecs = np.asarray(dense_encode(texts), dtype="float32")  # 批量编码成向量矩阵 [N, H]
        labels = AgglomerativeClustering(           # 层次凝聚聚类：自底向上合并最近的两簇
            n_clusters=None, metric="cosine", linkage="average",  # 不预设簇数；余弦距离；类平均链接
            distance_threshold=self.dist_threshold).fit_predict(vecs)  # 距离低于阈值即停 → 自动确定簇数

        groups = {}                                  # 簇标签 → 记录列表
        for r, lab in zip(recs, labels):             # 记录与簇标签配对
            groups.setdefault(int(lab), []).append(r)  # 归入对应簇
        summary = []                                 # 簇摘要列表
        for cid in sorted(groups, key=lambda c: -len(groups[c])):  # 簇按规模降序（大簇优先看）
            g = groups[cid]                          # 该簇的全部记录
            samples = [r["query"] for r in g][:3]    # 取前 3 条作代表样例
            summary.append({"cluster_id": cid, "size": len(g),
                            "samples": samples,
                            "avg_conf": round(sum(r["confidence"] for r in g) /
                                             len(g), 3)})  # 摘要：簇号/规模/样例/平均置信
        return summary                               # 返回按规模排序的簇摘要


# ── 周级迭代 + 灰度发布 ─────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════
# 功能说明：ActiveLearning 周级迭代器
# 一句话概括：把达标簇自动生成 draft FAQ 写入 KB（附答案占位），人工审核补答案后 promote 转正上线。
# 在流水线中的位置：数据闭环的"入库"环节；draft 状态保证未经审核的内容不会影响线上检索。
# ═════════════════════════════════════════════════════════════════════════
class ActiveLearning:
    """聚类挖掘 → draft 入库（灰度）→ promote 转正"""

    def __init__(self, kb_path=None, feedback_dir=None, queue=None):
        self.kb_path = Path(kb_path) if kb_path else KB_PATH  # 知识库路径（可覆盖）
        self.queue = queue or AnnotationQueue(feedback_dir)  # 复用传入队列或自建一个

    def load_kb(self):
        return json.loads(self.kb_path.read_text(encoding="utf-8"))  # 读 KB（JSON 数组）

    def save_kb(self, faqs):
        self.kb_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 KB 目录存在
        backup = self.kb_path.with_suffix(".json.bak")  # 备份路径（faq_extended.json → .json.bak）
        backup.write_text(self.kb_path.read_text(encoding="utf-8"), encoding="utf-8")  # 先备份旧文件（改坏可回滚）
        self.kb_path.write_text(json.dumps(faqs, ensure_ascii=False, indent=2),
                                encoding="utf-8")   # 再写新内容（缩进 2、中文原样）
        return backup                               # 返回备份路径

    def apply_clusters(self, summary, source="active_learning_weekly", min_size=3):
        """达标簇 → draft FAQ（不入在线索引，待人工审核后 promote）"""
        faqs = self.load_kb()                        # 读当前 KB
        max_id = max(f["id"] for f in faqs)          # 现有最大 id，新条目顺延
        created = []                                 # 本次新建的 draft 列表
        for c in [s for s in summary if s.get("size", 0) >= min_size]:  # 只处理达最小规模的簇
            max_id += 1                              # id 递增
            draft = {                                # 组装 draft 条目
                "id": max_id,                        # 新编号
                "question": c["samples"][0],         # 簇内第一条作标准问
                "variants": c["samples"][1:],       # 其余样例作变体（天然同义）
                "answer": "（该条目由低置信度问题聚类自动生成，答案待人工补充审核）",  # 答案占位，等人工填
                "status": "draft",                   # 草稿态：hybrid_retriever 会过滤它，不影响线上
                "source": source,                   # 溯源标记：周级主动学习生成
                "cluster_size": c["size"],           # 记录簇规模（审核参考：越大越优先）
            }
            faqs.append(draft)                       # 追加进 KB
            created.append(draft)                    # 登记到新建列表
        self.save_kb(faqs)                           # 落盘（含备份）
        return created                               # 返回新建条目

    def promote(self, faq_ids):
        """灰度转正：draft/gray → active"""
        faqs = self.load_kb()                        # 读 KB
        ids = set(int(i) for i in faq_ids)           # 待转正 id 集合
        done = []                                    # 实际转正成功的 id
        for f in faqs:                               # 遍历条目
            if f["id"] in ids and f.get("status", "active") != "active":  # 命中且当前非 active
                f["status"] = "active"               # 转正 → 立即参与线上检索
                f.pop("source", None)                # 清除溯源标记（转正后就是正式条目）
                f.pop("cluster_size", None)          # 清除簇规模（同上）
                done.append(f["id"])                # 记录成功 id
        self.save_kb(faqs)                           # 落盘
        return done                                  # 返回转正结果


# ── 命令行 ─────────────────────────────────────────────────────────────────

def _load_dense_for_cli():
    """聚类需要编码器：优先 ONNX，回退 PyTorch"""
    import sys                                       # （保留）标准库
    sys.path.insert(0, str(ROOT / "src"))            # 把 src/ 加进模块路径，才能 import onnx_runtime
    from onnx_runtime import load_dense_encoder      # 双后端编码器工厂
    ckpt = ROOT / "outputs" / "checkpoints" / "biencoder_hardneg_best.pt"  # 优先用难负例训练的权重
    if not ckpt.exists():                            # 难负例权重不存在
        ckpt = ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt"  # 回退到 cosine 训练的
    enc, meta = load_dense_encoder(ckpt, r"F:\aaa123\pretrain_models\bert-base-chinese",
                                   onnx_dir=ROOT / "outputs" / "onnx")  # int8→fp32→torch 自动降级加载
    print(f"[active_learning] 编码后端: {meta['backend']}-{meta.get('kind', '')} "
          f"({meta.get('onnx') or meta.get('device')})")  # 打印实际用哪个后端
    return enc                                       # 返回编码器（cluster 用它的 encode）


def main():
    """CLI 入口：分发 queue-list / queue-label / iterate / apply / promote 子命令。"""
    ap = argparse.ArgumentParser(description="主动学习闭环工具")  # 主解析器
    sub = ap.add_subparsers(dest="cmd", required=True)  # 必须给一个子命令

    p_list = sub.add_parser("queue-list")             # 子命令 1：查看队列
    p_list.add_argument("--status", default=None)    # 可按状态过滤
    p_list.add_argument("--limit", default=20, type=int)  # 默认看最近 20 条

    p_label = sub.add_parser("queue-label")           # 子命令 2：人工标注
    p_label.add_argument("--id", required=True)      # 必填：记录 id
    p_label.add_argument("--action", required=True,
                         choices=["attach", "new_intent", "reject"])  # 必填：三种流转动作
    p_label.add_argument("--faq_id", type=int, default=None)  # attach 时要挂的 FAQ 编号

    p_iter = sub.add_parser("iterate")                # 子命令 3：聚类预览（只读）
    p_iter.add_argument("--dist", default=0.35, type=float)  # 聚类距离阈值
    p_iter.add_argument("--min-size", default=3, type=int)  # 最小簇规模

    sub.add_parser("apply")                           # 子命令 4：达标簇写 KB（draft）
    p_promo = sub.add_parser("promote")               # 子命令 5：draft 转正
    p_promo.add_argument("--ids", nargs="+", required=True, type=int)  # 至少一个 FAQ id

    args = ap.parse_args()                            # 解析 CLI
    q = AnnotationQueue()                             # 队列实例（默认 feedback 目录）
    if args.cmd == "queue-list":                      # ── 查看队列 ──
        for r in q.list(args.status, args.limit):      # 拉取记录
            print(f"[{r['status']:>10}] {r['id']} conf={r['confidence']} 「{r['query']}」")  # 逐条打印状态/id/置信/原文
    elif args.cmd == "queue-label":                   # ── 人工标注 ──
        if args.action == "attach":                   # 挂到已有 FAQ
            q.label_attach(args.id, args.faq_id)      # 执行流转
        elif args.action == "new_intent":             # 标新意图
            q.label_new_intent(args.id)               # 执行流转
        else:                                        # 否则必是 reject
            q.reject(args.id)                         # 拒绝
        print("已更新。")                              # 提示完成
    elif args.cmd in ("iterate", "apply"):            # ── 聚类预览 / 落库 ──
        enc = _load_dense_for_cli()                   # 加载编码器（ONNX 优先）
        clu = IntentClustering(args.dist, args.min_size) if args.cmd == "iterate" \
            else IntentClustering()                   # iterate 用 CLI 阈值；apply 用默认值
        summary = clu.cluster(q.list("pending"), enc.encode)  # 对 pending 问题聚类
        print(f"共 {sum(s['size'] for s in summary)} 条待标注问题，"
              f"聚为 {len(summary)} 个意图簇：")        # 汇总规模与簇数
        for s in summary:                             # 逐簇打印
            mark = "★" if s["size"] >= 3 else " "     # ≥3 条的簇打星（值得建 FAQ）
            print(f"{mark} 簇#{s['cluster_id']} size={s['size']} avg_conf={s['avg_conf']} "
                  f"样例: {s['samples'][:2]}")        # 打印簇号/规模/平均置信/前两条样例
        if args.cmd == "apply":                       # apply 模式才真正写库
            created = ActiveLearning(queue=q).apply_clusters(summary)  # 达标簇生成 draft FAQ
            print(f"已生成 {len(created)} 条 draft FAQ（灰度，待审核后 promote）。")  # 提示结果
    elif args.cmd == "promote":                       # ── 灰度转正 ──
        done = ActiveLearning(queue=q).promote(args.ids)  # 批量转正
        print(f"灰度转正: {done}")                     # 打印成功的 id 列表


if __name__ == "__main__":
    main()                                           # 直跑才执行 CLI
