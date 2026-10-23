"""
离线冒烟测试（不依赖网络 / Key / GPU 的部分）

覆盖：
  1. llm_synth      —— 离线模板改写 / 边界负例 / 去重（无 Key 可跑）
  2. hybrid_retriever —— BM25 分词 / 双路召回 / RRF / 无 CE 降级 / 高低置信区分
  3. active_learning —— 标注队列流转 / 意图聚类 / draft 生成 / 灰度转正
  4. onnx_runtime   —— OnnxEncoder 缺失时工厂函数回退 PyTorch（不崩溃）

使用方式：python smoke_test.py
"""
# ↑ 模块级 docstring：任何环境都能跑的核心链路自检（合成→检索→闭环→回退）

import json                                       # 写临时 KB 文件
import sys                                        # 模块路径注入与退出码
import tempfile                                   # 临时目录（测试主动学习不污染真实数据）
import warnings                                   # 告警过滤
from pathlib import Path                          # 路径处理

ROOT = Path(__file__).parent                      # 项目根目录
sys.path.insert(0, str(ROOT / "src"))             # 注入 src/ 到模块路径
warnings.filterwarnings("ignore")                  # 屏蔽无关告警保持输出干净

results = []                                      # 全部断言 (名称, 通过与否) 的收集器


def check(name, cond, detail=""):
    """记录一次断言结果并打印；最终由 main 汇总失败项决定进程退出码。"""
    results.append((name, bool(cond)))             # 记录（cond 强转布尔）
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  [{detail}]" if detail else ""))  # 打印带符号的单行结果


def main():
    """依次冒烟四个子模块（合成/检索/主动学习/ONNX 回退），任一失败则非零退出。"""
    print("== 1. llm_synth 离线模板合成 ==")          # 第一组：数据合成离线路径
    from llm_synth import OfflineTemplateSynth, dedupe, normalize  # 导被测组件
    faq = {"id": 1, "question": "血压计怎么正确测量血压？",  # 测试 FAQ：标准问
           "variants": ["如何用血压计测血压？"], "answer": "静坐5分钟再测。"}  # 变体与答案
    s = OfflineTemplateSynth(seed=42)               # 固定种子的离线合成器
    paras = s.generate_paraphrases(faq, need=12)     # 生成 12 条改写
    negs = s.generate_negatives(faq, need=4)         # 生成 4 条边界负例
    check("生成 12 条改写候选", len(paras) >= 12)      # 断言改写数量达标
    check("改写均不与已有问法重复",
          all(normalize(p) != normalize(faq["question"]) for p in paras))  # 断言归一化后无一与标准问相同
    check("边界负例非空且含意图替换", len(negs) >= 2 and any(
        "校准" in n or "清洗" in n or "选购" in n for n in negs), str(negs[:2]))  # 断言负例真的换了意图词
    check("dedupe 去重有效", len(dedupe(["血压准吗？", "血压准吗", "血压准吗？"])) == 1)  # 断言标点差异被归一化去重

    print("== 2. hybrid_retriever 混合检索 ==")      # 第二组：混合检索链路
    import numpy as np                              # 构造假向量用
    from hybrid_retriever import HybridRetriever, tokenize  # 导被测组件
    toks = tokenize("HEM-7156 血压计袖带范围")        # 测分词：型号+中文混合句
    check("型号整词保留", "hem-7156" in toks, str(toks[:4]))  # 断言型号没被拆散（小写整词在列表中）
    import numpy.random as npr                      # numpy 随机模块
    npr.seed(0)                                    # 固定 numpy 种子（防随机数影响测试）

    faqs = [                                       # 造三条不同状态的 FAQ
        {"id": 1, "question": "血压计怎么校准？", "answer": "A1",
         "variants": ["校准血压计的方法"], "status": "active"},  # 正式条目
        {"id": 2, "question": "血压计电池能用多久？", "answer": "A2",
         "variants": ["电池多久换一次"], "status": "gray"},    # 灰度条目
        {"id": 3, "question": "草稿问题（不可检索）？", "answer": "D",
         "variants": [], "status": "draft"},                   # 草稿条目（不应入索引）
    ]

    def fake_dense(texts):                          # 假 Dense 编码器：按关键词给 one-hot 向量
        out = []                                    # 收集向量
        for t in texts:                             # 逐条编码
            v = np.zeros(8, dtype="float32")         # 8 维零向量
            if "天气" in t:
                pass  # 离题查询 → 零向量，与全部 KB 文本余弦 0（领域闸门触发拦截）
            elif "校准" in t:
                v[0] = 1.0                          # 校准语义 → 第 0 维激活
            elif "电池" in t:
                v[1] = 1.0                          # 电池语义 → 第 1 维激活
            else:
                v[2] = 1.0                          # 其他 → 第 2 维激活
            out.append(v / (np.linalg.norm(v) + 1e-9))  # 归一化（分母加小量防除零；零向量保持全 0）
        return np.array(out, dtype="float32")      # 堆叠返回 [N, 8]

    # CE 打分：问 A 答校准类给高分
    def fake_ce(query, texts):                      # 假 CrossEncoder：与关键词匹配时给高分
        return [0.95 if ("校准" in t and "校准" in query) else 0.1 for t in texts]  # 含"校准"的候选且查询也是校准 → 0.95，否则 0.1

    ret = HybridRetriever(faqs, dense_encode=fake_dense, cross_score=fake_ce)  # 建检索器（索引 4 条：FAQ1 两条 + FAQ2 两条）
    check("draft 不入索引", ret.n == 4, f"n={ret.n}")  # 断言草稿被过滤（4 = 1×2 + 1×2）
    out = ret.search("怎么校准血压计？", top_k=1, threshold=0.82)  # 查询校准问题
    check("CE 精排命中校准 FAQ", out["results"][0]["faq_id"] == 1,
          f"conf={out['confidence']}")               # 断言首位是 FAQ1（高置信）
    out2 = ret.search("今天天气怎么样", top_k=1, threshold=0.82)  # 查询领域外问题
    check("低置信拦截生效（离题问法）", out2["low_confidence"] is True,
          f"conf={out2['confidence']}")               # 断言被判低置信（服务层会拒答/兜底）
    st = ret.stats()                                # 取运行统计
    check("统计字段完整", "hit_distribution" in st and "index_size" in st)  # 断言关键字段存在

    print("== 3. active_learning 主动学习闭环 ==")   # 第三组：数据闭环链路
    from active_learning import (ActiveLearning, AnnotationQueue,
                                 IntentClustering)  # 导被测组件
    with tempfile.TemporaryDirectory() as td:        # 临时目录：with 结束自动清理
        td = Path(td)                                # 转 Path 便于拼接
        q = AnnotationQueue(td / "fb")               # 队列实例（指向临时目录）
        q.append("我的血压计开不了机怎么办", 0.32, [])  # 入队 1：开机类问题
        q.append("血压计不通电咋整", 0.30, [])         # 入队 2：开机类
        q.append("血压计没反应", 0.35, [])             # 入队 3：开机类（前三条同主题）
        q.append("仪器没反应是不是坏了", 0.28, [])     # 入队 4：近似开机类
        q.append("袖带漏气怎么修", 0.40, [])           # 入队 5：袖带类（不同主题）
        check("队列追加 5 条", len(q.list()) == 5)     # 断言追加成功
        q.label_attach(q.list()[0]["id"], 20)        # 把第一条挂到 FAQ 20
        check("attach 流转", q.list("attached")[0]["faq_id"] == 20)  # 断言状态与 faq_id 均正确

        pend = q.list("pending")                     # 剩余 4 条待处理
        # 简单向量：开机类聚一起（hash 编码即可体现相似）
        def dummy_encode(texts):                     # 假编码器：按关键词给 one-hot
            base = np.zeros((len(texts), 4), dtype="float32")  # [N,4] 零矩阵
            for i, t in enumerate(texts):            # 逐行填
                base[i, 0 if any(k in t for k in ("开不了机", "不通电", "没反应")) else 1] = 1  # 开机类第 0 维，否则第 1 维
            return base                              # 返回

        clu = IntentClustering(dist_threshold=0.3, min_size=3)  # 聚类器（阈值小、最少 3 条成簇）
        summary = clu.cluster(pend, dummy_encode)    # 对 pending 聚类
        check("聚类识别高频意图簇", max(s["size"] for s in summary) >= 3,
              str([(s["size"], s["samples"][0]) for s in summary]))  # 断言出现 ≥3 条的簇（开机类）

        kb = td / "faq_extended.json"                # 临时 KB 路径
        kb.write_text(json.dumps([
            {"id": 1, "question": "血压计怎么校准？", "answer": "A",
             "variants": [], "status": "active"}], ensure_ascii=False), encoding="utf-8")  # 预置一条 active FAQ
        learner = ActiveLearning(kb_path=kb, feedback_dir=td / "fb")  # 迭代器实例（用临时 KB/队列）
        created = learner.apply_clusters(summary)    # 达标簇写 draft
        check("达标簇生成 draft FAQ", len(created) >= 1 and created[0]["status"] == "draft")  # 断言生成了 draft 态条目
        learner.promote([created[0]["id"]]) if created else None  # 有产物则转正（防御空列表短路）
        kb2 = learner.load_kb()                      # 重读 KB 验证
        check("promote 灰度转正", any(f["status"] == "active" and f["id"] > 1 for f in kb2))  # 断言新条目已是 active

    print("== 4. onnx_runtime 回退策略 ==")           # 第四组：部署降级链路
    from onnx_runtime import load_dense_encoder      # 双后端工厂
    ckpt = ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt"  # 用 cosine 权重测试
    import torch                                     # 指定 CPU 设备
    enc, meta = load_dense_encoder(ckpt, r"F:\aaa123\pretrain_models\bert-base-chinese",
                                   onnx_dir=td / "missing_onnx", device=torch.device("cpu"),
                                   prefer="onnx")    # 指向不存在的 onnx 目录 → 应自动降级
    check("ONNX 缺失时回退 PyTorch 不崩溃", meta["backend"] == "pytorch",
          meta["backend"])                            # 断言后端确实是 pytorch（未抛异常即通过）

    print("\n==== 冒烟结果 ====")                      # 汇总标题
    failed = [n for n, ok in results if not ok]       # 收集失败项名称
    print(f"通过 {len(results) - len(failed)}/{len(results)}")  # 打印通过率
    if failed:                                       # 有失败
        print("失败项:", failed)                       # 列出失败断言
        sys.exit(1)                                   # 非零退出码（CI 可感知失败）
    print("全部通过 ✓")                                # 全过提示


if __name__ == "__main__":
    main()                                           # 直跑才执行冒烟测试
