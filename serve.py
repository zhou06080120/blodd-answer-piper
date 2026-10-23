"""
血压计 FAQ 智能问答在线服务（需求 3/4 落地：量化加速 + 异步并发 + 低置信度拦截闭环）

核心能力：
  - POST /query            混合检索两阶段回答；置信度 < 阈值(0.82) 时自动拦截，
                           异步推入待标注队列（不阻塞请求，符合 Celery 式离线解耦思路）
  - GET  /health /stats    健康检查与可观测性（命中贡献、分阶段延迟、队列水位）
  - GET  /annotations      查看/流转待标注队列（attach / new_intent / reject）
  - GET  /iterate          聚类挖掘高频未知意图（只读预览）
  - POST /iterate/apply    达标簇 → draft FAQ（灰度条目，不参与在线检索）
  - POST /admin/promote    灰度转正（draft/gray → active）并重建索引
  - POST /admin/refresh    重载 KB + 重建混合检索索引

启动：
  python -m uvicorn serve:app --host 0.0.0.0 --port 8000
环境变量（可选）：
  BP_KB  BP_BERT_PATH  BP_THRESHOLD(默认0.82)  BP_TOP_K(默认3)
  BP_API_KEYS/BP_ADMIN_KEYS（双级 Key 鉴权，均不配=本地开发放行）BP_AUDIT_PATH（审计日志）

依赖：fastapi / uvicorn / torch / transformers / onnxruntime
"""
# ↑ 模块级 docstring：把问答能力包装成 HTTP 在线服务（FastAPI）。进程启动时经
#   lifespan 一次性完成模型/索引/队列初始化；运行期提供四组接口：在线问答、
#   可观测性、标注队列流转、周级迭代（聚类→draft→灰度转正）。所有阻塞调用
#   （检索/落盘/聚类）都经 asyncio.to_thread 丢入线程池执行，避免慢请求
#   卡死事件循环——与 Celery 异步解耦思路一致，只是实现在同进程线程池内。

import asyncio                                   # 异步事件循环、Lock 与 to_thread
import json                                       # KB/请求体的 JSON 处理
import logging                                   # 日志框架（setup_logging 之后的结构化打点用）
import os                                         # 读取环境变量（BP_KB/BP_THRESHOLD 等）
import sys                                        # 运行时模块路径注入
import time                                       # 端到端延迟计时（perf_counter）
from contextlib import asynccontextmanager        # 把 async 生成器改造成异步上下文管理器（lifespan 用）
from pathlib import Path                           # 路径对象

ROOT = Path(__file__).parent                       # 项目根目录
SRC = ROOT / "src"                                 # 源码目录
sys.path.insert(0, str(SRC))                      # 注入 src/ 到模块搜索路径最前

import torch  # noqa: E402                         # 深度学习框架（改路径后导入，豁免 E402）
from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402  # Web 框架、可控异常、路由依赖与请求对象（P4 管理守卫/审计注入）
from fastapi.responses import JSONResponse  # noqa: E402  # 400 校验响应体（detail 与 HTTPException 同构）
from pydantic import BaseModel  # noqa: E402        # 请求体模型（自动校验/序列化）

from active_learning import ActiveLearning, AnnotationQueue, IntentClustering  # noqa: E402  # 主动学习闭环三件套：迭代编排/标注队列/意图聚类
from audit import record_audit  # noqa: E402  # P4 审计留痕：管理操作 JSONL 追加写（BP_AUDIT_PATH 可覆盖路径）
from auth import install_auth, require_admin  # noqa: E402  # P4 双级 Key 鉴权：中间件安装入口 + require_admin 管理端点守卫
from compliance import MEDICAL_DISCLAIMER, ComplianceError, validate_query  # noqa: E402  # P3 医疗合规：统一免责声明 + 输入校验（检索服务不做注入拦截）
from hybrid_retriever import HybridRetriever  # noqa: E402   # 两阶段混合检索器
from observability import install_observability, setup_logging  # noqa: E402  # P2 可观测性：结构化日志初始化 + 最外层指标中间件 + /metrics 端点
from onnx_runtime import load_dense_encoder  # noqa: E402    # Dense 编码后端工厂（ONNX-INT8 优先自动降级）
from rate_limit import install_rate_limit  # noqa: E402  # 按 IP 滑动窗口限流（环境变量 RATE_LIMIT_RPM 控制，0=关闭）

setup_logging()                                   # 初始化结构化日志（JSON 行式；须在常量/组件打日志之前完成，BP_LOG_FORMAT=text 可切人读格式）
logger = logging.getLogger(__name__)              # serve 模块日志器：lifespan 启动信息与低置信入队告警统一经它输出

KB_DEFAULT = ROOT / "data" / "blood_pressure_expanded" / "faq_extended.json"  # 默认知识库（扩容版）
CKPT_PREF = [ROOT / "outputs" / "checkpoints" / "biencoder_hardneg_best.pt",
             ROOT / "outputs" / "checkpoints" / "biencoder_cosine_best.pt"]  # Dense 权重优先级列表：难负例微调版 > 基线版
CE_CKPT = ROOT / "outputs" / "checkpoints" / "crossencoder_best.pt"  # CrossEncoder 精排权重路径
ONNX_DIR = ROOT / "outputs" / "onnx"               # ONNX 量化模型目录
BERT_PATH = os.getenv("BP_BERT_PATH", r"F:\aaa123\pretrain_models\bert-base-chinese")  # 中文 BERT 基座目录（环境变量 BP_BERT_PATH 可覆盖，容器/其他机器部署用）

FALLBACK_MSG = ("抱歉，我暂时没有完全把握回答这个问题。建议您换个问法，或转接人工客服获取更准确的帮助。"
                "(本回答仅供健康教育参考，不能替代医生诊疗，如有不适请及时就医。")  # 低置信拦截兜底话术（内嵌医疗免责声明）

S = {}  # 全局服务状态（单进程部署）                    # 启动期填充：dense_enc/cross_score/queue/threshold/top_k/lock/lat/retriever/kb/kb_path


# ── 请求/响应模型 ────────────────────────────────────────────────────────────

# /query 的请求体：query 必填，top_k 默认 3（pydantic 自动校验类型）
class QueryIn(BaseModel):
    query: str                                       # 用户原始问句
    top_k: int = 3                                   # 返回候选条数（默认 3）


# /annotations/{rid} 的请求体：标注动作三选一
class LabelIn(BaseModel):
    action: str  # attach / new_intent / reject      # 动作：挂到既有 FAQ / 判为新意图 / 拒绝噪音
    faq_id: int | None = None                        # 仅 attach 需要指定目标 FAQ 编号


# /admin/promote 的请求体：批量灰度转正
class PromoteIn(BaseModel):
    faq_ids: list[int]                               # 待从 draft/gray 转为 active 的 FAQ 编号列表


# ── 组件构建 ─────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 函数：build_cross_score
# 功能：加载 CrossEncoder 权重，返回"批量打分闭包" fn(queries, texts)→正类概率列表
# 流水线位置：启动期（lifespan）调用一次，闭包注入 HybridRetriever 做第二阶段精排
# 输入：ckpt_path=CE 权重路径；bert_path=基座目录；device=设备
# 输出：打分函数 fn（0~1 概率列表）
# ═══════════════════════════════════════════════════════════════════════════
def build_cross_score(ckpt_path, bert_path, device):
    """CrossEncoder 精排打分函数：fn(query, texts) -> list[float]（0~1 概率）"""
    from model import build_crossencoder             # 延迟导入：按需加载，加快进程启动
    from transformers import BertTokenizer           # CE 句对分词器

    rckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)  # 加载 CE checkpoint（含权重+结构超参）
    model = build_crossencoder(
        bert_path=bert_path,
        num_hidden_layers=rckpt.get("args", {}).get("num_hidden_layers"),
    ).to(device)                                     # 按训练时裁剪层数重建交叉编码器并搬上设备
    model.load_state_dict(rckpt["state_dict"])       # 载入训练权重
    model.eval()                                     # 切推理模式（关 dropout）
    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 分词器

    @torch.no_grad()                                  # 打分闭包：无梯度推理
    def fn(queries, texts):                          # 签名对齐 HybridRetriever：两个等长列表进、概率列出
        enc = tokenizer(list(zip(queries, texts)), max_length=128, truncation=True,
                        padding="max_length", return_tensors="pt")  # 逐对 (query, text) 拼成句对 batch 编码
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                       enc["token_type_ids"].to(device))  # 前向 → [B, 2] 两类 logits
        return torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()  # softmax 取正类概率，转 Python 列表返回

    return fn                                         # 返回打分闭包


# ═══════════════════════════════════════════════════════════════════════════
# 函数：load_kb
# 功能：按环境变量 BP_KB（缺省 faq_extended.json）读取知识库
# 输入：无（读环境变量）　输出：(faqs 列表, 库文件路径)
# ═══════════════════════════════════════════════════════════════════════════
def load_kb():
    """按环境变量 BP_KB（默认 faq_extended.json）加载 FAQ 库，返回 (faqs, path)。"""
    path = Path(os.getenv("BP_KB", str(KB_DEFAULT)))  # 优先环境变量指定的库，否则用默认扩容库
    return json.loads(path.read_text(encoding="utf-8")), path  # 读文件反序列化，连同路径一起返回


# ═══════════════════════════════════════════════════════════════════════════
# 函数：rebuild_index
# 功能：（重）建混合检索索引——刷新 S 中的 KB 引用与 HybridRetriever；
#       draft 灰度条目由 HybridRetriever 内部过滤，不会进入在线候选
# 输入：faqs=可选的新 KB；缺省时从磁盘重新读取
# 输出：无（写全局状态 S）——供启动、转正、refresh 三处复用
# ═══════════════════════════════════════════════════════════════════════════
def rebuild_index(faqs=None):
    """（重）建混合检索索引；draft 条目由 HybridRetriever 内部过滤"""
    if faqs is None:                                  # 未传入 KB → 从磁盘重读最新版
        faqs, _ = load_kb()
    S["kb"] = faqs                                    # 更新全局 KB 引用
    S["retriever"] = HybridRetriever(                 # 重建两阶段检索器（重新统计 BM25 + 全库 Dense 编码）
        faqs,
        dense_encode=S["dense_enc"].encode,           # 注入启动期建好的 Dense 编码函数
        cross_score=S.get("cross_score"),             # 注入精排打分闭包（可能为 None → 降级组合分）
    )
    S["kb_path"] = Path(os.getenv("BP_KB", str(KB_DEFAULT)))  # 记录当前 KB 文件路径（_attach_variant 落盘要用）


# ═══════════════════════════════════════════════════════════════════════════
# 函数：lifespan（异步上下文管理器）
# 功能：FastAPI 生命周期钩子——yield 之前做一次性初始化（设备/编码后端/精排/
#       标注队列/阈值/锁/延迟采样/索引），yield 之后清理全局状态
# 输入：app 实例　输出：无（填充全局状态 S）
# ═══════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选设备
    ckpt = next((c for c in CKPT_PREF if c.exists()), CKPT_PREF[-1])  # 按优先级取第一个存在的 Dense 权重（都不存在则取最后一个路径，由加载器报错）
    logger.info(f"[serve] dense checkpoint: {ckpt.name}", extra={"checkpoint": str(ckpt.name)})  # 记录选中的权重文件名
    S["dense_enc"], meta = load_dense_encoder(        # 构建 Dense 编码器（ONNX-INT8→fp32→PyTorch 自动降级）
        ckpt, BERT_PATH, onnx_dir=ONNX_DIR,
        max_length=64, prefer="onnx" if ONNX_DIR.exists() else "torch")  # ONNX 目录存在则优先量化后端
    logger.info(f"[serve] dense 编码后端: {meta['backend']}-{meta.get('kind') or meta.get('device', '')}",
                extra={"backend": meta["backend"], "kind": meta.get("kind") or meta.get("device", "")})  # 记录实际后端（onnx-int8/onnx-fp32/torch）

    S["cross_score"] = None                           # 先假定无精排
    if CE_CKPT.exists():                              # 精排权重存在
        S["cross_score"] = build_cross_score(CE_CKPT, BERT_PATH, device)  # 构建打分闭包
        logger.info(f"[serve] CrossEncoder 精排已加载（{str(device)}）", extra={"device": str(device)})  # 记录精排就绪
    else:                                             # 无 CE 权重 → 降级
        logger.warning("[serve] 未找到 CrossEncoder，降级 RRF+余弦组合分",
                       extra={"crossencoder_ckpt": str(CE_CKPT)})  # 说明降级打分方式

    S["queue"] = AnnotationQueue()                    # 标注队列（低置信样本落盘 data/feedback/queue.jsonl）
    S["threshold"] = float(os.getenv("BP_THRESHOLD", "0.82"))  # 置信度阈值（环境变量可覆盖，默认 0.82）
    S["top_k"] = int(os.getenv("BP_TOP_K", "3"))     # 默认返回条数（环境变量可覆盖）
    S["lock"] = asyncio.Lock()                        # 异步锁：序列化 rebuild_index，防止并发重建索引
    S["lat"] = []  # 端到端延迟采样（最近 200 次）      # 延迟环形统计窗口（超出 200 条即裁剪）

    faqs, kb_path = load_kb()                         # 读取知识库
    rebuild_index(faqs)                               # 首次建立混合检索索引
    logger.info(f"[serve] KB: {len(faqs)} 条 FAQ, 索引 {S['retriever'].n} 个候选问法 | "
                f"conf 阈值 = {S['threshold']}",
                extra={"faq_count": len(faqs), "index_size": S["retriever"].n,
                       "threshold": S["threshold"]})  # 记录服务就绪概要
    yield                                             # ← 服务运行期：处理请求；直到进程关闭才继续往下
    S.clear()                                         # 关闭期：清空全局状态，释放引用


app = FastAPI(title="血压计 FAQ 智能问答服务", lifespan=lifespan)  # 创建应用实例并挂载生命周期钩子
install_auth(app)  # P4 鉴权中间件（最内层）：读 BP_API_KEYS/BP_ADMIN_KEYS，未配置不注册（本地开发放行）
install_rate_limit(app)  # 注册按 IP 滑动窗口限流中间件（RATE_LIMIT_RPM 默认 60/min，0=关闭；/health 探针豁免）
install_observability(app)  # 注册可观测中间件（最外层：X-Request-ID + Prometheus 指标 + /metrics 端点）。
# ↑ 注册顺序关键（FastAPI 后注册的 HTTP 中间件位于洋葱外层）：请求流向 observability →
#   rate_limit → auth → 路由。限流先于鉴权执行可防 Key 暴力枚举；鉴权 401/403 与限流
#   429 全部流经最外层中间件，被 http_requests_total{status=...} 与结构化日志统计到。


# ── P4 管理端点守卫（require_admin 的依赖注入适配层）────────────────────────

# auth.require_admin 的 request 形参无类型注解，直接作 Depends(require_admin) 会被
# FastAPI 解析成 query 参数（恒 422）；这里用带 Request 注解的同语义薄包装正确注入
# 请求对象后原样调用——拦截/放行逻辑全在 auth.require_admin 内，本层零业务。
async def require_admin_dep(request: Request) -> None:
    """管理端点路由依赖：鉴权启用时非 admin 抛 403；未启用放行（向后兼容）。"""
    require_admin(request)


# ── 在线问答 ─────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 路由：POST /query —— 在线问答主接口
# 功能：输入合规校验（空拒/超长拒/控制字符剥离）→ 两阶段混合检索回答；
#       低置信自动拦截 + 异步入队；统计端到端延迟；全响应附带医疗免责声明
# 输入：QueryIn（query 必填 + top_k）　输出：命中结果/兜底话术 + 候选列表 + 延迟 + 免责声明
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/query")
async def query(q: QueryIn):
    try:                                              # P3 输入合规校验：不通过直接 400，不进检索
        cleaned = validate_query(q.query)             # 统一清洗入口（检索服务无 LLM 生成，天然免疫注入，仅做输入校验）
    except ComplianceError as e:                      # 校验失败（空输入 / 超长）
        return JSONResponse(status_code=400, content={"detail": str(e)})  # 异常 message 作 detail（str(e) 即构造时传入的报错文案）
    t0 = time.perf_counter()                          # 记录起始时间戳（高精度）
    ret = await asyncio.to_thread(S["retriever"].search, cleaned,
                                  q.top_k, S["threshold"], S["threshold"])  # 检索丢线程池执行，避免阻塞事件循环
    elapsed_ms = (time.perf_counter() - t0) * 1000    # 计算端到端毫秒延迟
    S["lat"].append(elapsed_ms)                       # 记录本次延迟样本
    del S["lat"][:-200]                               # 只保留最近 200 个样本（内存控制）

    top = ret["results"][0] if ret["results"] else None  # 最优候选（可能无结果）
    if not ret["matched"]:                            # 整体置信度未过阈值 → 拦截
        # 低置信度拦截：异步推入待标注队列（落盘不阻塞响应）
        try:                                           # 入队失败不应影响主回答
            await asyncio.to_thread(                   # 落盘写 JSONL 属阻塞 IO → 丢线程池
                S["queue"].append, cleaned, ret["confidence"],
                [{"faq_id": r["faq_id"], "question": r["question"],
                  "confidence": r["confidence"]} for r in ret["results"]])  # 记录原句+置信度+Top 候选，供标注员参考
        except Exception as e:                         # 任何写入异常
            logger.warning(f"[serve] 队列写入失败（忽略）: {e}", extra={"error": str(e)})  # 仅记告警日志，不中断响应

    return {                                           # 组装 JSON 响应
        "query": cleaned,                              # 回显清洗后的问句
        "matched": ret["matched"],                    # 是否过置信度阈值
        "confidence": ret["confidence"],               # 融合后的综合置信度
        "threshold": S["threshold"],                   # 当前生效阈值
        "low_confidence": ret["low_confidence"],       # 低置信标记（与 matched 互补）
        "faq_id": top["faq_id"] if top and ret["matched"] else None,  # 命中才返回编号
        "question": top["question"] if top and ret["matched"] else None,  # 命中才返回标准问
        "answer": top["answer"] if top and ret["matched"] else FALLBACK_MSG,  # 未命中返回兜底话术
        "candidates": [{"faq_id": r["faq_id"], "question": r["question"],
                        "confidence": r["confidence"]} for r in ret["results"]],  # Top-k 候选明细
        "elapsed_ms": round(elapsed_ms, 2),            # 本次端到端延迟（毫秒）
        "disclaimer": MEDICAL_DISCLAIMER,              # P3 医疗免责声明：命中与低置信拦截全响应统一附带
    }


# ── 可观测性 ─────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 路由：GET /health —— 存活/就绪检查（给负载均衡与监控探针用）
# 输入：无　输出：后端类型、库/索引规模、当前阈值
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok", "dense_backend": getattr(S.get("dense_enc"), "backend", None),  # Dense 后端类型（onnx/torch）
            "kb_size": len(S.get("kb", [])), "index_size": S.get("retriever", None) and S["retriever"].n,  # KB 条数与索引问法数
            "threshold": S.get("threshold")}          # 当前置信度阈值


# ═══════════════════════════════════════════════════════════════════════════
# 路由：GET /stats —— 可观测性指标汇总
# 功能：检索器内部分阶段指标 + 端到端延迟（均值/P95）+ 队列水位
# 输入：无　输出：扁平化统计字典
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/stats")
def stats():
    ret = S.get("retriever")                          # 取检索器实例（可能未初始化）
    base = ret.stats() if ret else {}                 # 检索器自带统计（命中贡献、分阶段延迟等）
    lat = S.get("lat", [])                            # 延迟样本窗口
    base["e2e_latency_ms"] = {                        # 叠加端到端延迟统计
        "recent": len(lat),                           # 当前样本数
        "mean": round(sum(lat) / len(lat), 2) if lat else None,  # 均值（空窗口返回 None）
        "p95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 2) if len(lat) >= 20 else None,  # P95（样本≥20 才有意义）
    }
    base["queue"] = S["queue"].stats() if S.get("queue") else {}  # 待标注队列水位统计
    return base                                       # 返回全部统计


# ── 标注队列（主动学习）──────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 路由：GET /annotations —— 查看待标注队列
# 输入：status=按状态过滤（可选）；limit=最多返回条数　输出：{total, items}
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/annotations")
def annotations(status: str | None = None, limit: int = 50):
    rows = S["queue"].list(status, limit)             # 按状态/条数过滤队列记录
    return {"total": len(rows), "items": rows}        # 返回条数与明细


# ═══════════════════════════════════════════════════════════════════════════
# 路由：POST /annotations/{rid} —— 标注流转（管理写操作）
# 功能：把一条待标注记录按 action 处理（attach 并入既有 FAQ / new_intent 判新 /
#       reject 拒绝）；attach 成功后自动把问法并回 KB（知识沉淀）
# 输入：rid=记录 id；LabelIn（action + 可选 faq_id）　输出：{ok, ...}
# 安全：P4——require_admin 守卫（非管理 Key 403）+ record_audit 审计留痕
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/annotations/{rid}", dependencies=[Depends(require_admin_dep)])
async def label(rid: str, body: LabelIn, request: Request):
    if body.action == "attach":                       # 动作一：挂到既有 FAQ
        if body.faq_id is None:                        # attach 必须指明目标 FAQ
            raise HTTPException(400, "attach 需要 faq_id")  # 缺参返回 400
        await asyncio.to_thread(S["queue"].label_attach, rid, body.faq_id)  # 线程池执行落盘状态流转
    elif body.action == "new_intent":                  # 动作二：判定为新意图（等聚类建 draft）
        await asyncio.to_thread(S["queue"].label_new_intent, rid)  # 线程池执行
    elif body.action == "reject":                     # 动作三：拒绝（噪音/无效问句）
        await asyncio.to_thread(S["queue"].reject, rid)  # 线程池执行
    else:                                             # 未知动作
        raise HTTPException(400, f"未知 action: {body.action}")  # 返回 400
    # attach 之后把该问法并入对应 FAQ 的 variants（自动知识沉淀）
    extra = {}                                        # 附加返回信息
    if body.action == "attach":                       # 仅 attach 需要并库
        rows = S["queue"].list()                      # 取回最新队列（找到该记录原文）
        r = next((x for x in rows if x["id"] == rid), None)  # 定位刚处理的那条记录
        if r:                                          # 找到了
            extra = _attach_variant(r["query"], body.faq_id)  # 把原问法并入目标 FAQ 的 variants
    record_audit("annotate", target=rid,              # P4 审计：谁在何时对哪条队列记录做了何种流转
                 detail={"action": body.action, "faq_id": body.faq_id},
                 actor=getattr(request.state, "actor_id", "-"))  # 操作者指纹（未鉴权时 "-"）
    return {"ok": True, **extra}                      # 返回成功 + 并库结果


# ═══════════════════════════════════════════════════════════════════════════
# 函数：_attach_variant
# 功能：把已标注问法并回知识库——追加到目标 FAQ 的 variants 并立即落盘
#       （active 条目在下次索引刷新后即可被检索命中）
# 输入：query=标注原句；faq_id=目标编号　输出：{merged: True/False, faq_id?}
# ═══════════════════════════════════════════════════════════════════════════
def _attach_variant(query, faq_id):
    """把已标注问法并回 KB（active FAQ 直接生效，索引刷新后检索可命中）"""
    for f in S["kb"]:                                  # 遍历内存中的 KB
        if f["id"] == int(faq_id):                     # 找到目标 FAQ
            if query not in f["variants"] and query != f["question"]:  # 去重：不与已有改写/标准问重复
                f["variants"].append(query)            # 追加为新同义改写
                S["kb_path"].write_text(
                    json.dumps(S["kb"], ensure_ascii=False, indent=2), encoding="utf-8")  # 立即落盘整个 KB（UTF-8、缩进 2）
                return {"merged": True, "faq_id": f["id"]}  # 返回并库成功
    return {"merged": False}                           # 未找到目标 FAQ → 并库失败


# ── 周级迭代：聚类 → draft → 灰度转正 ────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# 路由：GET /iterate —— 聚类挖掘高频未知意图（只读预览，不落库）
# 功能：把 pending 队列做层次聚类，返回簇摘要与"达到建 draft 标准（size≥3）"的簇数
# 输入：无　输出：{pending, clusters, qualify_draft}
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/iterate")
async def iterate_preview():
    pending = S["queue"].list("pending")               # 取全部待标注记录
    if not pending:                                    # 队列为空
        return {"clusters": [], "note": "队列为空"}   # 无事可做
    clu = IntentClustering()                           # 聚类器实例
    summary = await asyncio.to_thread(clu.cluster, pending, S["dense_enc"].encode)  # 编码+聚类丢线程池（耗时操作）
    reached = [c for c in summary if c["size"] >= 3]   # 只统计 size≥3 的达标簇（高频意图）
    return {"pending": len(pending), "clusters": summary,
            "qualify_draft": len(reached)}             # 返回队列规模、簇摘要、达标簇数


# ═══════════════════════════════════════════════════════════════════════════
# 路由：POST /iterate/apply —— 达标簇落地为 draft FAQ（灰度）（管理写操作）
# 功能：聚类后把达标簇建成 draft 状态的新 FAQ（不参与在线检索），待人工审核
# 输入：无　输出：新建 draft 列表 + 审核提示
# 安全：P4——require_admin 守卫（非管理 Key 403）+ record_audit 审计"触发"动作
#       （长任务只记触发，不追踪任务完成）
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/iterate/apply", dependencies=[Depends(require_admin_dep)])
async def iterate_apply(request: Request):
    pending = S["queue"].list("pending")               # 取待标注记录
    clu = IntentClustering()                           # 聚类器
    summary = await asyncio.to_thread(clu.cluster, pending, S["dense_enc"].encode)  # 线程池执行聚类
    learner = ActiveLearning(kb_path=S["kb_path"], queue=S["queue"])  # 迭代编排器（KB+队列）
    created = await asyncio.to_thread(learner.apply_clusters, summary)  # 达标簇 → draft FAQ 落库
    record_audit("iterate_apply",                     # P4 审计：记录"触发落地"这个动作本身
                 target=",".join(str(f["id"]) for f in created) or "-",  # 新建 draft 的编号列表
                 detail={"pending": len(pending), "created_drafts": [f["id"] for f in created]},
                 actor=getattr(request.state, "actor_id", "-"))  # 操作者指纹（未鉴权时 "-"）
    return {"created_drafts": [{"id": f["id"], "question": f["question"]}
                               for f in created],      # 返回新建 draft 的编号与标准问
            "note": "draft 不参与在线检索，审核后调用 /admin/promote 灰度转正"}  # 提示后续流程


# ═══════════════════════════════════════════════════════════════════════════
# 路由：POST /admin/promote —— 灰度转正（draft/gray → active）并重建索引（管理写操作）
# 输入：PromoteIn（faq_ids 列表）　输出：转正明细 + 新索引规模
# 安全：P4——require_admin 守卫（非管理 Key 403）+ record_audit 审计留痕
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/admin/promote", dependencies=[Depends(require_admin_dep)])
async def promote(body: PromoteIn, request: Request):
    learner = ActiveLearning(kb_path=S["kb_path"], queue=S["queue"])  # 迭代编排器
    done = await asyncio.to_thread(learner.promote, body.faq_ids)  # 执行状态流转（落盘）
    async with S["lock"]:                              # 持锁重建索引，避免并发刷新冲突
        await asyncio.to_thread(rebuild_index)        # 线程池执行重建（重读 KB + 重新编码）
    record_audit("promote_gray",                      # P4 审计：灰度转正了哪些 FAQ
                 target=",".join(map(str, body.faq_ids)),  # 操作对象 id（FAQ 编号列表）
                 detail={"faq_ids": body.faq_ids, "promoted": done, "index_size": S["retriever"].n},
                 actor=getattr(request.state, "actor_id", "-"))  # 操作者指纹（未鉴权时 "-"）
    return {"promoted": done, "index_rebuilt": True, "index_size": S["retriever"].n}  # 返回转正结果与新索引规模


# ═══════════════════════════════════════════════════════════════════════════
# 路由：POST /admin/refresh —— 手动重载 KB + 重建索引（KB 被外部编辑后调用）（管理写操作）
# 输入：无　输出：{ok, index_size, kb_size}
# 安全：P4——require_admin 守卫（非管理 Key 403）+ record_audit 审计留痕
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/admin/refresh", dependencies=[Depends(require_admin_dep)])
async def refresh(request: Request):
    async with S["lock"]:                              # 持锁防止与 promote 并发重建
        await asyncio.to_thread(rebuild_index)        # 线程池执行重读+重建
    record_audit("admin_refresh",                     # P4 审计：手动刷新索引（KB 被外部编辑后）
                 target=str(S.get("kb_path", "")),    # 操作对象：当前生效的 KB 文件路径
                 detail={"kb_size": len(S["kb"]), "index_size": S["retriever"].n},
                 actor=getattr(request.state, "actor_id", "-"))  # 操作者指纹（未鉴权时 "-"）
    return {"ok": True, "index_size": S["retriever"].n,
            "kb_size": len(S["kb"])}                  # 返回新索引与 KB 规模


if __name__ == "__main__":                             # 直接运行本文件时
    import uvicorn                                     # 导入 ASGI 服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)       # 监听所有网卡的 8000 端口启动服务
