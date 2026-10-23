# ═══════════════════════════════════════════════════════════════════
# 多阶段构建：builder 装依赖 → runtime 只带运行时产物
#
# 本镜像为"自带代码、外挂模型"形态：
#   - 镜像内：serve.py + src/ + data/（FAQ 知识库）+ CPU 推理依赖
#   - 运行时挂载：outputs/（Dense/CE 权重或 ONNX 量化模型）
#                 与 BERT 基座目录（BP_BERT_PATH 指向挂载点）
#   - 模型即数据不入镜像：无挂载时 lifespan 加载 checkpoint 失败，
#     容器直接退出（fail-fast，避免带病上线）
#
# 构建：  docker build -t blodd-answer-piper .
# 运行：  docker run -p 8000:8000 \
#           -v /path/to/outputs:/app/outputs \
#           -v /path/to/bert-base-chinese:/models/bert-base-chinese \
#           blodd-answer-piper
# ═══════════════════════════════════════════════════════════════════

# ── 阶段 1：安装依赖（依赖层缓存：仅 pyproject/README 变化时重建） ───────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
# 虚包安装：只落地 dependencies 声明的第三方包，不打包应用代码
RUN pip install --no-cache-dir --target=/deps .

# 在线服务最小模型栈：
#   - CPU 版 torch（约 200MB，避免 CUDA 版 2GB+；serve.py 模块级 import torch）
#   - transformers（lifespan 加载分词器 / CE 精排权重需要）
RUN pip install --no-cache-dir --target=/deps torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --target=/deps "transformers>=4.38.0"

# ── 阶段 2：运行时镜像（不含 pip 缓存与构建工具，体积最小化） ─────────────────
FROM python:3.12-slim

# 安全基线：非 root 运行
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# 依赖（来自 builder 的 site-packages）
COPY --from=builder /deps /deps

# 应用代码 + FAQ 知识库（serve.py 自行把 src/ 注入 sys.path）
COPY serve.py ./
COPY src ./src
COPY data ./data

USER appuser

ENV PYTHONPATH=/deps \
    PYTHONUNBUFFERED=1 \
    BP_BERT_PATH=/models/bert-base-chinese

EXPOSE 8000

# 轻量健康检查：slim 镜像无 curl，用标准库探活 /health
# start-period 放宽到 60s：冷启动需加载 ONNX 会话 + 全库 Dense 编码建索引
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
