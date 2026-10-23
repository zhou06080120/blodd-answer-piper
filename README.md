# 血压计 FAQ 语义检索引擎

[![CI](https://github.com/zhou06080120/blodd-answer-piper/actions/workflows/ci.yml/badge.svg)](https://github.com/zhou06080120/blodd-answer-piper/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-77%25-brightgreen)](https://github.com/zhou06080120/blodd-answer-piper/actions)

面向医疗器械（血压计）场景的 FAQ 智能问答引擎：Sentence-BERT 双塔召回（难负例对比训练）+ BM25/Dense/RRF 混合检索 + CrossEncoder 精排的两阶段架构，ONNX INT8 量化 CPU 部署，低置信拦截 + 意图聚类 + 灰度转正的主动学习数据闭环，通过 FastAPI 提供在线服务。

```
                ┌─────────────── 在线链路 ───────────────┐
用户提问 → BM25 ─┐                                        │
                ├→ RRF 融合 → CE 精排 → 双闸门置信度 ──┼─ ≥0.82 → 返回标准答案
Dense(INT8) ────┘                                        └─ <0.82 → 拦截入队 + 兜底话术
                └─────────────── 数据闭环 ───────────────┘
标注队列 → 意图聚类 → 达标簇 → draft FAQ（灰度）→ promote 转正 → 重建索引
```

## 核心特性

| 能力 | 说明 |
|---|---|
| **双塔难负例训练** | Sentence-BERT 双编码器（bert-base-chinese 截取 4 层，45.6M 参数）在 4970 条扩容句对（含难负例与闲聊负例）上对比训练，正例平均相似度 0.956→0.993，决策边界更陡 |
| **两阶段混合检索** | BM25 稀疏（HEM-7156 等型号/错误码整词保真）+ Dense 语义双路召回 → RRF 融合 → CrossEncoder 精排；CE×Dense 双闸门置信度，未过阈值拒答并转人工话术 |
| **ONNX INT8 量化** | 静态图导出 + 动态 INT8 量化，CPU 单句编码 p50 34.7ms→16.5ms（**2.28×，延迟 -56.2%**）；onnx-int8→onnx-fp32→pytorch 三级自动降级，任一层失败不崩溃 |
| **主动学习闭环** | 低置信（<0.82）自动拦截入标注队列 → 层次聚类挖掘高频未知意图 → 达标簇生成 draft FAQ → 人工转正重建索引，数据飞轮持续吸收线上未知问法 |
| **灰度发布** | FAQ 三态流转 draft/gray/active：draft 不入在线索引、gray 可检索可观测、active 正式生效；转正操作持异步锁重建索引，防并发写坏 |
| **可观测性** | /health /stats 暴露命中贡献分布、分阶段延迟、端到端 P50/P95、低置信率、队列水位，线上状态一屏可见 |

## 实测数据（全部来自 outputs/logs 训练与评测日志，可复现）

### 检索质量（216 查询 × 1075 候选问法，eval_retrieval.py）

| 配置 | Top1 | Top3 | 平均置信度 |
|---|---|---|---|
| Dense 基线（cosine 双塔） | 93.52% | 97.69% | 0.956 |
| + 难负例训练 | 93.98% | 97.69% | 0.993 |
| + 混合检索 + CE 精排（线上配置） | **94.91%** | **98.61%** | 0.969（低置信拦截率 4.17%） |

命中贡献：双路共同命中 88.9%、Dense 独占 11.1% —— BM25 主要在型号/术语类问法上与 Dense 形成互补。分阶段延迟：Dense 20.3ms / BM25 9.1ms / CE 精排 31.0ms。

### 编码延迟与量化收益（单句含 tokenize，bench_latency.py）

| 后端 | p50 | p95 |
|---|---|---|
| PyTorch CPU fp32 | 34.7ms | 37.7ms |
| ONNX fp32 | 24.9ms | 26.9ms |
| **ONNX INT8（线上默认）** | **16.5ms** | **17.8ms** |
| PyTorch GPU（参考） | 8.1ms | 10.6ms |

### 数据合成规模（synthesize_data.py）

30 条种子 FAQ → LLM（DeepSeek）+ 离线模板双通道合成 **1045 条问法变体**（平均 34.8 条/FAQ，全库 1075 候选）→ 按FAQ 维度 8:1:1 切分生成 4970 训练句对 / 606 验证 / 626 测试（同条 FAQ 不跨 split，防数据泄漏），另产出 20360 条 CE 句对与 5630 条闲聊负例。

> 基线对比实验（432 句对小数据集）：BiEncoder+CosineEmbeddingLoss F1=98.16% 显著优于 CrossEncoder F1=53.33%（小数据严重欠拟合）—— 验证了"双塔召回 + CE 仅做精排"的级联架构选择。

## 快速开始

### 安装

```bash
# 基础依赖（混合检索/数据合成/主动学习 + FastAPI）
pip install -e .

# CPU torch（serve.py 模块级依赖，约 200MB，避免 CUDA 版 2GB+）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 开发工具链（pytest / ruff / httpx）
pip install -e ".[dev]"

# 或直接装训练全家桶（torch / transformers / matplotlib / datasets）
pip install -e ".[train]"
```

### 配置

```bash
cp .env.example .env
# 可选：DEEPSEEK_API_KEY（数据合成用，不配置自动降级离线模板合成）
# BP_BERT_PATH 指向本地 bert-base-chinese 目录
# RATE_LIMIT_RPM 请求限流：每分钟每 IP 上限（默认 60，设 0 关闭）
```

请求限流为内置中间件（`src/rate_limit.py`，纯标准库滑动窗口）：默认每 IP 每分钟 60 次请求，超限返回 429 + `Retry-After`；`/health` 探针豁免。

### 启动服务

```bash
python -m uvicorn serve:app --host 0.0.0.0 --port 8000
```

前提：本地具备 bert-base-chinese 基座与训练产物（`outputs/checkpoints/` 或 `outputs/onnx/`）；无 ONNX 目录时自动降级 PyTorch 后端。

### 复现全流程

```bash
python synthesize_data.py               # ① 数据合成：扩容 KB + 训练句对（--no-llm 强制离线）
python train_sbert_hardneg.py           # ② 双塔难负例训练
python prepare_ce_data.py               # ③ CE 精排数据构建
python src/train_crossencoder.py        # ④ CrossEncoder 训练
python export_onnx.py --precision int8  # ⑤ ONNX 导出 + INT8 动态量化
python bench_latency.py                 # ⑥ 延迟基准（torch vs onnx 三档对比）
python eval_retrieval.py                # ⑦ 检索质量评测（三配置 Top1/Top3）
python smoke_test.py                    # ⑧ 零依赖冒烟自检（合成→检索→闭环→降级）
```

## Docker

镜像自带代码与 FAQ 知识库，模型即数据不入镜像，运行时挂载：

```bash
docker build -t blodd-answer-piper .
docker run -p 8000:8000 \
  -v /path/to/outputs:/app/outputs \
  -v /path/to/bert-base-chinese:/models/bert-base-chinese \
  blodd-answer-piper
```

## 测试与质量

```bash
pytest          # 81 用例，覆盖率 77%（门槛 30%，内置于 pyproject）
ruff check .    # lint
```

- 测试全程零外部依赖：FakeDenseEncoder 字符袋假编码器（L2 归一化）替代真实 BERT，无需模型权重 / API Key / GPU，本地与 CI 行为完全一致
- 单测驱动修复 3 处真实缺陷：exclude_texts 屏蔽项经 Dense 名次泄漏进融合候选尾部、嵌入 JSON 抽取正则缺捕获组导致兜底分支死代码、checkpoint 内记录的训练机路径覆盖 BP_BERT_PATH 环境变量（容器部署失效）

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query` | 在线问答；低置信自动拦截入队并返回兜底话术 |
| GET | `/health` | 健康检查（编码后端 / 索引规模） |
| GET | `/stats` | 可观测性（命中贡献 / 分阶段延迟 / P95 / 队列水位） |
| GET | `/annotations` | 待标注队列（低置信样本明细） |
| POST | `/annotations/{rid}` | 标注流转：attach 挂靠 FAQ / new_intent / reject |
| GET | `/iterate` | 聚类预览：高频未知意图簇（只读） |
| POST | `/iterate/apply` | 达标簇 → draft FAQ（灰度，不入在线索引） |
| POST | `/admin/promote` | draft/gray → active 并持锁重建索引 |
| POST | `/admin/refresh` | 重载 KB + 重建混合检索索引 |

## 项目结构

```
├── serve.py                    # FastAPI 在线服务（lifespan 加载模型 + 四组接口）
├── src/
│   ├── hybrid_retriever.py     # BM25+Dense 双路召回 → RRF → CE 精排 → 双闸门置信度
│   ├── active_learning.py      # 标注队列 / 意图聚类 / draft 生成 / 灰度转正
│   ├── onnx_runtime.py         # ONNX 导出 + INT8 量化 + 双后端编码器工厂
│   ├── llm_synth.py            # LLM / 离线模板双通道数据合成
│   ├── model.py                # BiEncoder / CrossEncoder 定义
│   ├── dataset.py              # Pair / Triplet / CE 三种 Dataset
│   ├── train_biencoder.py      # 基线训练（cosine / triplet 对比实验）
│   ├── train_crossencoder.py   # CE 精排训练
│   ├── evaluate.py             # 阈值搜索 + Acc / F1 / AUC
│   ├── analyze_badcases.py     # FP/FN 分类 + 置信度分级 + 可视化
│   └── compare_methods.py      # 三方案横向对比
├── train_sbert_hardneg.py      # 双塔难负例训练（线上主力权重）
├── synthesize_data.py          # 数据合成管线（扩容 KB + 句对切分）
├── prepare_ce_data.py          # CE 精排训练数据构建
├── export_onnx.py              # ONNX 导出 + INT8 动态量化
├── bench_latency.py            # 编码延迟基准
├── eval_retrieval.py           # 检索质量评测
├── faq_answer.py               # 命令行交互问答（--no-hybrid / --no-onnx 教学对比）
├── smoke_test.py               # 零依赖冒烟自检
├── tests/                      # 81 用例，7 个测试模块
└── data/                       # 基线数据集（432 句对）+ 扩容数据集（4970 句对）
```

## License

MIT
