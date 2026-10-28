# 公网部署手册（blodd-answer-piper）

> Render / UptimeRobot 通用步骤见姊妹项目
> [bp-monitor-service-agent/deploy/DEPLOY.md](../bp_monitor_service_agent/deploy/DEPLOY.md)，
> 本手册只写本项目差异。本项目为**纯本地模型服务（无 API Key）**，
> 部署核心是把模型带进镜像（免费实例无持久卷，无法沿用外挂挂载的生产形态）。

## 免费额度策略

同 blood-pressure-agent：**不配心跳、按需唤醒**（Render 免费 750 实例小时/月只够主 Demo 常驻）。
本项目冷启动最重（BERT 加载 + 全库 Dense 编码建索引，start-period 90s），唤醒后首个请求可能再等 30-60 秒——README 已如实标注。

## 一、模型准备（部署前必做：上传 HF Hub）

生产形态的 `outputs/` 被 .gitignore 忽略，且实测产物（biencoder checkpoint 173.85MB）超 GitHub 单文件 100MB 硬限制——**训练产物统一托管到 Hugging Face Hub**，构建期自动拉取。

最小部署产物集（实测体积，总计约 391MB）：

| 文件 | 体积 | 必需性 |
|---|---|---|
| `outputs/checkpoints/biencoder_hardneg_best.pt` | 173.9MB | **必需**：serve.py 启动时 `torch.load` 读训练超参（bert_path / pool / max_length），绕不开 |
| `outputs/onnx/biencoder_hardneg_best_int8.onnx` | 43.1MB | **必需**：实际推理后端（ONNX-INT8，16.5ms/次口径） |
| `outputs/checkpoints/crossencoder_best.pt` | 173.9MB | 可选：缺失自动降级 RRF+余弦组合分（线上检索效果略降，评测口径以仓库 CI 为准） |

### 上传步骤

```powershell
# 1. huggingface.co 注册 → Settings → Access Tokens → New token（勾 write 权限）
# 2. 上传（脚本自动建公开仓 + 逐文件上传，可断点重跑）：
$env:HF_TOKEN = "hf_xxxxxxxx"
python deploy/upload_models_to_hf.py
```

- 默认仓 `zhou06080120/blodd-answer-piper-models`（改仓名需同步 `Dockerfile.deploy` 中 `snapshot_download` 一行）
- 不想用脚本：网页建同名 Model Repository 后，按上表「文件 → 仓内路径」拖拽上传也可
- **推荐公开仓**：配合开源叙事、Render 构建拉取免凭证，模型页本身又是可展示资产；若必须私仓，需在 Render 服务环境变量中加 `HF_TOKEN`

BERT 基座无需上传——`Dockerfile.deploy` 构建期自动从 HF 拉取 `bert-base-chinese`（约 400MB）。

## 二、部署（约 10 分钟 + 首次 build 15 分钟）

1. render.com → **New → Blueprint** → 选本仓库 → 自动识别 `render.yaml`
   （注意它用的是 `Dockerfile.deploy`，根目录生产 Dockerfile 保持外挂 fail-fast 设计不动）
2. 无任何 Key 需要填写（纯本地模型）
3. 首次 build 较慢（拉 torch CPU + BERT 基座约 15 分钟），完成后 `curl https://<服务名>.onrender.com/health`

## 三、资源风险（如实评估）

免费实例 512MB RAM 需同时承载：torch CPU 运行时 + ONNX 会话 + 全库 Dense 索引。ONNX INT8 量化形态（简历口径 16.5ms/次）预期可行，但接近上限。若 Render Logs 出现 `Killed`（OOM）：
- 升级 Render Starter（$7/月），或
- 切 Zeabur（内存档位更宽裕，约 $5/月），配置同 `render.yaml`

## 四、Zeabur 部署（无国际信用卡时）

Render 创建服务（含免费档）已强制绑 Visa/MasterCard 验证。没有双币卡时走 Zeabur：

1. [zeabur.com](https://zeabur.com) 注册（GitHub 登录）→ 充值（**支持支付宝**）
2. 创建 Project → Add Service → **Git** → 授权并选 `blodd-answer-piper` 仓库
3. 构建配置：仓库根的 `zbpack.json` 已指定 `Dockerfile.deploy`（模型进镜像的 Demo 形态），Zeabur 自动识别，无需手填
4. 区域选**东京**（对国内访问最近）；内存档位建议 **1 GB**（BERT + torch，512MB 紧张）
5. 域名：服务 → Networking → 生成默认域名（`.zeabur.app`，国内直连快；也可绑自定义域名）

## 五、README 接线

部署后替换 `README.md`「在线体验」一节的 `{YOUR-APP}.onrender.com` 占位符。
