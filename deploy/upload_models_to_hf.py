# -*- coding: utf-8 -*-
"""把 blodd-answer-piper 训练产物上传到 Hugging Face Hub（公网部署前置步骤）

为什么走 HF Hub：biencoder checkpoint 173.85MB 超 GitHub 单文件 100MB 硬限制；
且 serve.py 启动时无条件 torch.load(.pt) 读取训练超参，.pt 为部署必需品。

最小部署产物集（实测体积）：
  - checkpoints/biencoder_hardneg_best.pt    173.9MB  必需（启动读 args 元数据 + PyTorch 回退）
  - onnx/biencoder_hardneg_best_int8.onnx    43.1MB   必需（实际推理后端，ONNX-INT8）
  - checkpoints/crossencoder_best.pt         173.9MB  可选（缺失自动降级 RRF+余弦组合分）

用法：
  1. huggingface.co 注册 → Settings → Access Tokens → New token（勾 write 权限）
  2. PowerShell：
       $env:HF_TOKEN = "hf_xxxxxxxx"
       python deploy/upload_models_to_hf.py                          # 默认仓：zhou06080120/blodd-answer-piper-models
       python deploy/upload_models_to_hf.py --repo 你的用户名/仓名    # 自定义仓名（需同步改 Dockerfile.deploy）

说明：脚本创建公开仓库（推荐——配合开源叙事，且 Render 构建拉取无需凭证）；
若必须私仓，需在 Render 环境变量加 HF_TOKEN 供构建期 snapshot_download 使用。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (本地路径, 仓内路径, 是否必需)
FILES = [
    ("outputs/checkpoints/biencoder_hardneg_best.pt", "checkpoints/biencoder_hardneg_best.pt", True),
    ("outputs/onnx/biencoder_hardneg_best_int8.onnx", "onnx/biencoder_hardneg_best_int8.onnx", True),
    ("outputs/checkpoints/crossencoder_best.pt", "checkpoints/crossencoder_best.pt", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="zhou06080120/blodd-answer-piper-models",
                    help="HF 模型仓 ID（用户名/仓名）")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("缺少 HF_TOKEN 环境变量（huggingface.co → Settings → Access Tokens → write 权限）")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    print(f"目标仓库：https://huggingface.co/{args.repo}\n")

    for local, remote, required in FILES:
        p = ROOT / local
        if not p.exists():
            if required:
                sys.exit(f"致命：必需产物缺失 {local}（先按 README 训练流程生成）")
            print(f"跳过（可选）：{local} 不存在，线上将降级 RRF+余弦组合分\n")
            continue
        print(f"上传 {local}（{p.stat().st_size / 1048576:.1f} MB）→ {args.repo}/{remote}")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote,
                        repo_id=args.repo, repo_type="model")

    print(f"\n完成。验证：https://huggingface.co/{args.repo}/tree/main")
    print("下一步：push 仓库后到 Render 部署（Dockerfile.deploy 会自动从此仓拉取产物）")


if __name__ == "__main__":
    main()
