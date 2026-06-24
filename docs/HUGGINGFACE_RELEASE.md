# Hugging Face 发布说明

本项目代码仓库不提交 `checkpoints/`、`data/`、`external/`。这些大文件通过 Hugging Face 分发。

推荐拆成两个仓库：

```text
模型权重 repo:  <HF_USER_OR_ORG>/keepedit-release-weights
数据集 repo:    <HF_USER_OR_ORG>/keepedit-release-data
```

其中：

```text
weights repo 类型: model
data repo 类型:    dataset
```

## 1. 发布内容

### 1.1 权重

只上传最终 LoRA，不上传中间 step，不上传 Qwen 基座模型：

```text
checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors
checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors
checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors
```

上传后 repo 内结构为：

```text
qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors
qwen_edit_2511_mtp_phasea/step-2269.safetensors
qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors
README.md
```

### 1.2 数据

发布版默认不把 17 万多个数据文件直接平铺到 Hugging Face，而是上传归档包。为避免单个文件超过 20GB，实际发布使用 5GB 左右的分卷：

```text
archives/data_processed.tar.000.part
archives/data_candidates.tar.000.part
archives/data_teachers.tar.000.part
...
archives/MANIFEST.sha256
```

解包后恢复为：

```text
data/processed/
data/candidates/
data/teachers/
data/diffsynth/
data/outputs/
```

如果只想重新生成最小复现实验，可以只保留：

```text
data/processed/
data/teachers/
data/diffsynth/magicbrush_train_mtp_phasea/
data/outputs/
```

但完整 MoE 复现需要 `data/candidates/`。

## 2. 登录 Hugging Face

安装依赖：

```bash
pip install -U huggingface_hub
```

登录：

```bash
huggingface-cli login
```

或者使用环境变量：

```bash
export HF_TOKEN=hf_xxx
```

## 3. 上传流程

默认创建公开仓库。当前发布版按公开模型仓库和公开数据集仓库发布，不使用 `--private`。

### 3.1 上传权重

只上传最终 LoRA 权重。基座模型不进入本项目发布仓库。

```bash
hf upload <HF_USER_OR_ORG>/keepedit-release-weights \
  checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors \
  qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors \
  --repo-type model

hf upload <HF_USER_OR_ORG>/keepedit-release-weights \
  checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors \
  qwen_edit_2511_mtp_phasea/step-2269.safetensors \
  --repo-type model

hf upload <HF_USER_OR_ORG>/keepedit-release-weights \
  checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors \
  qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors \
  --repo-type model

hf upload <HF_USER_OR_ORG>/keepedit-release-weights \
  hf_release/weights/README.md README.md \
  --repo-type model
```

### 3.2 打包数据

先打包数据：

```bash
bash scripts/pack_release_data_archives.sh
```

再切成 Hugging Face 更稳的 5GB 分卷：

```bash
bash scripts/split_release_data_archives.sh
```

### 3.3 上传数据分卷

发布版使用分卷上传。脚本会：

```text
1. 检查远端是否已经存在该分卷，存在则跳过；
2. 单分卷上传失败时自动重试；
3. 单分卷上传长时间无响应时按超时中断并重试；
4. 上传全部 data_*.tar.*.part 和 MANIFEST.sha256。
```

推荐命令：

```bash
REPO_ID=<HF_USER_OR_ORG>/keepedit-release-data \
bash scripts/upload_split_release_archives.sh
```

如果网络慢或单文件经常卡住，可以调大超时：

```bash
REPO_ID=<HF_USER_OR_ORG>/keepedit-release-data \
UPLOAD_TIMEOUT_SECONDS=3600 \
MAX_RETRIES=5 \
bash scripts/upload_split_release_archives.sh
```

## 4. 下载方式

下载权重到项目的 `checkpoints/`：

```bash
huggingface-cli download <HF_USER_OR_ORG>/keepedit-release-weights \
  --repo-type model \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

下载数据到项目根目录：

```bash
huggingface-cli download <HF_USER_OR_ORG>/keepedit-release-data \
  --repo-type dataset \
  --local-dir . \
  --local-dir-use-symlinks False

bash scripts/unpack_release_data_archives.sh
```

下载外部依赖和基座模型：

```bash
bash scripts/download_required_assets.sh
python scripts/check_required_assets.py
```

## 5. 为什么代码仓库不包含大文件

`.gitignore` 已经排除：

```text
data/
checkpoints/
external/
reports/*.csv
reports/*.json
reports/*.jsonl
reports/visual_gallery*/
```

代码仓库只保留：

```text
src/
scripts/
configs/
docs/
README.md
environment.yml
pyproject.toml
Makefile
```

这样 Git 仓库保持轻量，所有大文件通过 Hugging Face 下载。

## 6. 发布前检查

```bash
python - <<'PY'
from pathlib import Path
required = [
  "checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors",
  "checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors",
  "checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors",
  "data/processed/magicbrush_train/train.jsonl",
  "data/processed/magicbrush_dev/dev.jsonl",
  "data/teachers/magicbrush_train_moe_fusion/predictions.jsonl",
  "data/teachers/magicbrush_dev_moe_fusion/predictions.jsonl",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("Missing:\\n" + "\\n".join(missing))
print("release assets are ready")
PY
```
