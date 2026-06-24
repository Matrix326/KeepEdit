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

## 3. 一键上传

默认创建公开仓库。当前发布版按公开模型仓库和公开数据集仓库发布，不使用 `--private`。
先打包数据：

```bash
bash scripts/pack_release_data_archives.sh
```

再切成 Hugging Face 更稳的 5GB 分卷：

```bash
bash scripts/split_release_data_archives.sh
```

默认 Python 上传脚本会上传 `hf_release/staging/hf_dataset/archives/**` 中的完整 tar；如果使用分卷发布，推荐用 Hugging Face CLI 逐个上传 `hf_release/staging/hf_dataset_split/archives/*` 到 repo 的 `archives/` 目录，失败时可以从单个分卷继续。

完整 tar 上传命令如下。上传过程使用 Hugging Face 的 `upload_large_folder`，会在归档根目录生成可恢复上传缓存 `.cache/.huggingface/`。

```bash
python scripts/upload_release_to_hf.py \
  --weights_repo_id <HF_USER_OR_ORG>/keepedit-release-weights \
  --data_repo_id <HF_USER_OR_ORG>/keepedit-release-data
```

如果本机代理使用自签名证书，`huggingface_hub` 可能在 TLS 校验处失败。确认当前网络环境可信后，可以加：

```bash
python scripts/upload_release_to_hf.py \
  --weights_repo_id <HF_USER_OR_ORG>/keepedit-release-weights \
  --data_repo_id <HF_USER_OR_ORG>/keepedit-release-data \
  --disable_ssl_verification
```

如果只上传权重：

```bash
python scripts/upload_release_to_hf.py \
  --weights_repo_id <HF_USER_OR_ORG>/keepedit-release-weights \
  --data_repo_id <HF_USER_OR_ORG>/keepedit-release-data \
  --skip_data
```

如果只上传数据：

```bash
python scripts/upload_release_to_hf.py \
  --weights_repo_id <HF_USER_OR_ORG>/keepedit-release-weights \
  --data_repo_id <HF_USER_OR_ORG>/keepedit-release-data \
  --skip_weights
```

如果确实要上传原始 `data/**` 小文件树，而不是归档包，额外加 `--upload_raw_data`；不建议常规发布这样做。

如果希望连 `reports/` 也上传到 dataset repo：

```bash
python scripts/upload_release_to_hf.py \
  --weights_repo_id <HF_USER_OR_ORG>/keepedit-release-weights \
  --data_repo_id <HF_USER_OR_ORG>/keepedit-release-data \
  --include_reports
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
