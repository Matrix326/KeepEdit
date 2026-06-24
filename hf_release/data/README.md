---
license: cc-by-nc-4.0
tags:
  - image-editing
  - magicbrush
  - keepedit
---

# KeepEdit Release Data

本仓库存放 KeepEdit 发布版实验数据与中间结果。为避免在 Hugging Face 上平铺十几万个小文件，发布内容采用归档分卷形式；下载后运行解包脚本即可恢复项目根目录下的 `data/`。

主要内容：

```text
archives/data_processed.tar.000.part
archives/data_candidates.tar.000.part
archives/data_teachers.tar.000.part
...
archives/MANIFEST.sha256
```

`scripts/unpack_release_data_archives.sh` 会自动识别分卷，按文件名前缀顺序拼接后解包。

解包后得到：

```text
data/processed/
  magicbrush_train/train.jsonl
  magicbrush_dev/dev.jsonl
  images/
  masks/

data/candidates/
  magicbrush_train_pix2pix_qwen_editar/
  magicbrush_dev_pix2pix_qwen_editar/

data/teachers/
  magicbrush_train_moe_fusion/
  magicbrush_dev_moe_fusion/

data/diffsynth/
  magicbrush_train_mtp_phasea/

data/outputs/
  magicbrush_dev_qwen2511_base/
  magicbrush_dev_qwen2511_gt_onestage/
  magicbrush_dev_qwen2511_mtp_phasea/
  magicbrush_dev_qwen2511_moe_teacher_onestage/
```

下载到项目目录：

```bash
huggingface-cli download <DATA_REPO_ID> \
  --repo-type dataset \
  --local-dir . \
  --local-dir-use-symlinks False

bash scripts/unpack_release_data_archives.sh
```

如果不下载本仓库数据，也可以从原始 MagicBrush parquet 重新预处理：

```bash
python scripts/download_hf_dataset_files.py \
  --repo_id osunlp/MagicBrush \
  --out_dir data/raw/MagicBrush \
  --pattern "data/train-*.parquet" \
  --pattern "data/dev-*.parquet"

python scripts/validate_magicbrush_parquet.py \
  --root data/raw/MagicBrush/data \
  --split train \
  --split dev

python scripts/prepare_magicbrush_parquet.py \
  --parquet_dir data/raw/MagicBrush \
  --split train \
  --out_dir data/processed/magicbrush_train

python scripts/prepare_magicbrush_parquet.py \
  --parquet_dir data/raw/MagicBrush \
  --split dev \
  --out_dir data/processed/magicbrush_dev
```

数据再分发应遵守 MagicBrush 与相关模型输出的许可证约束。
