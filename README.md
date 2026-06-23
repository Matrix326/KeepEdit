# KeepEdit

KeepEdit 是一个面向指令图像编辑的研究项目。最终发布版围绕 Qwen-Image-Edit-2511 展开，比较四条主线：

```text
1. Qwen2511 Base
2. GT-LoRA
3. MTP LoRA
4. MoE Teacher LoRA
```

所有可部署模型的推理形式都保持一致：

```text
source image + instruction -> edited image
```

训练期可以使用 target、mask、专家候选和 MoE teacher；最终推理时不输入这些额外信息。

## 1. 仓库结构

代码仓库只保留轻量文件：

```text
configs/        实验配置
docs/           中文流程文档
scripts/        数据、训练、评估和上传脚本
src/keepedit/   KeepEdit 核心代码
reports/        小型 release 指标文件
```

以下大目录不进入 git：

```text
checkpoints/
data/
external/
```

它们通过 Hugging Face 或官方仓库下载。详见：

```text
docs/HUGGINGFACE_RELEASE.md
docs/LICENSE_NOTES.md
```

## 2. 环境配置

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate hw4diff
pip install -e ".[all]"
```

项目依赖 DiffSynth-Studio 和 EditAR：

```text
external/DiffSynth-Studio/
external/EditAR/
```

这些会通过下载脚本或手动 clone 准备。

## 3. 下载权重和数据

### 3.1 下载 KeepEdit 发布权重

```bash
huggingface-cli download <HF_USER_OR_ORG>/keepedit-release-weights \
  --repo-type model \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

下载后应得到：

```text
checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors
checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors
checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors
```

### 3.2 下载 KeepEdit 发布数据

```bash
huggingface-cli download <HF_USER_OR_ORG>/keepedit-release-data \
  --repo-type dataset \
  --local-dir . \
  --local-dir-use-symlinks False

bash scripts/unpack_release_data_archives.sh
```

数据集仓库发布为少量归档文件，下载后先得到：

```text
archives/data_processed.tar
archives/data_candidates.tar
archives/data_teachers.tar
archives/data_diffsynth.tar
archives/data_outputs.tar
archives/MANIFEST.sha256
```

解包后应包含：

```text
data/processed/
data/candidates/
data/teachers/
data/diffsynth/
data/outputs/
```

### 3.3 下载基座模型和外部依赖

```bash
bash scripts/download_required_assets.sh
python scripts/check_required_assets.py
```

该脚本会准备：

```text
Qwen/Qwen-Image-Edit-2511
Qwen/Qwen-Image
Qwen/Qwen-Image-Edit
timbrooks/instruct-pix2pix
Qwen3-VL-8B-Instruct
EditAR 相关权重
```

## 4. 从 MagicBrush 原始数据重新预处理

如果不使用发布数据，可以从 MagicBrush parquet 重新生成 `data/processed`：

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

生成：

```text
data/processed/magicbrush_train/train.jsonl
data/processed/magicbrush_dev/dev.jsonl
```

## 5. Qwen2511 Base

不加载 LoRA，直接评估原始 Qwen-Image-Edit-2511：

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
GPUS=0 \
bash scripts/evaluate_qwen_edit_experiment.sh
```

多卡并行：

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
GPUS=0,1,2,3 \
PARALLEL_GPUS=0,1,2,3 \
bash scripts/evaluate_qwen_edit_experiment.sh
```

输出：

```text
data/outputs/magicbrush_dev_qwen2511_base/
reports/magicbrush_dev_qwen2511_base_release_metrics.csv
reports/magicbrush_dev_qwen2511_base_release_metrics_summary.json
```

详细说明见：

```text
docs/QWEN2511_BASELINE.md
```

## 6. GT-LoRA

GT-LoRA 直接使用 MagicBrush target 作为监督目标：

```text
source image + instruction -> target image
```

训练：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_gt_lora_qwen_edit.sh
```

输出：

```text
checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors
data/outputs/magicbrush_dev_qwen2511_gt_onestage/
reports/magicbrush_dev_qwen2511_gt_onestage_release_metrics.csv
```

详细说明见：

```text
docs/GT_LORA_WORKFLOW.md
```

## 7. MTP LoRA

MTP 使用 mask-preserved clean target：

```text
G_bar = M_soft * target + (1 - M_soft) * source
```

它显式告诉模型：编辑区域学习 target，背景区域保持 source。

训练：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_mtp_phasea.sh
```

输出：

```text
checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors
data/diffsynth/magicbrush_train_mtp_phasea/
data/outputs/magicbrush_dev_qwen2511_mtp_phasea/
reports/magicbrush_dev_qwen2511_mtp_phasea_release_metrics.csv
```

详细说明见：

```text
docs/MTP_LORA_WORKFLOW.md
docs/MTP_ALGORITHM.md
```

## 8. MoE Teacher LoRA

MoE 路线分两步：

```text
1. Pix2Pix / Qwen-Image-Edit / EditAR 生成专家候选
2. target + mask 监督下区域级融合，得到 MoE-Fusion Teacher
3. 用 teacher 训练 source-only Qwen2511 LoRA
```

构造专家候选和 teacher：

```bash
GPUS=0,1,2,3 bash scripts/run_keepedit_moe_fusion.sh
```

训练 MoE Teacher LoRA：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_moe_teacher_lora.sh
```

输出：

```text
data/candidates/magicbrush_train_pix2pix_qwen_editar/
data/teachers/magicbrush_train_moe_fusion/
checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors
data/outputs/magicbrush_dev_qwen2511_moe_teacher_onestage/
reports/magicbrush_dev_qwen2511_moe_teacher_onestage_release_metrics.csv
```

详细说明见：

```text
docs/MOE_LORA_WORKFLOW.md
```

## 9. 统一评估

评估任意已有 LoRA：

```bash
EXPERIMENT_NAME=qwen2511_gt_onestage \
LORA_PATH=checkpoints/qwen_edit_2511_keepedit_gt_onestage \
bash scripts/evaluate_qwen_edit_experiment.sh

EXPERIMENT_NAME=qwen2511_mtp_phasea \
LORA_PATH=checkpoints/qwen_edit_2511_mtp_phasea \
bash scripts/evaluate_qwen_edit_experiment.sh

EXPERIMENT_NAME=qwen2511_moe_teacher_onestage \
LORA_PATH=checkpoints/qwen_edit_2511_moe_teacher_onestage \
bash scripts/evaluate_qwen_edit_experiment.sh
```

启用 MLLM：

```bash
RUN_MLLM=1 \
MLLM_BACKEND=qwen3_vl \
bash scripts/evaluate_qwen_edit_experiment.sh
```

最终总表：

```text
reports/keepedit_release_full_metrics_comparison.csv
```

## 10. 当前发布版指标

```text
Qwen2511 Base:
  Target--Output SSIM = 0.450
  BG-SSIM = 0.696

GT-LoRA:
  Target--Output SSIM = 0.696
  BG-SSIM = 0.821

MTP LoRA:
  Target--Output SSIM = 0.740
  BG-SSIM = 0.828

MoE Teacher LoRA:
  Target--Output SSIM = 0.763
  BG-SSIM = 0.852
```

MoE Teacher LoRA 是当前客观指标最好的可部署模型；MTP LoRA 是更简洁、不依赖专家候选的强改进版本。

## 11. 上传到 Hugging Face

```bash
python scripts/upload_release_to_hf.py \
  --weights_repo_id <HF_USER_OR_ORG>/keepedit-release-weights \
  --data_repo_id <HF_USER_OR_ORG>/keepedit-release-data
```

说明见：

```text
docs/HUGGINGFACE_RELEASE.md
```
