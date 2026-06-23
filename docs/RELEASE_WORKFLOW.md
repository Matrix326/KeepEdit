# KeepEdit Release Workflow

This document describes the release code path retained in the repository.

## 1. Preprocessing

Input data is converted to MagicBrush-style JSONL:

```text
id
input_image
instruction
target_image
mask_image
metadata
```

Main scripts:

```text
scripts/prepare_magicbrush_parquet.py
scripts/validate_magicbrush_parquet.py
```

## 2. Expert Candidate Generation

The release expert set is:

```text
Pix2Pix
Qwen-Image-Edit
EditAR
```

Main scripts:

```text
scripts/run_experts_multi_gpu.py
scripts/run_experts_by_expert_multi_gpu.py
src/keepedit/pipelines/run_experts.py
src/keepedit/experts/
```

## 3. MoE-Fusion Teacher

The MoE teacher uses target/mask supervision at training time. It scores each
expert in edit regions and background regions, routes connected edit components
to the best expert, and blends regions with feathering/Laplacian composition.

Main files:

```text
scripts/build_moe_fusion_teacher.py
scripts/run_keepedit_moe_fusion.sh
src/keepedit/moe/scoring.py
src/keepedit/moe/fusion.py
```

## 4. Qwen-Image-Edit-2511 Experiments

Retained Qwen2511 experiments:

```text
qwen2511_base
qwen2511_gt_onestage
qwen2511_mtp_phasea
qwen2511_moe_teacher_onestage
```

Final inference is always source-only:

```text
edit_image = [source image]
prompt = instruction
```

No expert candidate, teacher image, target image, or mask is passed as model
condition at inference time.

Main files:

```text
scripts/prepare_qwen_lora_metadata.py
scripts/run_gt_lora_qwen_edit.sh
scripts/run_stage2_onestage_qwen_edit.sh
scripts/prepare_mtp_lora_metadata.py
scripts/run_mtp_lora_qwen_edit.sh
scripts/run_mtp_phasea.sh
scripts/run_moe_teacher_lora.sh
src/keepedit/pipelines/run_qwen_edit_lora.py
```

## 5. Release Metrics

The retained metrics are:

```text
SSIM / PSNR (Target, Output)
BG-SSIM
SSIM / PSNR (Input, Output)
Edit-Region Change
MLLM preference metrics
```

Main files:

```text
src/keepedit/evaluation/release_metrics.py
src/keepedit/evaluation/mllm_preference.py
scripts/evaluate_qwen_edit_experiment.sh
```

The release workflow intentionally keeps only the four model families above:
base inference, GT-LoRA, MTP-PhaseA, and MoE-Teacher LoRA.
