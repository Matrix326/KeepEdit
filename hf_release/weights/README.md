---
license: apache-2.0
library_name: diffusers
tags:
  - image-editing
  - qwen-image-edit
  - lora
  - keepedit
---

# KeepEdit Release LoRA Weights

本仓库只存放 KeepEdit 发布版 LoRA 权重，不包含 Qwen-Image-Edit-2511 基座模型。

目录结构：

```text
qwen_edit_2511_keepedit_gt_onestage/
  step-4404.safetensors

qwen_edit_2511_mtp_phasea/
  step-2269.safetensors

qwen_edit_2511_moe_teacher_onestage/
  step-2202.safetensors
```

三个 LoRA 的推理条件完全一致：

```text
source image + instruction -> edited image
```

推理时不需要 target、mask、专家候选图或 MoE teacher 图。

下载到项目目录：

```bash
huggingface-cli download <WEIGHTS_REPO_ID> \
  --repo-type model \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

随后可运行：

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
