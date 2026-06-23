# MTP-PhaseA Algorithm

This document describes the release MTP path only.

## Goal

Train a Qwen-Image-Edit-2511 LoRA that is deployed as:

```text
source image + instruction -> edited image
```

During training, MTP uses the MagicBrush target and mask to build a cleaner
training target. At inference time, the model receives only the source image and
instruction. It does not receive the target, mask, clean target, or expert image.

## Inputs

For each MagicBrush sample:

```text
I: source image
p: edit instruction
G: ground-truth target image
M_data: dataset mask, if available
```

MTP outputs:

```text
G_bar: mask-preserved clean target
M_star: selected hard edit mask
M_soft: feathered soft edit mask
M_bd: boundary mask
```

The DiffSynth metadata row is:

```json
{
  "edit_image": ["source image path"],
  "prompt": "instruction",
  "image": "G_bar path",
  "mask_image": "M_soft path",
  "boundary_image": "M_bd path",
  "phase": "mtp_sft"
}
```

## Mask Candidate Construction

MTP does not blindly trust the dataset mask. It creates candidates from the
source-target difference and the dataset mask polarity:

```text
M_candidates = {
  source_target_diff_otsu,
  dataset_mask,
  inverted_dataset_mask
}
```

The source-target difference map is:

```text
D(i,j) = mean_c |I(i,j,c) - G(i,j,c)|
```

The diff candidate is produced by Otsu thresholding, with a fixed threshold
fallback when the Otsu foreground is too small.

Every mask candidate is cleaned by:

```text
remove small objects
remove small holes
binary closing
binary dilation
```

## Mask Scoring

For a candidate mask `M`:

```text
area(M) = mean(M)
inside(M) = mean(D | M = 1)
outside(M) = mean(D | M = 0)
coverage(M) = sum(M * D) / sum(D)
```

The score is:

```text
S(M) =
  inside(M) - outside(M)
  + beta * coverage(M)
  - gamma * broad_penalty(M)
  - eta * tiny_penalty(M)
  - local_penalty(M)
```

The selected hard mask is:

```text
M_star = argmax_M S(M)
```

Global edit prompts such as style, lighting, or whole-background edits are
allowed to use full-image masks. Local edits are penalized when the mask is too
broad, because MTP is designed to keep unrelated regions anchored to the source.

## Clean Target

For local edits:

```text
M_soft = GaussianBlur(Dilate(M_star))
G_bar = M_soft * G + (1 - M_soft) * I
```

For global edits:

```text
M_soft = 1
G_bar = G
```

This makes the training target explicit:

```text
edit region      -> learn the target
background       -> preserve the source
mask boundary    -> transition smoothly
```

## Loss

Qwen-Image-Edit-2511 is trained with flow matching. Let `E` be the per-latent
squared error between the predicted flow and the scheduler target. MTP resizes
the masks to latent resolution and applies region-normalized losses:

```text
L_edit = sum(M * E) / (sum(M) + eps)
L_bg   = sum((1 - M) * E) / (sum(1 - M) + eps)
L_bd   = sum(M_bd * E) / (sum(M_bd) + eps)
```

The final MTP-PhaseA loss is:

```text
L_mtp =
  (lambda_edit * L_edit
   + lambda_bg * L_bg
   + lambda_boundary * L_bd)
  / (lambda_edit + lambda_bg + lambda_boundary)
```

Release settings:

```text
lambda_edit = 4.0
lambda_bg = 0.3
lambda_boundary = 0.15
LoRA rank = 16
learning rate = 5e-5
epochs = 1
```

## No-op Regularization

MTP adds a small number of source-preservation rows:

```text
source image + "Do not change the image." -> source image
```

These rows use low loss weight and zero masks. They regularize preservation but
do not dominate the edit supervision.

## Release Commands

Train and evaluate the release MTP-PhaseA LoRA:

```bash
GPUS=0,1,2,3 NUM_PROCESSES=4 bash scripts/run_mtp_phasea.sh
```

Lower-level training entry:

```bash
QWEN_DATASET_DIR=data/diffsynth/magicbrush_train_mtp_phasea \
CKPT_DIR=checkpoints/qwen_edit_2511_mtp_phasea \
bash scripts/run_mtp_lora_qwen_edit.sh
```

Evaluate an existing checkpoint:

```bash
EXPERIMENT_NAME=qwen2511_mtp_phasea \
LORA_PATH=checkpoints/qwen_edit_2511_mtp_phasea \
bash scripts/evaluate_qwen_edit_experiment.sh
```
