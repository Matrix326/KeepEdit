# GT-LoRA 微调流程说明

本文档说明 `qwen2511_gt_onestage` 是如何训练和评测的。GT-LoRA 是最直接的监督微调基线：输入原图和指令，目标图使用 MagicBrush 的人工 target。

## 1. 方法目标

GT-LoRA 训练一个 Qwen-Image-Edit-2511 LoRA，使模型学习：

```text
source image + instruction -> MagicBrush target image
```

训练阶段可以使用 target 和 mask 计算 loss；推理阶段只输入原图和 prompt。

## 2. LoRA 原理

Qwen2511 的 DiT 主干参数冻结为 `W0`，只训练低秩增量：

```text
W' = W0 + ΔW
ΔW = alpha / r * B A
```

其中：

```text
r: LoRA rank
alpha: LoRA scaling
A, B: 可训练低秩矩阵
```

本项目将 LoRA 注入 Qwen2511 DiT 的 attention 和 modulation/MLP 层：

```text
to_q, to_k, to_v,
add_q_proj, add_k_proj, add_v_proj,
to_out.0, to_add_out,
img_mlp.net.2, img_mod.1,
txt_mlp.net.2, txt_mod.1
```

## 3. 训练数据构造

输入 JSONL：

```text
data/processed/magicbrush_train/train.jsonl
```

运行：

```bash
python scripts/prepare_qwen_lora_metadata.py \
  --jsonl data/processed/magicbrush_train/train.jsonl \
  --out_dir data/diffsynth/magicbrush_train_qwen2511_gt_onestage \
  --target_mode gt
```

生成 DiffSynth metadata：

```text
data/diffsynth/magicbrush_train_qwen2511_gt_onestage/metadata.json
data/diffsynth/magicbrush_train_qwen2511_gt_onestage/summary.json
```

每条 metadata 的核心结构：

```json
{
  "image": "MagicBrush target image",
  "edit_image": ["source image"],
  "prompt": "instruction + preservation suffix",
  "mask_image": "edit mask",
  "phase": "gt_onestage",
  "condition_mode": "input_only"
}
```

注意：`image` 是监督目标，`edit_image` 才是模型输入。

## 4. 训练 loss

训练采用 DiffSynth/Qwen-Image-Edit 的 flow matching SFT。设预测 flow 和目标 flow 的逐元素误差为 `E`，将 mask resize 到 latent 尺度后：

```text
L_edit = sum(M * E) / (sum(M) + eps)
L_bg   = sum((1 - M) * E) / (sum(1 - M) + eps)
```

GT-LoRA 使用区域加权：

```text
L = (lambda_edit * L_edit + lambda_bg * L_bg)
    / (lambda_edit + lambda_bg)
```

默认：

```text
lambda_edit = 1.5
lambda_bg = 0.5
```

## 5. 一键训练命令

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_gt_lora_qwen_edit.sh
```

该脚本会依次执行：

```text
1. 生成 GT metadata
2. 校验 source-only metadata
3. 使用 accelerate 启动 DiffSynth 训练
4. 保存 LoRA
5. 默认运行 dev 推理和 release metrics
```

关键默认参数：

```text
QWEN_EPOCHS=2
QWEN_LR=1e-4
QWEN_RANK=32
QWEN_SAVE_STEPS=1000
QWEN_MAX_PIXELS=262144
QWEN_DATASET_REPEAT=1
```

## 6. 训练产物

最终 LoRA：

```text
checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors
```

评测输出：

```text
data/outputs/magicbrush_dev_qwen2511_gt_onestage/
reports/magicbrush_dev_qwen2511_gt_onestage_release_metrics.csv
reports/magicbrush_dev_qwen2511_gt_onestage_release_metrics_summary.json
reports/magicbrush_dev_qwen2511_gt_onestage_mllm_preference.jsonl
```

## 7. 单独评测已有权重

```bash
EXPERIMENT_NAME=qwen2511_gt_onestage \
LORA_PATH=checkpoints/qwen_edit_2511_keepedit_gt_onestage \
bash scripts/evaluate_qwen_edit_experiment.sh
```

## 8. 当前发布版结果

```text
Target--Output SSIM: 0.696
Target--Output PSNR: 18.244
BG-SSIM:             0.821
Input--Output SSIM:  0.702
Edit-Region Change:  0.043
```

GT-LoRA 证明 Qwen2511 对 MagicBrush 任务微调是有效的，但也暴露出一个问题：模型会变得偏保守，编辑区域变化较小，局部位置和背景保持仍有提升空间。
