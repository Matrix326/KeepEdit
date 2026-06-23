# MTP LoRA 流程说明

本文档说明 `qwen2511_mtp_phasea` 的算法、数据处理、训练命令和产物位置。MTP 是 Masked Target Programming，只保留 PhaseA，不再使用 GT-anchor PhaseB。

## 1. 方法动机

GT-LoRA 直接学习：

```text
I, p -> G
```

其中 `G` 是完整 target。问题是：target 中除了真正需要编辑的区域外，背景也可能因为人工编辑、对齐误差或生成噪声发生变化。模型直接拟合整图 target 时，容易学到不必要的背景改动。

MTP 的核心做法是构造 clean target：

```text
G_bar = M_soft * G + (1 - M_soft) * I
```

含义：

```text
编辑区域：学习 target
背景区域：锚定 source
边界区域：soft mask 平滑过渡
```

最终模型仍然只输入 source image 和 instruction，不输入 mask。

## 2. Mask 选择

输入样本：

```text
I: source image
p: instruction
G: target image
M_data: MagicBrush dataset mask
```

MTP 构造多个候选 mask：

```text
source_target_diff_otsu
dataset_mask
dataset_mask_inverted
```

source-target 差异图：

```text
D(i,j) = mean_c |I(i,j,c) - G(i,j,c)|
```

每个 mask 候选会经过：

```text
remove small objects
remove small holes
binary closing
binary dilation
```

候选 mask 评分：

```text
S(M) =
  inside_diff(M) - outside_diff(M)
  + beta * diff_coverage(M)
  - broad_penalty(M)
  - tiny_penalty(M)
  - local_penalty(M)
```

最终：

```text
M_star = argmax_M S(M)
```

如果 prompt 是全局编辑，或者 mask 面积超过全局阈值，则使用全图监督。

## 3. Clean Target 与边界

局部编辑时：

```text
M_dilate = Dilate(M_star, soft_dilate_radius)
M_soft   = GaussianBlur(M_dilate, soft_blur_sigma)
M_bd     = Dilate(M_star, boundary_radius) - Erode(M_star, boundary_radius)
G_bar    = M_soft * G + (1 - M_soft) * I
```

全局编辑时：

```text
M_soft = 1
G_bar = G
```

同时加入少量 no-op 保持样本：

```text
source image + "Do not change the image." -> source image
```

这些样本使用低权重，只作为背景保持正则。

## 4. Metadata 生成

一键脚本内部会调用：

```bash
python scripts/prepare_mtp_lora_metadata.py \
  --jsonl data/processed/magicbrush_train/train.jsonl \
  --out_dir data/diffsynth/magicbrush_train_mtp_phasea \
  --mask_edit_weight 4.0 \
  --mask_bg_weight 0.3 \
  --boundary_weight 0.15 \
  --soft_dilate_radius 24 \
  --soft_blur_sigma 7.0 \
  --noop_fraction 0.03 \
  --noop_weight 0.03
```

生成：

```text
data/diffsynth/magicbrush_train_mtp_phasea/metadata.json
data/diffsynth/magicbrush_train_mtp_phasea/summary.json
data/diffsynth/magicbrush_train_mtp_phasea/clean_targets/
data/diffsynth/magicbrush_train_mtp_phasea/mtp_masks/
```

metadata 核心结构：

```json
{
  "image": "G_bar path",
  "edit_image": ["source image path"],
  "prompt": "instruction + preservation suffix",
  "mask_image": "M_soft path",
  "boundary_image": "M_bd path",
  "phase": "mtp_sft"
}
```

## 5. MTP Loss

MTP 使用 flow matching 的逐 latent 误差 `E`，并加入编辑区、背景区和边界区三项：

```text
L_edit = sum(M * E) / (sum(M) + eps)
L_bg   = sum((1 - M) * E) / (sum(1 - M) + eps)
L_bd   = sum(M_bd * E) / (sum(M_bd) + eps)
```

最终：

```text
L_mtp =
  (lambda_edit * L_edit
   + lambda_bg * L_bg
   + lambda_boundary * L_bd)
  / (lambda_edit + lambda_bg + lambda_boundary)
```

发布版参数：

```text
lambda_edit = 4.0
lambda_bg = 0.3
lambda_boundary = 0.15
LoRA rank = 16
learning rate = 5e-5
epochs = 1
```

## 6. 一键训练命令

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_mtp_phasea.sh
```

该脚本会执行：

```text
1. 生成 MTP clean-target metadata
2. 校验 source-only metadata
3. 训练 Qwen2511 MTP LoRA
4. 保存最终 LoRA
5. 默认运行 dev 推理和 release metrics
```

## 7. 训练产物

最终 LoRA：

```text
checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors
```

数据产物：

```text
data/diffsynth/magicbrush_train_mtp_phasea/
```

评测产物：

```text
data/outputs/magicbrush_dev_qwen2511_mtp_phasea/
reports/magicbrush_dev_qwen2511_mtp_phasea_release_metrics.csv
reports/magicbrush_dev_qwen2511_mtp_phasea_release_metrics_summary.json
reports/visual_gallery_magicbrush_dev_qwen2511_mtp_phasea/index.html
```

## 8. 当前发布版结果

```text
Target--Output SSIM: 0.740
Target--Output PSNR: 18.456
BG-SSIM:             0.828
Input--Output SSIM:  0.761
Edit-Region Change:  0.172
```

相比 GT-LoRA，MTP LoRA 的编辑区域变化更明显，同时背景保持略有提升。
