# Qwen2511 Base 基线说明

本文档说明如何得到 `qwen2511_base` 基线结果。该基线不加载任何 LoRA，只使用原始 Qwen-Image-Edit-2511 对 MagicBrush-dev 做 source-only 图像编辑推理。

## 1. 基线定义

输入：

```text
原图 I + 编辑指令 p
```

输出：

```text
编辑图 O = Qwen-Image-Edit-2511(I, p)
```

不使用：

```text
target image
mask
expert candidate
MoE teacher
LoRA checkpoint
```

这个实验用于回答：不做任何任务微调时，Qwen-Image-Edit-2511 在 MagicBrush-dev 上的基础能力如何。

## 2. 需要下载什么

Base 推理依赖 DiffSynth-Studio 版本的 Qwen-Image-Edit-2511 及 Qwen-Image text encoder / VAE：

```text
checkpoints/diffsynth/Qwen/Qwen-Image-Edit-2511/
checkpoints/diffsynth/Qwen/Qwen-Image/
external/DiffSynth-Studio/
```

推荐直接运行统一下载脚本：

```bash
bash scripts/download_required_assets.sh
python scripts/check_required_assets.py
```

如果已经从 Hugging Face 下载了 KeepEdit 发布版数据，则项目根目录应包含：

```text
data/processed/magicbrush_dev/dev.jsonl
data/processed/magicbrush_dev/images/
data/processed/magicbrush_dev/masks/
```

也可以从 MagicBrush parquet 重新预处理，见 README 的“数据准备”部分。

## 3. 数据格式

`data/processed/magicbrush_dev/dev.jsonl` 每行是一个样本：

```json
{
  "id": "34_2",
  "input_image": "data/processed/magicbrush_dev/images/34_2_input.png",
  "instruction": "Open the zebra's mouth.",
  "target_image": "data/processed/magicbrush_dev/images/34_2_target.png",
  "mask_image": "data/processed/magicbrush_dev/masks/34_2_mask.png"
}
```

其中 `target_image` 和 `mask_image` 不会送入模型，只用于后续评测。

## 4. 运行命令

单卡推理：

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
GPUS=0 \
bash scripts/evaluate_qwen_edit_experiment.sh
```

多卡并行推理：

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
GPUS=0,1,2,3 \
PARALLEL_GPUS=0,1,2,3 \
bash scripts/evaluate_qwen_edit_experiment.sh
```

常用参数：

```text
QWEN_INFER_STEPS=40
QWEN_CFG_SCALE=4.0
QWEN_DENOISING_STRENGTH=0.9
QWEN_MAX_PIXELS=262144
QWEN_LIMIT=        # 为空表示全量 dev
RUN_MLLM=0        # 设置为 1 时运行 Qwen3-VL 评测
```

## 5. 输出文件

推理结果：

```text
data/outputs/magicbrush_dev_qwen2511_base/
  predictions.jsonl
  images/
  raw/
  masks/
```

可视化：

```text
reports/visual_gallery_magicbrush_dev_qwen2511_base/index.html
```

指标：

```text
reports/magicbrush_dev_qwen2511_base_release_metrics.csv
reports/magicbrush_dev_qwen2511_base_release_metrics_summary.json
reports/magicbrush_dev_qwen2511_base_mllm_preference.jsonl   # RUN_MLLM=1 时生成
```

## 6. 当前发布版结果

MagicBrush-dev 上的发布版客观指标记录在：

```text
reports/keepedit_release_full_metrics_comparison.csv
```

`qwen2511_base` 当前结果：

```text
Target--Output SSIM: 0.450
Target--Output PSNR: 14.292
BG-SSIM:             0.696
Input--Output SSIM:  0.450
Edit-Region Change:  0.135
```

这个基线说明 Qwen2511 具备较强通用编辑能力，但没有针对 MagicBrush 目标分布微调时，目标相似度仍然偏低。
