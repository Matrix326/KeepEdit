# MoE LoRA 流程说明

本文档说明 `qwen2511_moe_teacher_onestage` 的完整流程：三专家候选生成、MoE-Fusion Teacher 构造，以及将 teacher 蒸馏到 Qwen2511 source-only LoRA。

## 1. 总体思想

MoE-Fusion Teacher 是训练期 teacher。

训练期允许使用：

```text
source image I
instruction p
target image G
edit mask M
Pix2Pix / Qwen-Image-Edit / EditAR 专家候选
```

构造区域级 teacher：

```text
T_moe = MoE_Fusion(I, p, G, M, candidates)
```

然后训练一个单独的 Qwen2511 LoRA：

```text
I, p -> T_moe
```

## 2. 专家候选

发布版真实专家：

```text
Pix2Pix           -> C_pix2pix
Qwen-Image-Edit   -> C_qwen
EditAR            -> C_editar
```

配置文件：

```text
configs/experiments/keepedit_stage1_moe.yaml
```

一键生成或复用候选：

```bash
GPUS=0,1,2,3 bash scripts/run_keepedit_moe_fusion.sh
```

脚本会先检查候选缓存是否完整：

```text
data/candidates/magicbrush_train_pix2pix_qwen_editar/predictions.jsonl
data/candidates/magicbrush_dev_pix2pix_qwen_editar/predictions.jsonl
```

如果不完整，会调用：

```bash
python scripts/run_experts_by_expert_multi_gpu.py
```

每个样本的候选会写入：

```text
data/candidates/.../_expert_runs/
```

## 3. Qwen 虚拟专家族

由于 Qwen-Image-Edit 的语义能力通常较强，但编辑强度可能偏保守或偏激进，MoE 在已有 `C_qwen` 上构造两个虚拟专家：

```text
C_qwen_cons = clip(I + 0.65 * (C_qwen - I), 0, 1)
C_qwen_aggr = clip(I + 1.25 * (C_qwen - I), 0, 1)
```

因此路由专家族为：

```text
{pix2pix, qwen, qwen_cons, qwen_aggr, editar}
```

## 4. 专家评分

每个专家在编辑区和背景区分别评分。

编辑区目标相似度：

```text
E_edit(C) = MSE(C, G; M) + alpha_s * (1 - SSIM(C, G; M))
```

背景保持：

```text
E_bg(C) = MSE(C, I; 1 - M) + alpha_b * (1 - SSIM(C, I; 1 - M))
```

编辑方向一致性：

```text
D_dir(C) = 1 - cos((C - I) * M, (G - I) * M)
```

边界惩罚：

```text
B(C) = rho * |C - I|_{M_bd}
     + (1 - rho) * |grad(C) - grad(I)|_{M_bd}
```

综合分数越低越好：

```text
S(C) =
  lambda_e  * E_edit(C)
  + lambda_b  * E_bg(C)
  + lambda_d  * D_dir(C)
  + lambda_bd * B(C)
  + lambda_f  * MSE(C, G)
```

## 5. 区域级 routing

MoE 不做整图选择，而是将编辑 mask 分解为连通区域：

```text
M = union_k M_k
```

每个区域独立选择专家：

```text
e_k* = argmin_e S_k(C_e)
```

如果最优专家明显优于第二名，则 hard routing；如果置信度较低，则使用 top-k soft routing：

```text
pi_{k,e} = exp(-S_k(C_e) / tau) / sum_e exp(-S_k(C_e) / tau)
C_k = sum_e pi_{k,e} C_e
```

局部编辑默认使用 source 背景：

```text
C_bg = I
```

这样可以避免专家的背景噪声被一起蒸馏给学生。

## 6. 融合与 fallback

初始融合：

```text
T0 = (1 - M_soft) * C_bg + sum_k M_soft_k * C_k
```

之后做：

```text
1. feather mask
2. color harmonization
3. Laplacian pyramid blending
```

如果融合结果比最佳单专家更差，或编辑区变化过小，则 fallback 到：

```text
best expert
canonical target
dataset target
```

这样避免 teacher 变成“几乎没改”的图。

## 7. 构造 MoE-Fusion Teacher

一键脚本：

```bash
GPUS=0,1,2,3 bash scripts/run_keepedit_moe_fusion.sh
```

生成：

```text
data/teachers/magicbrush_train_moe_fusion/predictions.jsonl
data/teachers/magicbrush_train_moe_fusion/images/
data/teachers/magicbrush_train_moe_fusion/scores/
data/teachers/magicbrush_train_moe_fusion/attribution/
data/teachers/magicbrush_train_moe_fusion/confidence/

data/teachers/magicbrush_dev_moe_fusion/predictions.jsonl
```

同时生成 dev teacher 指标：

```text
reports/magicbrush_dev_moe_fusion_teacher_release_metrics.csv
reports/magicbrush_dev_moe_fusion_teacher_release_metrics_summary.json
reports/visual_gallery_magicbrush_dev_moe_fusion/index.html
```

## 8. MoE Teacher LoRA 微调

将 teacher 作为训练目标，仍然使用 source-only metadata：

```json
{
  "image": "T_moe path",
  "edit_image": ["source image path"],
  "prompt": "instruction + preservation suffix",
  "mask_image": "teacher or dataset mask",
  "phase": "moe_teacher_onestage",
  "teacher_confidence": 0.0
}
```

样本权重由 teacher confidence 调节：

```text
w = teacher_min_weight + confidence * (teacher_max_weight - teacher_min_weight)
```

训练命令：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_moe_teacher_lora.sh
```

默认参数：

```text
QWEN_EPOCHS=1
QWEN_LR=5e-5
QWEN_RANK=32
```

## 9. 训练产物

最终 LoRA：

```text
checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors
```

评测产物：

```text
data/outputs/magicbrush_dev_qwen2511_moe_teacher_onestage/
reports/magicbrush_dev_qwen2511_moe_teacher_onestage_release_metrics.csv
reports/magicbrush_dev_qwen2511_moe_teacher_onestage_release_metrics_summary.json
reports/magicbrush_dev_qwen2511_moe_teacher_onestage_mllm_preference.jsonl
reports/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage/index.html
```

## 10. 当前发布版结果

```text
Target--Output SSIM: 0.763
Target--Output PSNR: 19.006
BG-SSIM:             0.852
Input--Output SSIM:  0.792
Edit-Region Change:  0.167
```

MoE Teacher LoRA 是当前客观指标最好的可部署模型。它的关键点是：复杂的专家搜索和 target/mask 监督只发生在训练期，最终能力被蒸馏到单个 Qwen2511 LoRA 中。
