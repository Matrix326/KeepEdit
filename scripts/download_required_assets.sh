#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
LOG_DIR="${LOG_DIR:-reports/logs}"
mkdir -p "$LOG_DIR"

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"

DL=(conda run --no-capture-output -n "${ENV_NAME:-hw4diff}" python scripts/download_hf_with_curl.py --connections "${HF_CONNECTIONS:-8}")

echo "[1/9] Stage 1 Pix2Pix baseline"
"${DL[@]}" \
  --repo_id timbrooks/instruct-pix2pix \
  --pattern "*" \
  --local_dir checkpoints/hf/timbrooks__instruct-pix2pix

echo "[2/9] Stage 1 Qwen-Image-Edit baseline"
"${DL[@]}" \
  --repo_id Qwen/Qwen-Image-Edit \
  --pattern "*" \
  --local_dir checkpoints/hf/Qwen__Qwen-Image-Edit

echo "[3/9] EditAR release checkpoint"
"${DL[@]}" \
  --repo_type dataset \
  --repo_id JitengMu/CVPR2025_EditAR_release \
  --file editar_release/editar_release.pt \
  --local_dir external/EditAR/checkpoints/editar
ln -sf editar_release/editar_release.pt external/EditAR/checkpoints/editar/editar_release.pt

echo "[4/9] EditAR VQ tokenizer"
"${DL[@]}" \
  --repo_id peizesun/llamagen_t2i \
  --file vq_ds16_t2i.pt \
  --local_dir external/EditAR/pretrained_models

echo "[5/9] EditAR LlamaGen T2I pretrain"
"${DL[@]}" \
  --repo_id peizesun/llamagen_t2i \
  --file t2i_XL_stage2_512.pt \
  --local_dir external/EditAR/pretrained_models

echo "[6/9] EditAR flan-t5-xl text encoder"
"${DL[@]}" \
  --repo_id google/flan-t5-xl \
  --pattern config.json \
  --pattern generation_config.json \
  --pattern model*.safetensors \
  --pattern model.safetensors.index.json \
  --pattern special_tokens_map.json \
  --pattern spiece.model \
  --pattern tokenizer.json \
  --pattern tokenizer_config.json \
  --local_dir external/EditAR/pretrained_models/t5-ckpt/flan-t5-xl

echo "[7/9] Stage 2 Qwen-Image-Edit-2511 transformer and processor"
"${DL[@]}" \
  --repo_id Qwen/Qwen-Image-Edit-2511 \
  --pattern "transformer/*" \
  --pattern "processor/*" \
  --pattern model_index.json \
  --local_dir checkpoints/diffsynth/Qwen/Qwen-Image-Edit-2511

echo "[8/9] Stage 2 Qwen-Image text encoder, tokenizer, VAE"
"${DL[@]}" \
  --repo_id Qwen/Qwen-Image \
  --pattern "text_encoder/*" \
  --pattern "tokenizer/*" \
  --pattern "vae/*" \
  --local_dir checkpoints/diffsynth/Qwen/Qwen-Image

echo "[9/9] Stage 3 Qwen3-VL-8B-Instruct"
"${DL[@]}" \
  --repo_id Qwen/Qwen3-VL-8B-Instruct \
  --pattern "*" \
  --local_dir checkpoints/hf/Qwen3-VL-8B-Instruct

echo "[done] all requested upstream assets are present or downloading was skipped due to existing files"
