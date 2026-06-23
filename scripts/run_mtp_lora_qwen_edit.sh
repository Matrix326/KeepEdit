#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hw4diff}"
GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
LOG_DIR="${LOG_DIR:-reports/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TRAIN_REQUESTS="${TRAIN_REQUESTS:-data/processed/magicbrush_train/train.jsonl}"
QWEN_DATASET_DIR="${QWEN_DATASET_DIR:-data/diffsynth/magicbrush_train_mtp_lora}"
METADATA="${METADATA:-$QWEN_DATASET_DIR/metadata.json}"

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-external/DiffSynth-Studio}"
QWEN_MODEL_BASE="${QWEN_MODEL_BASE:-checkpoints/diffsynth}"
QWEN_EDIT_MODEL_ID="${QWEN_EDIT_MODEL_ID:-Qwen/Qwen-Image-Edit-2511}"
QWEN_TEXT_VAE_MODEL_ID="${QWEN_TEXT_VAE_MODEL_ID:-Qwen/Qwen-Image}"
CKPT_DIR="${CKPT_DIR:-checkpoints/qwen_edit_2511_mtp_lora}"

QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-262144}"
QWEN_DATASET_REPEAT="${QWEN_DATASET_REPEAT:-1}"
QWEN_EPOCHS="${QWEN_EPOCHS:-1}"
QWEN_LR="${QWEN_LR:-5e-5}"
QWEN_RANK="${QWEN_RANK:-16}"
QWEN_GRAD_ACCUM="${QWEN_GRAD_ACCUM:-1}"
QWEN_SAVE_STEPS="${QWEN_SAVE_STEPS:-1000}"
QWEN_LIMIT="${QWEN_LIMIT:-}"
QWEN_NUM_WORKERS="${QWEN_NUM_WORKERS:-8}"
MTP_METADATA_WORKERS="${MTP_METADATA_WORKERS:-0}"
MTP_METADATA_CHUNKSIZE="${MTP_METADATA_CHUNKSIZE:-8}"
MASK_EDIT_WEIGHT="${MASK_EDIT_WEIGHT:-1.5}"
MASK_BG_WEIGHT="${MASK_BG_WEIGHT:-1.0}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.3}"
SOFT_DILATE_RADIUS="${SOFT_DILATE_RADIUS:-8}"
SOFT_BLUR_SIGMA="${SOFT_BLUR_SIGMA:-5.0}"
NOOP_FRACTION="${NOOP_FRACTION:-0.10}"
NOOP_WEIGHT="${NOOP_WEIGHT:-0.10}"
BG_PRESERVATION_WEIGHT="${BG_PRESERVATION_WEIGHT:-0.0}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/mtp_lora_qwen_edit_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PWD/src:$PWD/$DIFFSYNTH_ROOT:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/$QWEN_MODEL_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_LOG_LOSS_STEPS="${DIFFSYNTH_LOG_LOSS_STEPS:-50}"

echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "Algorithm: MTP-LoRA = masked clean target + boundary-weighted source-only QwenEdit LoRA"
echo "Train requests: $TRAIN_REQUESTS"
echo "Metadata: $METADATA"
echo "Checkpoint dir: $CKPT_DIR"
echo "GPUs: $GPUS num_processes=$NUM_PROCESSES"
echo "rank=$QWEN_RANK lr=$QWEN_LR epochs=$QWEN_EPOCHS limit=${QWEN_LIMIT:-none}"
echo "weights: edit=$MASK_EDIT_WEIGHT bg=$MASK_BG_WEIGHT boundary=$BOUNDARY_WEIGHT noop_fraction=$NOOP_FRACTION noop_weight=$NOOP_WEIGHT"
echo "soft_dilate_radius=$SOFT_DILATE_RADIUS soft_blur_sigma=$SOFT_BLUR_SIGMA metadata_workers=$MTP_METADATA_WORKERS lora_checkpoint=${LORA_CHECKPOINT:-none}"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    return 1
  fi
}

latest_lora() {
  conda run --no-capture-output -n "$ENV_NAME" python - "$1" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
files = []
if root.is_file() and root.suffix == ".safetensors":
    files = [root]
elif root.exists():
    files = sorted(root.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not files:
        files = sorted(root.glob("**/*.safetensors"), key=lambda p: p.stat().st_mtime)
if not files:
    raise SystemExit(f"No .safetensors found under {root}")
print(files[-1])
PY
}

ckpt_complete() {
  latest_lora "$1" >/dev/null 2>&1
}

prep_args=(
  scripts/prepare_mtp_lora_metadata.py
  --jsonl "$TRAIN_REQUESTS"
  --out_dir "$QWEN_DATASET_DIR"
  --mask_edit_weight "$MASK_EDIT_WEIGHT"
  --mask_bg_weight "$MASK_BG_WEIGHT"
  --boundary_weight "$BOUNDARY_WEIGHT"
  --soft_dilate_radius "$SOFT_DILATE_RADIUS"
  --soft_blur_sigma "$SOFT_BLUR_SIGMA"
  --noop_fraction "$NOOP_FRACTION"
  --noop_weight "$NOOP_WEIGHT"
  --bg_preservation_weight "$BG_PRESERVATION_WEIGHT"
  --num_workers "$MTP_METADATA_WORKERS"
  --chunksize "$MTP_METADATA_CHUNKSIZE"
)
if [[ -n "$QWEN_LIMIT" ]]; then
  prep_args+=(--limit "$QWEN_LIMIT")
fi

echo "[1/4] Prepare MTP clean-target metadata"
conda run --no-capture-output -n "$ENV_NAME" python "${prep_args[@]}"
require_file "$METADATA"

echo "[2/4] Validate source-only metadata"
conda run --no-capture-output -n "$ENV_NAME" python scripts/validate_diffsynth_metadata.py \
  --metadata "$METADATA" \
  --min_input_only_fraction 1.0 \
  --min_teacher_guided_fraction 0.0 \
  --require_source_only_conditions \
  --require_mask_image \
  --allow_identity_targets \
  --allowed_phase mtp_sft \
  --allowed_phase noop_preservation

echo "[3/4] Train MTP-LoRA"
if [[ "$FORCE_TRAIN" == "1" || "$FORCE_TRAIN" == "true" ]] || ! ckpt_complete "$CKPT_DIR"; then
  lora_args=()
  if [[ -n "$LORA_CHECKPOINT" ]]; then
    require_file "$LORA_CHECKPOINT"
    lora_args+=(--lora_checkpoint "$LORA_CHECKPOINT")
  fi
  conda run --no-capture-output -n "$ENV_NAME" accelerate launch \
    --num_processes "$NUM_PROCESSES" \
    "$DIFFSYNTH_ROOT/examples/qwen_image/model_training/train.py" \
    --dataset_base_path / \
    --dataset_metadata_path "$METADATA" \
    --data_file_keys "image,edit_image,mask_image,boundary_image" \
    --extra_inputs "edit_image" \
    --max_pixels "$QWEN_MAX_PIXELS" \
    --dataset_repeat "$QWEN_DATASET_REPEAT" \
    --model_id_with_origin_paths "$QWEN_EDIT_MODEL_ID:transformer/diffusion_pytorch_model*.safetensors,$QWEN_TEXT_VAE_MODEL_ID:text_encoder/model*.safetensors,$QWEN_TEXT_VAE_MODEL_ID:vae/diffusion_pytorch_model.safetensors" \
    --tokenizer_path "$PWD/$QWEN_MODEL_BASE/$QWEN_TEXT_VAE_MODEL_ID/tokenizer" \
    --processor_path "$PWD/$QWEN_MODEL_BASE/$QWEN_EDIT_MODEL_ID/processor" \
    --learning_rate "$QWEN_LR" \
    --num_epochs "$QWEN_EPOCHS" \
    --gradient_accumulation_steps "$QWEN_GRAD_ACCUM" \
    --save_steps "$QWEN_SAVE_STEPS" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$CKPT_DIR" \
    --lora_base_model "dit" \
    --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
    --lora_rank "$QWEN_RANK" \
    "${lora_args[@]}" \
    --use_gradient_checkpointing \
    --dataset_num_workers "$QWEN_NUM_WORKERS" \
    --find_unused_parameters \
    --task "sft" \
    --zero_cond_t
else
  echo "Reuse existing MTP-LoRA: $(latest_lora "$CKPT_DIR")"
fi

echo "[4/4] Done"
echo "Final LoRA: $(latest_lora "$CKPT_DIR")"
echo "Metadata summary: $QWEN_DATASET_DIR/summary.json"
echo "Log file: $LOG_FILE"
