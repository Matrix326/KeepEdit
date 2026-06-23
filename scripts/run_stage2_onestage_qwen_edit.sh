#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hw4diff}"
GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
LOG_DIR="${LOG_DIR:-reports/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ONESTAGE_METADATA="${ONESTAGE_METADATA:?Set ONESTAGE_METADATA to a DiffSynth metadata.json}"
ONESTAGE_PHASE="${ONESTAGE_PHASE:-onestage}"
ONESTAGE_CKPT_DIR="${ONESTAGE_CKPT_DIR:?Set ONESTAGE_CKPT_DIR to the output checkpoint dir}"

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-external/DiffSynth-Studio}"
QWEN_MODEL_BASE="${QWEN_MODEL_BASE:-checkpoints/diffsynth}"
QWEN_EDIT_MODEL_ID="${QWEN_EDIT_MODEL_ID:-Qwen/Qwen-Image-Edit-2511}"
QWEN_TEXT_VAE_MODEL_ID="${QWEN_TEXT_VAE_MODEL_ID:-Qwen/Qwen-Image}"

QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-262144}"
QWEN_DATASET_REPEAT="${QWEN_DATASET_REPEAT:-1}"
QWEN_EPOCHS="${QWEN_EPOCHS:-2}"
QWEN_LR="${QWEN_LR:-1e-4}"
QWEN_RANK="${QWEN_RANK:-32}"
QWEN_GRAD_ACCUM="${QWEN_GRAD_ACCUM:-1}"
QWEN_SAVE_STEPS="${QWEN_SAVE_STEPS:-1000}"
QWEN_NUM_WORKERS="${QWEN_NUM_WORKERS:-8}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stage2_onestage_qwen_edit_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PWD/$DIFFSYNTH_ROOT:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/$QWEN_MODEL_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD=true

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

force_train_enabled() {
  [[ "$FORCE_TRAIN" == "1" || "$FORCE_TRAIN" == "true" ]]
}

echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "GPUs: $GPUS"
echo "Metadata: $ONESTAGE_METADATA"
echo "Phase: $ONESTAGE_PHASE"
echo "Output ckpt: $ONESTAGE_CKPT_DIR"
echo "Epochs: $QWEN_EPOCHS"

if [[ ! -s "$ONESTAGE_METADATA" ]]; then
  echo "Missing metadata: $ONESTAGE_METADATA" >&2
  exit 1
fi

echo "[1/3] Validate source-only metadata"
conda run --no-capture-output -n "$ENV_NAME" python scripts/validate_diffsynth_metadata.py \
  --metadata "$ONESTAGE_METADATA" \
  --min_input_only_fraction 1.0 \
  --min_teacher_guided_fraction 0.0 \
  --require_source_only_conditions \
  --require_mask_image \
  --allowed_phase "$ONESTAGE_PHASE"

echo "[2/3] Train one-stage Qwen-Image-Edit LoRA from base"
if force_train_enabled || ! ckpt_complete "$ONESTAGE_CKPT_DIR"; then
  conda run --no-capture-output -n "$ENV_NAME" accelerate launch \
    --num_processes "$NUM_PROCESSES" \
    "$DIFFSYNTH_ROOT/examples/qwen_image/model_training/train.py" \
    --dataset_base_path / \
    --dataset_metadata_path "$ONESTAGE_METADATA" \
    --data_file_keys "image,edit_image,mask_image" \
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
    --output_path "$ONESTAGE_CKPT_DIR" \
    --lora_base_model "dit" \
    --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
    --lora_rank "$QWEN_RANK" \
    --use_gradient_checkpointing \
    --dataset_num_workers "$QWEN_NUM_WORKERS" \
    --find_unused_parameters \
    --zero_cond_t
else
  echo "Reuse existing one-stage LoRA: $(latest_lora "$ONESTAGE_CKPT_DIR")"
fi

echo "[3/3] One-stage training done"
echo "LoRA: $(latest_lora "$ONESTAGE_CKPT_DIR")"
echo "Log file: $LOG_FILE"
