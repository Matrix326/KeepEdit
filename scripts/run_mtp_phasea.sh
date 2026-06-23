#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hw4diff}"
RUN_ID="${RUN_ID:-mtp_phasea_$(date +%Y%m%d_%H%M%S)}"
TRAIN_REQUESTS="${TRAIN_REQUESTS:-data/processed/magicbrush_train/train.jsonl}"
DATASET_DIR="${DATASET_DIR:-data/diffsynth/magicbrush_train_mtp_phasea}"
CKPT_DIR="${CKPT_DIR:-checkpoints/qwen_edit_2511_mtp_phasea}"
GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
RUN_EVAL="${RUN_EVAL:-1}"

echo "Run MTP-PhaseA Qwen2511 LoRA"
echo "Run ID: $RUN_ID"
echo "Dataset: $DATASET_DIR"
echo "Checkpoint: $CKPT_DIR"

TRAIN_REQUESTS="$TRAIN_REQUESTS" \
QWEN_DATASET_DIR="$DATASET_DIR" \
CKPT_DIR="$CKPT_DIR" \
RUN_ID="$RUN_ID" \
GPUS="$GPUS" \
NUM_PROCESSES="$NUM_PROCESSES" \
QWEN_EPOCHS="${QWEN_EPOCHS:-1}" \
QWEN_LR="${QWEN_LR:-5e-5}" \
QWEN_RANK="${QWEN_RANK:-16}" \
QWEN_SAVE_STEPS="${QWEN_SAVE_STEPS:-500}" \
MASK_EDIT_WEIGHT="${MASK_EDIT_WEIGHT:-4.0}" \
MASK_BG_WEIGHT="${MASK_BG_WEIGHT:-0.3}" \
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.15}" \
NOOP_FRACTION="${NOOP_FRACTION:-0.03}" \
NOOP_WEIGHT="${NOOP_WEIGHT:-0.03}" \
SOFT_DILATE_RADIUS="${SOFT_DILATE_RADIUS:-24}" \
SOFT_BLUR_SIGMA="${SOFT_BLUR_SIGMA:-7.0}" \
bash scripts/run_mtp_lora_qwen_edit.sh

if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" ]]; then
  EXPERIMENT_NAME=qwen2511_mtp_phasea \
  LORA_PATH="$CKPT_DIR" \
  RUN_ID="${RUN_ID}_eval" \
  GPUS="${EVAL_GPUS:-0}" \
  PARALLEL_GPUS="${PARALLEL_GPUS:-}" \
  QWEN_INFER_STEPS="${QWEN_INFER_STEPS:-28}" \
  bash scripts/evaluate_qwen_edit_experiment.sh
fi
