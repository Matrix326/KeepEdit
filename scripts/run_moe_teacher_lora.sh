#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hw4diff}"
RUN_ID="${RUN_ID:-moe_teacher_lora_$(date +%Y%m%d_%H%M%S)}"
TRAIN_REQUESTS="${TRAIN_REQUESTS:-data/processed/magicbrush_train/train.jsonl}"
TEACHER_JSONL="${TEACHER_JSONL:-data/teachers/magicbrush_train_moe_fusion/predictions.jsonl}"
DATASET_DIR="${DATASET_DIR:-data/diffsynth/magicbrush_train_qwen2511_moe_teacher_onestage}"
METADATA="${METADATA:-$DATASET_DIR/metadata.json}"
CKPT_DIR="${CKPT_DIR:-checkpoints/qwen_edit_2511_moe_teacher_onestage}"
GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
RUN_EVAL="${RUN_EVAL:-1}"

echo "Run MoE-Teacher LoRA Qwen2511"
echo "Run ID: $RUN_ID"
echo "Teacher: $TEACHER_JSONL"
echo "Metadata: $METADATA"
echo "Checkpoint: $CKPT_DIR"

conda run --no-capture-output -n "$ENV_NAME" python scripts/prepare_qwen_lora_metadata.py \
  --jsonl "$TRAIN_REQUESTS" \
  --teacher_jsonl "$TEACHER_JSONL" \
  --out_dir "$DATASET_DIR" \
  --target_mode moe_teacher

ONESTAGE_METADATA="$METADATA" \
ONESTAGE_PHASE=moe_teacher_onestage \
ONESTAGE_CKPT_DIR="$CKPT_DIR" \
RUN_ID="$RUN_ID" \
GPUS="$GPUS" \
NUM_PROCESSES="$NUM_PROCESSES" \
QWEN_EPOCHS="${QWEN_EPOCHS:-1}" \
QWEN_LR="${QWEN_LR:-5e-5}" \
QWEN_RANK="${QWEN_RANK:-32}" \
bash scripts/run_stage2_onestage_qwen_edit.sh

if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" ]]; then
  EXPERIMENT_NAME=qwen2511_moe_teacher_onestage \
  LORA_PATH="$CKPT_DIR" \
  RUN_ID="${RUN_ID}_eval" \
  GPUS="${EVAL_GPUS:-0}" \
  PARALLEL_GPUS="${PARALLEL_GPUS:-}" \
  bash scripts/evaluate_qwen_edit_experiment.sh
fi
