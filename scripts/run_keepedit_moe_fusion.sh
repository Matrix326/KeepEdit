#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hw4diff}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG_STAGE1="${CONFIG_STAGE1:-configs/experiments/keepedit_stage1_moe.yaml}"
LOG_DIR="${LOG_DIR:-reports/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TRAIN_REQUESTS="${TRAIN_REQUESTS:-data/processed/magicbrush_train/train.jsonl}"
DEV_REQUESTS="${DEV_REQUESTS:-data/processed/magicbrush_dev/dev.jsonl}"
TRAIN_CANDIDATES_DIR="${TRAIN_CANDIDATES_DIR:-data/candidates/magicbrush_train_pix2pix_qwen_editar}"
DEV_CANDIDATES_DIR="${DEV_CANDIDATES_DIR:-data/candidates/magicbrush_dev_pix2pix_qwen_editar}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-$TRAIN_CANDIDATES_DIR/predictions.jsonl}"
DEV_CANDIDATES="${DEV_CANDIDATES:-$DEV_CANDIDATES_DIR/predictions.jsonl}"
TRAIN_TEACHER_DIR="${TRAIN_TEACHER_DIR:-data/teachers/magicbrush_train_moe_fusion}"
DEV_TEACHER_DIR="${DEV_TEACHER_DIR:-data/teachers/magicbrush_dev_moe_fusion}"
STAGE1_GALLERY="${STAGE1_GALLERY:-reports/visual_gallery_magicbrush_dev_moe_fusion}"

EXPECTED_EXPERTS="${EXPECTED_EXPERTS:-pix2pix,qwen_image_edit,editar}"
ALLOW_PARTIAL_EXPERTS="${ALLOW_PARTIAL_EXPERTS:-0}"
STAGE1_PARALLEL_EXPERTS="${STAGE1_PARALLEL_EXPERTS:-0}"
STAGE1_EXPERT_GPU_GROUPS="${STAGE1_EXPERT_GPU_GROUPS:-pix2pix=0,1,2,3;qwen_image_edit=0,1,2,3;editar=0,1,2,3}"
STAGE1_LIMIT="${STAGE1_LIMIT:-${QWEN_LIMIT:-}}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-32}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stage1_moe_fusion_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "Stage 1 config: $CONFIG_STAGE1"
echo "GPUs: $GPUS"
echo "Expected experts: $EXPECTED_EXPERTS"
echo "Stage 1 CPU workers: $STAGE1_NUM_WORKERS"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    return 1
  fi
}

candidate_cache_complete() {
  local predictions="$1"
  local requests="$2"
  local expected="$3"
  [[ -s "$predictions" ]] || return 1
  conda run --no-capture-output -n "$ENV_NAME" python - "$predictions" "$requests" "$expected" <<'PY'
import json
import sys
from pathlib import Path

pred_path, req_path, expected_csv = sys.argv[1:4]
expected = {x.strip() for x in expected_csv.split(",") if x.strip()}

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

preds = read_jsonl(pred_path)
reqs = read_jsonl(req_path)
pred_by_id = {str(row.get("id")): row for row in preds}
if len(pred_by_id) < len(reqs):
    raise SystemExit(1)
for row in reqs:
    item = pred_by_id.get(str(row.get("id")))
    if not item:
        raise SystemExit(1)
    candidates = item.get("candidates") or []
    valid = {
        str(candidate.get("name"))
        for candidate in candidates
        if candidate.get("image_path") and Path(candidate["image_path"]).exists()
    }
    if not expected.issubset(valid):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

maybe_limit_args=()
if [[ -n "$STAGE1_LIMIT" ]]; then
  maybe_limit_args+=(--limit "$STAGE1_LIMIT")
fi

expert_args=()
if [[ "$STAGE1_PARALLEL_EXPERTS" == "1" || "$STAGE1_PARALLEL_EXPERTS" == "true" ]]; then
  expert_args+=(--parallel_experts)
fi
if [[ -n "$STAGE1_EXPERT_GPU_GROUPS" ]]; then
  expert_args+=(--expert_gpu_groups "$STAGE1_EXPERT_GPU_GROUPS")
fi

teacher_args=(--expected_experts "$EXPECTED_EXPERTS")
if [[ "$ALLOW_PARTIAL_EXPERTS" == "1" || "$ALLOW_PARTIAL_EXPERTS" == "true" ]]; then
  teacher_args+=(--allow_partial_experts)
fi

echo "[0/5] Check inputs"
require_file "$TRAIN_REQUESTS"
require_file "$DEV_REQUESTS"

echo "[1/5] Generate or reuse train expert candidates"
if ! candidate_cache_complete "$TRAIN_CANDIDATES" "$TRAIN_REQUESTS" "$EXPECTED_EXPERTS"; then
  conda run --no-capture-output -n "$ENV_NAME" python scripts/run_experts_by_expert_multi_gpu.py \
    --config "$CONFIG_STAGE1" \
    --requests "$TRAIN_REQUESTS" \
    --output_dir "$TRAIN_CANDIDATES_DIR" \
    --gpus "$GPUS" \
    "${expert_args[@]}" \
    "${maybe_limit_args[@]}"
else
  echo "Reuse complete train candidates: $TRAIN_CANDIDATES"
fi

echo "[2/5] Generate or reuse dev expert candidates"
if ! candidate_cache_complete "$DEV_CANDIDATES" "$DEV_REQUESTS" "$EXPECTED_EXPERTS"; then
  conda run --no-capture-output -n "$ENV_NAME" python scripts/run_experts_by_expert_multi_gpu.py \
    --config "$CONFIG_STAGE1" \
    --requests "$DEV_REQUESTS" \
    --output_dir "$DEV_CANDIDATES_DIR" \
    --gpus "$GPUS" \
    "${expert_args[@]}" \
    "${maybe_limit_args[@]}"
else
  echo "Reuse complete dev candidates: $DEV_CANDIDATES"
fi

echo "[3/5] Build train MoE-Fusion teacher"
conda run --no-capture-output -n "$ENV_NAME" python scripts/build_moe_fusion_teacher.py \
  --config "$CONFIG_STAGE1" \
  --requests "$TRAIN_REQUESTS" \
  --candidates_jsonl "$TRAIN_CANDIDATES" \
  --output_dir "$TRAIN_TEACHER_DIR" \
  --num_workers "$STAGE1_NUM_WORKERS" \
  "${teacher_args[@]}" \
  "${maybe_limit_args[@]}"

echo "[4/5] Build dev MoE-Fusion teacher, release metrics, gallery"
conda run --no-capture-output -n "$ENV_NAME" python scripts/build_moe_fusion_teacher.py \
  --config "$CONFIG_STAGE1" \
  --requests "$DEV_REQUESTS" \
  --candidates_jsonl "$DEV_CANDIDATES" \
  --output_dir "$DEV_TEACHER_DIR" \
  --num_workers "$STAGE1_NUM_WORKERS" \
  "${teacher_args[@]}" \
  "${maybe_limit_args[@]}"

conda run --no-capture-output -n "$ENV_NAME" python -m keepedit.evaluation.release_metrics \
  --predictions "$DEV_TEACHER_DIR/predictions.jsonl" \
  --out_csv reports/magicbrush_dev_moe_fusion_teacher_release_metrics.csv \
  --out_summary_json reports/magicbrush_dev_moe_fusion_teacher_release_metrics_summary.json

conda run --no-capture-output -n "$ENV_NAME" python scripts/make_visual_gallery.py \
  --predictions "$DEV_TEACHER_DIR/predictions.jsonl" \
  --out_dir "$STAGE1_GALLERY" \
  --limit 128

echo "[5/5] Stage 1 done"
echo "Train teacher: $TRAIN_TEACHER_DIR/predictions.jsonl"
echo "Dev teacher: $DEV_TEACHER_DIR/predictions.jsonl"
echo "Release metrics: reports/magicbrush_dev_moe_fusion_teacher_release_metrics.csv"
echo "Gallery: $STAGE1_GALLERY/index.html"
echo "Log file: $LOG_FILE"
