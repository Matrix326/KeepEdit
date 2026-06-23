#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hw4diff}"
GPUS="${GPUS:-0}"
CONFIG_STAGE2="${CONFIG_STAGE2:-configs/experiments/keepedit_stage2_qwen_edit_lora.yaml}"
LOG_DIR="${LOG_DIR:-reports/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME, e.g. qwen2511_base}"
DEV_REQUESTS="${DEV_REQUESTS:-data/processed/magicbrush_dev/dev.jsonl}"
LORA_PATH="${LORA_PATH:-none}"
OUT_DIR="${OUT_DIR:-data/outputs/magicbrush_dev_${EXPERIMENT_NAME}}"
GALLERY_DIR="${GALLERY_DIR:-reports/visual_gallery_magicbrush_dev_${EXPERIMENT_NAME}}"
RELEASE_METRICS_CSV="${RELEASE_METRICS_CSV:-reports/magicbrush_dev_${EXPERIMENT_NAME}_release_metrics.csv}"
RELEASE_SUMMARY_JSON="${RELEASE_SUMMARY_JSON:-reports/magicbrush_dev_${EXPERIMENT_NAME}_release_metrics_summary.json}"
PREFERENCE_JSONL="${PREFERENCE_JSONL:-reports/magicbrush_dev_${EXPERIMENT_NAME}_mllm_preference.jsonl}"

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-external/DiffSynth-Studio}"
QWEN_MODEL_BASE="${QWEN_MODEL_BASE:-checkpoints/diffsynth}"
QWEN_EDIT_MODEL_ID="${QWEN_EDIT_MODEL_ID:-Qwen/Qwen-Image-Edit-2511}"
QWEN_TEXT_VAE_MODEL_ID="${QWEN_TEXT_VAE_MODEL_ID:-Qwen/Qwen-Image}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-262144}"
QWEN_INFER_STEPS="${QWEN_INFER_STEPS:-40}"
QWEN_CFG_SCALE="${QWEN_CFG_SCALE:-4.0}"
QWEN_DENOISING_STRENGTH="${QWEN_DENOISING_STRENGTH:-0.9}"
QWEN_LIMIT="${QWEN_LIMIT:-}"
PARALLEL_GPUS="${PARALLEL_GPUS:-}"
SHARD_DIR="${SHARD_DIR:-$OUT_DIR/_eval_shards_${RUN_ID}}"
RUN_MLLM="${RUN_MLLM:-0}"
MLLM_COMMAND="${MLLM_COMMAND:-}"
MLLM_BACKEND="${MLLM_BACKEND:-qwen3_vl}"
MLLM_MODEL_PATH="${MLLM_MODEL_PATH:-checkpoints/hf/Qwen3-VL-8B-Instruct}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/evaluate_qwen_edit_${EXPERIMENT_NAME}_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$PWD/src:$PWD/$DIFFSYNTH_ROOT:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/$QWEN_MODEL_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD=true

echo "Run ID: $RUN_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Log file: $LOG_FILE"
echo "GPUs: $GPUS"
echo "Parallel GPUs: ${PARALLEL_GPUS:-disabled}"
echo "LoRA path: $LORA_PATH"
echo "Output dir: $OUT_DIR"
echo "Denoising strength: $QWEN_DENOISING_STRENGTH"

maybe_limit_args=()
if [[ -n "$QWEN_LIMIT" ]]; then
  maybe_limit_args+=(--limit "$QWEN_LIMIT")
fi

lora_args=()
if [[ "$LORA_PATH" != "none" && "$LORA_PATH" != "no_lora" && "$LORA_PATH" != "base" && "$LORA_PATH" != "raw" ]]; then
  lora_args+=(--lora_path "$LORA_PATH")
fi

echo "[1/5] QwenEdit inference"
if [[ -n "$PARALLEL_GPUS" ]]; then
  mkdir -p "$SHARD_DIR"
  conda run --no-capture-output -n "$ENV_NAME" python - "$DEV_REQUESTS" "$SHARD_DIR" "$PARALLEL_GPUS" "${QWEN_LIMIT:-}" <<'PY'
import json
import sys
from pathlib import Path

requests = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
gpus = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
limit = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
rows = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines() if line.strip()]
if limit is not None:
    rows = rows[:limit]
for idx, row in enumerate(rows):
    row.setdefault("metadata", {})
    row["metadata"]["keepedit_eval_order"] = idx
for shard_idx, _gpu in enumerate(gpus):
    shard_rows = [row for idx, row in enumerate(rows) if idx % len(gpus) == shard_idx]
    path = shard_dir / f"requests_{shard_idx}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in shard_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Shard {shard_idx}: {len(shard_rows)} rows -> {path}")
print(f"Total rows: {len(rows)} across {len(gpus)} shards")
PY

  IFS=',' read -r -a gpu_array <<< "$PARALLEL_GPUS"
  pids=()
  shard_index=0
  for gpu in "${gpu_array[@]}"; do
    gpu="$(echo "$gpu" | xargs)"
    [[ -z "$gpu" ]] && continue
    shard_requests="$SHARD_DIR/requests_${shard_index}.jsonl"
    shard_out="$SHARD_DIR/output_${shard_index}"
    shard_log="$LOG_DIR/evaluate_qwen_edit_${EXPERIMENT_NAME}_${RUN_ID}_shard${shard_index}_gpu${gpu}.log"
    echo "Start shard $shard_index on GPU $gpu: $shard_requests -> $shard_out"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      conda run --no-capture-output -n "$ENV_NAME" python -m keepedit.pipelines.run_qwen_edit_lora \
        --config "$CONFIG_STAGE2" \
        --requests "$shard_requests" \
        --diffsynth_root "$DIFFSYNTH_ROOT" \
        --model_base "$QWEN_MODEL_BASE" \
        --edit_model_id "$QWEN_EDIT_MODEL_ID" \
        --text_vae_model_id "$QWEN_TEXT_VAE_MODEL_ID" \
        --condition_mode input_only \
        --output_dir "$shard_out" \
        --num_inference_steps "$QWEN_INFER_STEPS" \
        --cfg_scale "$QWEN_CFG_SCALE" \
        --denoising_strength "$QWEN_DENOISING_STRENGTH" \
        --max_pixels "$QWEN_MAX_PIXELS" \
        --no_background_compose \
        "${lora_args[@]}"
    ) > >(tee -a "$shard_log") 2>&1 &
    pids+=("$!")
    shard_index=$((shard_index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  conda run --no-capture-output -n "$ENV_NAME" python - "$OUT_DIR" "$SHARD_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
merged = []
for pred_path in sorted(shard_dir.glob("output_*/predictions.jsonl")):
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        order = (row.get("metadata") or {}).get("keepedit_eval_order")
        if order is None:
            order = len(merged)
        row["_merge_order"] = int(order)
        merged.append(row)
for row in merged:
    row.pop("_merge_order", None)
merged.sort(key=lambda row: int((row.get("metadata") or {}).get("keepedit_eval_order", 0)))

for subdir in ["images", "raw", "masks"]:
    target_dir = out_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(shard_dir.glob(f"output_*/{subdir}/*")):
        if src.is_file():
            dst = target_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)

predictions = out_dir / "predictions.jsonl"
with predictions.open("w", encoding="utf-8") as handle:
    for row in merged:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Merged {len(merged)} predictions -> {predictions}")
PY
else
  conda run --no-capture-output -n "$ENV_NAME" python -m keepedit.pipelines.run_qwen_edit_lora \
    --config "$CONFIG_STAGE2" \
    --requests "$DEV_REQUESTS" \
    --diffsynth_root "$DIFFSYNTH_ROOT" \
    --model_base "$QWEN_MODEL_BASE" \
    --edit_model_id "$QWEN_EDIT_MODEL_ID" \
    --text_vae_model_id "$QWEN_TEXT_VAE_MODEL_ID" \
    --condition_mode input_only \
    --output_dir "$OUT_DIR" \
    --num_inference_steps "$QWEN_INFER_STEPS" \
    --cfg_scale "$QWEN_CFG_SCALE" \
    --denoising_strength "$QWEN_DENOISING_STRENGTH" \
    --max_pixels "$QWEN_MAX_PIXELS" \
    --no_background_compose \
    "${lora_args[@]}" \
    "${maybe_limit_args[@]}"
fi

echo "[2/5] Gallery"
conda run --no-capture-output -n "$ENV_NAME" python scripts/make_visual_gallery.py \
  --predictions "$OUT_DIR/predictions.jsonl" \
  --out_dir "$GALLERY_DIR" \
  --limit 128

echo "[3/5] MLLM preference"
if [[ "$RUN_MLLM" == "1" || "$RUN_MLLM" == "true" ]]; then
  pref_args=(
    python -m keepedit.evaluation.mllm_preference
    --predictions "$OUT_DIR/predictions.jsonl"
    --out_jsonl "$PREFERENCE_JSONL"
    --backend "$MLLM_BACKEND"
    --model_path "$MLLM_MODEL_PATH"
  )
  if [[ -n "$MLLM_COMMAND" ]]; then
    pref_args+=(--command "$MLLM_COMMAND")
  fi
  conda run --no-capture-output -n "$ENV_NAME" "${pref_args[@]}"
else
  echo "Skip MLLM preference because RUN_MLLM=$RUN_MLLM"
fi

echo "[4/5] Release metrics"
release_args=(
  python -m keepedit.evaluation.release_metrics
  --predictions "$OUT_DIR/predictions.jsonl"
  --out_csv "$RELEASE_METRICS_CSV"
  --out_summary_json "$RELEASE_SUMMARY_JSON"
)
if [[ -s "$PREFERENCE_JSONL" ]]; then
  release_args+=(--mllm_jsonl "$PREFERENCE_JSONL")
fi
conda run --no-capture-output -n "$ENV_NAME" "${release_args[@]}"

echo "[5/5] Done"
echo "Predictions: $OUT_DIR/predictions.jsonl"
echo "Release metrics: $RELEASE_METRICS_CSV"
echo "Release summary: $RELEASE_SUMMARY_JSON"
echo "Gallery: $GALLERY_DIR/index.html"
echo "Preference: $PREFERENCE_JSONL"
echo "Log file: $LOG_FILE"
