from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from keepedit.image_ops import load_mask
from keepedit.io import ensure_dir, read_jsonl


DEFAULT_MAX_SIDE = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute release metrics for image editing: Target/Output SSIM+PSNR, "
            "BG-SSIM, Input/Output SSIM+PSNR, Edit-Region Change, plus optional MLLM scores."
        )
    )
    parser.add_argument("--predictions", required=True, help="Prediction JSONL.")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary_json")
    parser.add_argument("--mllm_jsonl", help="Optional Qwen3-VL/MLLM preference JSONL to merge by id.")
    parser.add_argument("--max_side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _fit_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return width, height
    scale = max_side / float(longest)
    return max(8, round(width * scale)), max(8, round(height * scale))


def _load_rgb(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    min_side = min(a.shape[:2])
    if min_side < 7:
        mse = float(np.mean((a - b) ** 2))
        return float(max(0.0, 1.0 - mse))
    win_size = min(7, min_side if min_side % 2 else min_side - 1)
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0, win_size=win_size))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def _masked_crop(a: np.ndarray, b: np.ndarray, mask: np.ndarray, fill: str) -> tuple[np.ndarray, np.ndarray]:
    hard = mask > 0.5
    if hard.sum() <= 4:
        return a, b
    if fill == "black":
        return a * hard[..., None], b * hard[..., None]
    # For background SSIM, keep the common foreground white so the score is
    # dominated by non-edit regions rather than by the masked-out area.
    inv = (~hard)[..., None]
    return a * inv + (1.0 - inv), b * inv + (1.0 - inv)


def _edit_region_change(input_image: np.ndarray, output_image: np.ndarray, mask: np.ndarray) -> float:
    hard = mask > 0.5
    diff = np.abs(output_image - input_image).mean(axis=2)
    if hard.sum() <= 4:
        return float(diff.mean())
    return float(diff[hard].mean())


def _mllm_index(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = row.get("id")
        if row_id is not None:
            mapping[str(row_id)] = row
    return mapping


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _merge_mllm(row: dict[str, Any], mllm: dict[str, Any] | None) -> None:
    if not mllm:
        return
    mapping = {
        "preference_score": "mllm_preference_score",
        "instruction_correctness": "mllm_instruction_correctness",
        "background_preservation": "mllm_background_preservation",
        "target_similarity": "mllm_target_similarity",
    }
    for src, dst in mapping.items():
        value = _to_float(mllm.get(src))
        if value is not None:
            row[dst] = value
    if mllm.get("reason"):
        row["mllm_reason"] = mllm.get("reason")


def compute_one(item: dict[str, Any], max_side: int) -> dict[str, Any]:
    input_path = item["input_image"]
    output_path = item.get("output_image") or item.get("prediction") or item.get("edited_image")
    target_path = item.get("target_image")
    mask_path = item.get("mask_image")
    if not output_path:
        raise ValueError(f"Missing output image for row {item.get('id')}")
    if not target_path:
        raise ValueError(f"Missing target image for row {item.get('id')}")

    with Image.open(input_path) as image:
        size = _fit_size(image.size, max_side)
    input_image = _load_rgb(input_path, size)
    output_image = _load_rgb(output_path, size)
    target_image = _load_rgb(target_path, size)
    mask = load_mask(mask_path, size) if mask_path else np.ones(input_image.shape[:2], dtype=np.float32)

    bg_input, bg_output = _masked_crop(input_image, output_image, mask, fill="white")
    row = {
        "id": item.get("id"),
        "instruction": item.get("instruction", ""),
        "input_image": input_path,
        "output_image": output_path,
        "target_image": target_path,
        "mask_image": mask_path or "",
        "target_output_ssim": _safe_ssim(target_image, output_image),
        "target_output_psnr": _psnr(target_image, output_image),
        "bg_ssim": _safe_ssim(bg_input, bg_output),
        "input_output_ssim": _safe_ssim(input_image, output_image),
        "input_output_psnr": _psnr(input_image, output_image),
        "edit_region_change": _edit_region_change(input_image, output_image, mask),
    }
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(rows)}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"median_{key}"] = float(np.median(values))
    return summary


def main() -> None:
    args = parse_args()
    items = read_jsonl(args.predictions)
    if args.limit:
        items = items[: args.limit]
    mllm_rows = _mllm_index(args.mllm_jsonl)
    rows: list[dict[str, Any]] = []
    for item in tqdm(items, desc="release-metrics"):
        row = compute_one(item, max_side=args.max_side)
        _merge_mllm(row, mllm_rows.get(str(row.get("id"))))
        rows.append(row)

    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)
    fieldnames = [
        "id",
        "instruction",
        "target_output_ssim",
        "target_output_psnr",
        "bg_ssim",
        "input_output_ssim",
        "input_output_psnr",
        "edit_region_change",
        "mllm_preference_score",
        "mllm_instruction_correctness",
        "mllm_background_preservation",
        "mllm_target_similarity",
        "mllm_reason",
        "input_image",
        "output_image",
        "target_image",
        "mask_image",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} release metric rows to {out_csv}")

    if args.out_summary_json:
        summary = summarize(rows)
        out_json = Path(args.out_summary_json)
        ensure_dir(out_json.parent)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote release metric summary to {out_json}")


if __name__ == "__main__":
    main()
