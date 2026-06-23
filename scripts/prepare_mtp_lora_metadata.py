#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image
from skimage import filters, morphology
from tqdm import tqdm

from keepedit.image_ops import (
    diff_mask,
    dilate_mask,
    edit_band,
    feather_mask,
    float_to_image,
    image_to_float,
    load_mask,
    load_rgb,
    save_image,
    save_mask,
)
from keepedit.io import ensure_dir, read_jsonl
from keepedit.schemas import EditRequest


GLOBAL_PROMPT_TERMS = (
    "change the style",
    "make it look like",
    "turn the image into",
    "change the lighting",
    "make it night",
    "make it snowy",
    "change the background",
    "make the background",
    "in the style of",
    "as a painting",
    "make it a cartoon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MTP-LoRA clean-target metadata for Qwen-Image-Edit-2511 SFT.")
    parser.add_argument("--jsonl", required=True, help="MagicBrush-style JSONL with input_image, instruction, target_image, optional mask_image.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metadata_name", default="metadata.json")
    parser.add_argument("--clean_dirname", default="clean_targets")
    parser.add_argument("--mask_dirname", default="mtp_masks")
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel image-processing workers. 0/1 keeps the sequential path.")
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--soft_dilate_radius", type=int, default=8)
    parser.add_argument("--soft_blur_sigma", type=float, default=5.0)
    parser.add_argument("--boundary_radius", type=int, default=6)
    parser.add_argument("--global_area_threshold", type=float, default=0.50)
    parser.add_argument("--fallback_diff_threshold", type=float, default=0.06)
    parser.add_argument("--mask_min_area_ratio", type=float, default=0.001)
    parser.add_argument("--mask_edit_weight", type=float, default=1.5)
    parser.add_argument("--mask_bg_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=0.3)
    parser.add_argument("--noop_fraction", type=float, default=0.10)
    parser.add_argument("--noop_weight", type=float, default=0.10)
    parser.add_argument("--bg_preservation_weight", type=float, default=0.0)
    parser.add_argument(
        "--prompt_suffix",
        default=" Apply the requested edit to the original image. Preserve all regions not mentioned by the instruction.",
    )
    return parser.parse_args()


def is_global_prompt(instruction: str) -> bool:
    text = instruction.lower()
    return any(term in text for term in GLOBAL_PROMPT_TERMS)


def clean_hard_mask(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    hard = mask > 0.5
    min_area = max(4, int(hard.size * min_area_ratio))
    hard = morphology.remove_small_objects(hard, min_size=min_area)
    hard = morphology.remove_small_holes(hard, area_threshold=min_area)
    hard = morphology.binary_closing(hard, morphology.disk(2))
    hard = morphology.binary_dilation(hard, morphology.disk(3))
    return hard.astype(np.float32)


class MaskCandidate(NamedTuple):
    name: str
    mask: np.ndarray
    score: float
    area: float
    inside_diff: float
    outside_diff: float
    diff_coverage: float
    local_allowed: bool


def source_target_diff_map(source: Image.Image, target: Image.Image) -> np.ndarray:
    target = target.resize(source.size, Image.Resampling.LANCZOS)
    return np.abs(image_to_float(source) - image_to_float(target)).mean(axis=2).astype(np.float32)


def diff_mask_otsu(source: Image.Image, target: Image.Image, threshold: float, min_area_ratio: float) -> np.ndarray:
    diff_rgb = source_target_diff_map(source, target)
    try:
        tau = float(filters.threshold_otsu(diff_rgb))
    except Exception:
        tau = float(np.quantile(diff_rgb, 0.85))
    tau = max(min(tau, float(np.quantile(diff_rgb, 0.95))), threshold * 0.5)
    mask = (diff_rgb > tau).astype(np.float32)
    if mask.mean() < min_area_ratio:
        mask = diff_mask(source, target, threshold=threshold, min_area_ratio=min_area_ratio)
    return clean_hard_mask(mask, min_area_ratio=min_area_ratio)


def score_mask_candidate(
    name: str,
    mask: np.ndarray,
    diff_rgb: np.ndarray,
    min_area_ratio: float,
    global_area_threshold: float,
    global_prompt: bool,
) -> MaskCandidate:
    hard = clean_hard_mask(mask, min_area_ratio=min_area_ratio) > 0.5
    area = float(hard.mean())
    total_diff = float(diff_rgb.sum()) + 1e-8
    if area <= 0.0:
        return MaskCandidate(name, hard.astype(np.float32), -1e6, 0.0, 0.0, 0.0, 0.0, False)

    inside = float(diff_rgb[hard].mean()) if hard.any() else 0.0
    outside_mask = ~hard
    outside = float(diff_rgb[outside_mask].mean()) if outside_mask.any() else 0.0
    coverage = float(diff_rgb[hard].sum() / total_diff)

    # MTP is designed to keep non-edited regions anchored to source. A very
    # broad mask is only trusted for explicit global edits; otherwise it often
    # means the dataset mask polarity or extent is unsuitable for source
    # preservation training.
    local_allowed = global_prompt or area <= max(global_area_threshold, 0.55)
    broad_penalty = max(0.0, area - 0.45)
    tiny_penalty = max(0.0, min_area_ratio * 4.0 - area) * 10.0
    local_penalty = 0.0 if local_allowed else 0.50 + broad_penalty
    score = (inside - outside) + 0.08 * coverage - 0.18 * broad_penalty - tiny_penalty - local_penalty
    return MaskCandidate(name, hard.astype(np.float32), float(score), area, inside, outside, coverage, local_allowed)


def candidate_to_dict(candidate: MaskCandidate) -> dict[str, Any]:
    return {
        "score": round(candidate.score, 6),
        "area": round(candidate.area, 6),
        "inside_diff": round(candidate.inside_diff, 6),
        "outside_diff": round(candidate.outside_diff, 6),
        "diff_coverage": round(candidate.diff_coverage, 6),
        "local_allowed": candidate.local_allowed,
    }


def build_mask(request: EditRequest, source: Image.Image, target: Image.Image, args: argparse.Namespace) -> tuple[np.ndarray, str, dict[str, Any]]:
    diff_rgb = source_target_diff_map(source, target)
    global_prompt = is_global_prompt(request.instruction)
    candidates: list[MaskCandidate] = []
    diff_candidate = diff_mask_otsu(
        source,
        target,
        threshold=args.fallback_diff_threshold,
        min_area_ratio=args.mask_min_area_ratio,
    )
    candidates.append(
        score_mask_candidate(
            "source_target_diff_otsu",
            diff_candidate,
            diff_rgb,
            args.mask_min_area_ratio,
            args.global_area_threshold,
            global_prompt,
        )
    )
    if request.mask_image and Path(request.mask_image).exists():
        dataset_mask = clean_hard_mask(load_mask(request.mask_image, source.size), min_area_ratio=args.mask_min_area_ratio)
        if dataset_mask.mean() > 0:
            candidates.append(
                score_mask_candidate(
                    "dataset_mask",
                    dataset_mask,
                    diff_rgb,
                    args.mask_min_area_ratio,
                    args.global_area_threshold,
                    global_prompt,
                )
            )
            inverted_mask = clean_hard_mask(1.0 - dataset_mask, min_area_ratio=args.mask_min_area_ratio)
            candidates.append(
                score_mask_candidate(
                    "dataset_mask_inverted",
                    inverted_mask,
                    diff_rgb,
                    args.mask_min_area_ratio,
                    args.global_area_threshold,
                    global_prompt,
                )
            )

    # Prefer trustworthy local masks. If every candidate is broad, fall back to
    # the best broad candidate and let the global-edit rule decide whether the
    # clean target should become the full target.
    local_candidates = [candidate for candidate in candidates if candidate.local_allowed]
    pool = local_candidates or candidates
    best = max(pool, key=lambda item: item.score)
    diagnostics = {
        "selected_mask": best.name,
        "mask_polarity_inverted": best.name == "dataset_mask_inverted",
        "mask_diff_inside": best.inside_diff,
        "mask_diff_outside": best.outside_diff,
        "mask_diff_coverage": best.diff_coverage,
        "mask_score": best.score,
        "mask_candidates": {candidate.name: candidate_to_dict(candidate) for candidate in candidates},
    }
    return best.mask.astype(np.float32), best.name, diagnostics


def make_summary_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_summary_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_summary_safe(item) for item in value]
    return value


def mask_strategy_bucket(mask_strategy: str) -> str:
    if mask_strategy.startswith("dataset_mask"):
        return mask_strategy
    return "source_target_diff_otsu"



def construct_clean_target(source: Image.Image, target: Image.Image, soft_mask: np.ndarray) -> Image.Image:
    target = target.resize(source.size, Image.Resampling.LANCZOS)
    x = image_to_float(source)
    y = image_to_float(target)
    alpha = soft_mask[..., None].astype(np.float32).clip(0, 1)
    return float_to_image(alpha * y + (1.0 - alpha) * x)


def row_common(
    request: EditRequest,
    prompt: str,
    clean_target: str,
    soft_mask: str,
    boundary: str,
    hard_mask: str,
    mask_strategy: str,
    mask_diagnostics: dict[str, Any],
    global_edit: bool,
    mask_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "id": f"{request.id}__mtp_lora",
        "image": clean_target,
        "edit_image": [str(request.input_image.resolve())],
        "source_image": str(request.input_image.resolve()),
        "target_image": str(request.target_image.resolve()) if request.target_image else None,
        "prompt": prompt,
        "source_instruction": request.instruction,
        "condition_mode": "input_only",
        "student_condition": "source_only",
        "phase": "mtp_sft",
        "target_role": "masked_target_preserved_clean_target",
        "mask_image": soft_mask,
        "hard_mask_image": hard_mask,
        "boundary_image": boundary,
        "mask_strategy": mask_strategy,
        "mask_diagnostics": mask_diagnostics,
        "global_edit": bool(global_edit),
        "edit_mask_ratio": round(float(mask_ratio), 6),
        "mask_edit_weight": args.mask_edit_weight,
        "mask_bg_weight": args.mask_bg_weight,
        "boundary_weight": args.boundary_weight,
        "bg_preservation_weight": args.bg_preservation_weight,
        "edit_vector_weight": 0.0,
        "loss_weight": 1.0,
        "algorithm": "mtp_lora",
    }


def process_sample(task: tuple[int, dict[str, Any], str, argparse.Namespace]) -> dict[str, Any] | None:
    index, row, request_base_text, args = task
    request_base = Path(request_base_text)
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / args.clean_dirname
    mask_dir = out_dir / args.mask_dirname

    request = EditRequest.from_dict(row, base_dir=request_base)
    if not request.target_image:
        return None
    source = load_rgb(request.input_image)
    target = load_rgb(request.target_image, source.size)
    hard_mask, mask_strategy, mask_diagnostics = build_mask(request, source, target, args)
    mask_ratio = float(hard_mask.mean())
    global_edit = mask_ratio > args.global_area_threshold or is_global_prompt(request.instruction)
    if global_edit:
        hard_mask = np.ones_like(hard_mask, dtype=np.float32)
        soft_mask = hard_mask
        boundary = np.zeros_like(hard_mask, dtype=np.float32)
        clean = target
    else:
        dilated = dilate_mask(hard_mask, args.soft_dilate_radius)
        soft_mask = feather_mask(dilated, int(round(args.soft_blur_sigma)))
        boundary = edit_band(hard_mask, radius=args.boundary_radius)
        clean = construct_clean_target(source, target, soft_mask)

    clean_path = save_image(clean, clean_dir / f"{request.id}.png")
    hard_path = save_mask(hard_mask, mask_dir / "hard" / f"{request.id}.png")
    soft_path = save_mask(soft_mask, mask_dir / "soft" / f"{request.id}.png")
    boundary_path = save_mask(boundary, mask_dir / "boundary" / f"{request.id}.png")
    prompt = request.instruction.strip() + args.prompt_suffix
    metadata_row = row_common(
        request,
        prompt,
        str(clean_path.resolve()),
        str(soft_path.resolve()),
        str(boundary_path.resolve()),
        str(hard_path.resolve()),
        mask_strategy,
        mask_diagnostics,
        global_edit,
        mask_ratio,
        args,
    )
    return {
        "index": index,
        "row": metadata_row,
        "request_id": request.id,
        "source_instruction": request.instruction,
        "input_image": str(request.input_image.resolve()),
        "source_size": source.size,
        "mask_strategy": mask_strategy,
        "mask_bucket": mask_strategy_bucket(mask_strategy),
        "mask_polarity_inverted": bool(mask_diagnostics.get("mask_polarity_inverted")),
        "global_edit": bool(global_edit),
        "edit_mask_ratio": mask_ratio,
    }


def noop_row(result: dict[str, Any], zero_path: str, args: argparse.Namespace) -> dict[str, Any]:
    request_id = result["request_id"]
    input_image = result["input_image"]
    return {
        "id": f"{request_id}__mtp_noop",
        "image": input_image,
        "edit_image": [input_image],
        "source_image": input_image,
        "target_image": input_image,
        "prompt": "Do not change the image. Preserve the original image exactly.",
        "source_instruction": result["source_instruction"],
        "condition_mode": "input_only",
        "student_condition": "source_only",
        "phase": "noop_preservation",
        "target_role": "source_identity_noop",
        "mask_image": zero_path,
        "hard_mask_image": zero_path,
        "boundary_image": zero_path,
        "mask_strategy": "noop_zero_mask",
        "global_edit": False,
        "edit_mask_ratio": 0.0,
        "mask_edit_weight": 0.0,
        "mask_bg_weight": 1.0,
        "boundary_weight": 0.0,
        "bg_preservation_weight": max(0.0, args.bg_preservation_weight),
        "edit_vector_weight": 0.0,
        "loss_weight": args.noop_weight,
        "algorithm": "mtp_lora_noop_regularization",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / args.clean_dirname
    mask_dir = out_dir / args.mask_dirname
    ensure_dir(clean_dir)
    ensure_dir(mask_dir / "hard")
    ensure_dir(mask_dir / "soft")
    ensure_dir(mask_dir / "boundary")
    ensure_dir(mask_dir / "zero")

    source_jsonl = Path(args.jsonl)
    request_base = source_jsonl.parent
    rows: list[dict[str, Any]] = []
    stats = {
        "source_jsonl": str(source_jsonl),
        "limit": args.limit,
        "samples": 0,
        "noop_rows": 0,
        "global_rows": 0,
        "dataset_mask_rows": 0,
        "diff_mask_rows": 0,
        "inverted_mask_rows": 0,
        "selected_mask_counts": {},
    }
    noop_stride = int(round(1.0 / args.noop_fraction)) if args.noop_fraction > 0 else 0
    zero_mask_cache: dict[tuple[int, int], str] = {}

    input_rows = read_jsonl(source_jsonl)
    if args.limit is not None:
        input_rows = input_rows[: args.limit]
    tasks = [(index, row, str(request_base), args) for index, row in enumerate(input_rows)]
    if args.num_workers and args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            iterator = executor.map(process_sample, tasks, chunksize=max(1, args.chunksize))
            results = tqdm(iterator, total=len(tasks), desc=f"mtp-metadata-x{args.num_workers}")
            processed = (result for result in results if result is not None)
            result_iterable = processed
            for result in result_iterable:
                index = result["index"]
                metadata_row = result["row"]
                mask_strategy = result["mask_strategy"]
                global_edit = result["global_edit"]
                source_size = tuple(result["source_size"])
                rows.append(metadata_row)
                stats["samples"] += 1
                if global_edit:
                    stats["global_rows"] += 1
                bucket = result["mask_bucket"]
                stats["selected_mask_counts"][bucket] = stats["selected_mask_counts"].get(bucket, 0) + 1
                if mask_strategy.startswith("dataset_mask"):
                    stats["dataset_mask_rows"] += 1
                else:
                    stats["diff_mask_rows"] += 1
                if result["mask_polarity_inverted"]:
                    stats["inverted_mask_rows"] = stats.get("inverted_mask_rows", 0) + 1

                if noop_stride and index % noop_stride == 0:
                    if source_size not in zero_mask_cache:
                        width, height = source_size
                        zero = np.zeros((height, width), dtype=np.float32)
                        zero_mask_cache[source_size] = str(save_mask(zero, mask_dir / "zero" / f"zero_{width}x{height}.png").resolve())
                    rows.append(noop_row(result, zero_mask_cache[source_size], args))
                    stats["noop_rows"] += 1
    else:
        iterable = tqdm(tasks, desc="mtp-metadata")
        for task in iterable:
            result = process_sample(task)
            if result is None:
                continue
            index = result["index"]
            metadata_row = result["row"]
            mask_strategy = result["mask_strategy"]
            global_edit = result["global_edit"]
            source_size = tuple(result["source_size"])
            rows.append(metadata_row)
            stats["samples"] += 1
            if global_edit:
                stats["global_rows"] += 1

            bucket = result["mask_bucket"]
            stats["selected_mask_counts"][bucket] = stats["selected_mask_counts"].get(bucket, 0) + 1
            if mask_strategy.startswith("dataset_mask"):
                stats["dataset_mask_rows"] += 1
            else:
                stats["diff_mask_rows"] += 1
            if result["mask_polarity_inverted"]:
                stats["inverted_mask_rows"] = stats.get("inverted_mask_rows", 0) + 1

            if noop_stride and index % noop_stride == 0:
                if source_size not in zero_mask_cache:
                    width, height = source_size
                    zero = np.zeros((height, width), dtype=np.float32)
                    zero_mask_cache[source_size] = str(save_mask(zero, mask_dir / "zero" / f"zero_{width}x{height}.png").resolve())
                rows.append(noop_row(result, zero_mask_cache[source_size], args))
                stats["noop_rows"] += 1

    metadata_path = out_dir / args.metadata_name
    metadata_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["metadata_rows"] = len(rows)
    sft_rows = [row for row in rows if row.get("phase") == "mtp_sft"]
    stats["mean_edit_mask_ratio"] = float(np.mean([row.get("edit_mask_ratio", 0.0) for row in sft_rows])) if sft_rows else 0.0
    stats = make_summary_safe(stats)
    (out_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote MTP metadata: {metadata_path}")


if __name__ == "__main__":
    main()
