#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from keepedit.io import ensure_dir, read_jsonl
from keepedit.schemas import EditRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare source-only Qwen-Image-Edit-2511 LoRA metadata for GT or MoE-teacher supervision."
    )
    parser.add_argument("--jsonl", required=True, help="MagicBrush-style request JSONL.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_mode", choices=("gt", "moe_teacher"), required=True)
    parser.add_argument("--teacher_jsonl", help="MoE-Fusion teacher predictions.jsonl, required for --target_mode moe_teacher.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metadata_name", default="metadata.json")
    parser.add_argument(
        "--prompt_suffix",
        default=" Apply the requested edit to the original image. Preserve all regions not mentioned by the instruction.",
    )
    parser.add_argument("--mask_edit_weight", type=float, default=1.5)
    parser.add_argument("--mask_bg_weight", type=float, default=0.5)
    parser.add_argument("--teacher_min_weight", type=float, default=0.25)
    parser.add_argument("--teacher_max_weight", type=float, default=1.25)
    parser.add_argument("--skip_teacher_below_confidence", type=float, default=-1.0)
    return parser.parse_args()


def resolve(base: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    if path.exists():
        return str(path.resolve())
    return str((base / path).resolve())


def load_teacher_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    base = source.parent
    mapping: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source):
        item = dict(row)
        item["output_image"] = resolve(base, item.get("output_image"))
        item["mask_image"] = resolve(base, item.get("mask_image"))
        metadata = dict(item.get("metadata") or {})
        for key in ("attribution_map", "confidence_map"):
            if metadata.get(key):
                metadata[key] = resolve(base, str(metadata[key]))
        item["metadata"] = metadata
        mapping[str(item.get("id"))] = item
    return mapping


def teacher_confidence(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    metadata = row.get("metadata") or {}
    for key in ("teacher_confidence", "confidence"):
        value = metadata.get(key, row.get(key))
        if value is None:
            continue
        try:
            return float(max(0.0, min(1.0, value)))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def teacher_weight(confidence: float, min_weight: float, max_weight: float) -> float:
    value = min_weight + confidence * (max_weight - min_weight)
    return round(max(min_weight, min(max_weight, value)), 6)


def main() -> None:
    args = parse_args()
    if args.target_mode == "moe_teacher" and not args.teacher_jsonl:
        raise SystemExit("--teacher_jsonl is required when --target_mode moe_teacher")

    source_jsonl = Path(args.jsonl)
    request_base = source_jsonl.parent
    out_dir = ensure_dir(args.out_dir)
    teacher_map = load_teacher_map(args.teacher_jsonl)
    input_rows = read_jsonl(source_jsonl)
    if args.limit is not None:
        input_rows = input_rows[: args.limit]

    rows: list[dict[str, Any]] = []
    skipped = 0
    for raw in input_rows:
        request = EditRequest.from_dict(raw, base_dir=request_base)
        if not request.target_image:
            continue
        source_image = str(request.input_image.resolve())
        target_image = str(request.target_image.resolve())
        prompt = request.instruction.strip() + args.prompt_suffix
        teacher_row = teacher_map.get(request.id)
        confidence = teacher_confidence(teacher_row)

        if args.target_mode == "gt":
            target = target_image
            phase = "gt_onestage"
            target_role = "magicbrush_target"
            loss_weight = 1.0
            teacher_image = None
            teacher_source = None
            mask_image = str(request.mask_image.resolve()) if request.mask_image else None
        else:
            target = (teacher_row or {}).get("output_image")
            if not target:
                skipped += 1
                continue
            if confidence < args.skip_teacher_below_confidence:
                skipped += 1
                continue
            phase = "moe_teacher_onestage"
            target_role = "moe_fusion_teacher"
            loss_weight = teacher_weight(confidence, args.teacher_min_weight, args.teacher_max_weight)
            teacher_image = target
            teacher_source = "moe_fusion_teacher"
            mask_image = (teacher_row or {}).get("mask_image") or (str(request.mask_image.resolve()) if request.mask_image else None)

        metadata = dict((teacher_row or {}).get("metadata") or {})
        rows.append(
            {
                "id": f"{request.id}__{phase}",
                "image": target,
                "edit_image": [source_image],
                "source_image": source_image,
                "target_image": target_image,
                "prompt": prompt,
                "source_instruction": request.instruction,
                "condition_mode": "input_only",
                "student_condition": "source_only",
                "phase": phase,
                "target_role": target_role,
                "mask_image": mask_image,
                "mask_edit_weight": args.mask_edit_weight,
                "mask_bg_weight": args.mask_bg_weight,
                "loss_weight": loss_weight,
                "teacher_image": teacher_image,
                "teacher_source": teacher_source,
                "teacher_confidence": confidence if args.target_mode == "moe_teacher" else None,
                "teacher_selected_bg_expert": metadata.get("selected_bg_expert"),
                "teacher_selected_edit_experts": metadata.get("selected_edit_experts"),
                "teacher_per_expert_scores": metadata.get("per_expert_scores"),
                "algorithm": f"qwen2511_{phase}",
            }
        )

    metadata_path = out_dir / args.metadata_name
    metadata_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "source_jsonl": str(source_jsonl),
        "target_mode": args.target_mode,
        "rows": len(rows),
        "skipped": skipped,
        "metadata": str(metadata_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
