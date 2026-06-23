#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DiffSynth Qwen edit metadata before LoRA training.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--min_input_only_fraction", type=float, default=0.35)
    parser.add_argument("--min_teacher_guided_fraction", type=float, default=0.35)
    parser.add_argument("--require_teacher_metadata", action="store_true")
    parser.add_argument("--require_source_only_conditions", action="store_true")
    parser.add_argument("--require_mask_image", action="store_true")
    parser.add_argument(
        "--allowed_phase",
        action="append",
        default=[],
        help="Allowed phase value. Can be repeated, e.g. --allowed_phase moe_warmup --allowed_phase gt_refine.",
    )
    parser.add_argument(
        "--require_teacher_rows",
        action="store_true",
        help="Deprecated alias for require_teacher_metadata.",
    )
    parser.add_argument(
        "--allow_identity_targets",
        action="store_true",
        help="Allow rows where image == edit_image for explicit no-op/source-identity preservation phases.",
    )
    return parser.parse_args()


def as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def main() -> None:
    args = parse_args()
    path = Path(args.metadata)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"No metadata rows in {path}")

    input_only = 0
    teacher_guided_rows = 0
    source_only_condition_rows = 0
    missing: list[str] = []
    suspicious_same_path: list[str] = []
    bad_edit_images: list[str] = []
    bad_source_conditions: list[str] = []
    bad_loss_weights: list[str] = []
    bad_masks: list[str] = []
    bad_phases: list[str] = []
    leaked_teacher: list[str] = []
    phase_counts: dict[str, int] = {}

    for row in rows:
        row_id = str(row.get("id"))
        target = row.get("image")
        teacher_image = row.get("teacher_image")
        edit_images = as_list(row.get("edit_image"))
        condition_mode = row.get("condition_mode") or row.get("student_condition")
        phase = str(row.get("phase") or "")
        if phase:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if args.allowed_phase and phase not in set(args.allowed_phase):
            bad_phases.append(f"{row_id}: phase {phase!r} not in {args.allowed_phase}")

        if phase == "region_dpo":
            winner_image = row.get("winner_image")
            loser_image = row.get("loser_image")
            if not winner_image:
                missing.append(f"{row_id}: missing winner_image")
            elif not Path(str(winner_image)).exists():
                missing.append(f"{row_id}: winner_image not found: {winner_image}")
            if not loser_image:
                missing.append(f"{row_id}: missing loser_image")
            elif not Path(str(loser_image)).exists():
                missing.append(f"{row_id}: loser_image not found: {loser_image}")
        elif not target:
            missing.append(f"{row_id}: missing target image")
        elif not Path(str(target)).exists():
            missing.append(f"{row_id}: target image not found: {target}")

        if not edit_images:
            bad_edit_images.append(f"{row_id}: missing edit_image condition")
        for item in edit_images:
            if not Path(item).exists():
                missing.append(f"{row_id}: edit_image not found: {item}")

        if len(edit_images) == 1:
            source_only_condition_rows += 1
        elif args.require_source_only_conditions:
            bad_source_conditions.append(f"{row_id}: expected exactly one source edit_image, got {len(edit_images)}")

        if condition_mode == "input_only" or row.get("student_condition") == "input_only":
            input_only += 1

        if row.get("teacher_candidate") or row.get("teacher_image") or row.get("teacher_per_expert_scores"):
            teacher_guided_rows += 1

        loss_weight = row.get("loss_weight", 1.0)
        try:
            loss_weight_float = float(loss_weight)
        except Exception:
            bad_loss_weights.append(f"{row_id}: loss_weight is not numeric: {loss_weight!r}")
        else:
            if not (0.0 < loss_weight_float <= 10.0):
                bad_loss_weights.append(f"{row_id}: loss_weight out of range: {loss_weight_float}")

        identity_target_allowed = (
            args.allow_identity_targets
            and phase in {"counterfactual_preservation", "noop_preservation", "source_identity_preservation"}
            and row.get("target_role") in {"source_identity_noop", "source_identity_counterfactual"}
        )
        if target and any(str(target) == item for item in edit_images) and not identity_target_allowed:
            suspicious_same_path.append(row_id)
        if teacher_image and any(str(teacher_image) == item for item in edit_images):
            leaked_teacher.append(row_id)
        if teacher_image and not Path(str(teacher_image)).exists():
            missing.append(f"{row_id}: teacher_image not found: {teacher_image}")

        mask_image = row.get("mask_image")
        if args.require_mask_image and not mask_image:
            bad_masks.append(f"{row_id}: missing mask_image")
        elif mask_image and not Path(str(mask_image)).exists():
            missing.append(f"{row_id}: mask_image not found: {mask_image}")

    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing[:50]))
    if bad_edit_images:
        raise SystemExit("Invalid edit_image rows:\n" + "\n".join(bad_edit_images[:50]))
    if suspicious_same_path:
        raise SystemExit(
            "Found rows where target image is also used as edit condition; this would leak the answer:\n"
            + "\n".join(suspicious_same_path[:50])
        )
    if leaked_teacher:
        raise SystemExit(
            "Found rows where teacher image is used as edit condition; teacher must remain offline only:\n"
            + "\n".join(leaked_teacher[:50])
        )
    if bad_source_conditions:
        raise SystemExit("Invalid source-only conditions:\n" + "\n".join(bad_source_conditions[:50]))
    if bad_loss_weights:
        raise SystemExit("Invalid loss weights:\n" + "\n".join(bad_loss_weights[:50]))
    if bad_masks:
        raise SystemExit("Invalid mask rows:\n" + "\n".join(bad_masks[:50]))
    if bad_phases:
        raise SystemExit("Invalid phases:\n" + "\n".join(bad_phases[:50]))

    input_only_fraction = input_only / len(rows)
    teacher_guided_fraction = teacher_guided_rows / len(rows)
    if input_only_fraction < args.min_input_only_fraction:
        raise SystemExit(
            f"Input-only rows are too sparse: {input_only}/{len(rows)} "
            f"({input_only_fraction:.2%}) < {args.min_input_only_fraction:.2%}"
        )
    if (args.require_teacher_metadata or args.require_teacher_rows) and teacher_guided_fraction < args.min_teacher_guided_fraction:
        raise SystemExit(
            f"Teacher-guided rows are too sparse: {teacher_guided_rows}/{len(rows)} "
            f"({teacher_guided_fraction:.2%}) < {args.min_teacher_guided_fraction:.2%}"
        )
    if args.require_source_only_conditions and source_only_condition_rows != len(rows):
        raise SystemExit(
            f"Expected all rows to use source-only edit conditions, got "
            f"{source_only_condition_rows}/{len(rows)}."
        )

    print(
        "DiffSynth metadata OK: "
        f"rows={len(rows)} input_only={input_only} source_only_conditions={source_only_condition_rows} "
        f"teacher_guided={teacher_guided_rows} "
        f"phases={phase_counts} "
        f"input_only_fraction={input_only_fraction:.2%} "
        f"teacher_guided_fraction={teacher_guided_fraction:.2%}"
    )


if __name__ == "__main__":
    main()
