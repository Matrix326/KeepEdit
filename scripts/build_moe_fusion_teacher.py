#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from keepedit.grounding import MaskGenerator
from keepedit.io import dump_json, ensure_dir, load_yaml, read_jsonl, write_jsonl
from keepedit.moe import MoEFusionConfig, build_moe_fusion_teacher
from keepedit.planner import RulePlanner
from keepedit.schemas import Candidate, EditRequest, EditResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build supervised KeepEdit MoE-Fusion teachers from Pix2Pix/Qwen/EditAR candidates. "
            "This stage may use target and mask because it is train-time teacher construction."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_workers", type=int, default=1, help="CPU workers for per-sample teacher fusion.")
    parser.add_argument(
        "--allow_partial_experts",
        action="store_true",
        help="Continue when only a subset of experts has a valid candidate for a sample.",
    )
    parser.add_argument(
        "--expected_experts",
        default="pix2pix,qwen_image_edit,editar",
        help="Comma-separated expert names expected in the candidate cache.",
    )
    return parser.parse_args()


def load_candidate_map(path: str | Path) -> dict[str, list[Candidate]]:
    source = Path(path)
    base = source.parent
    mapping: dict[str, list[Candidate]] = {}
    for row in read_jsonl(source):
        candidates = [
            Candidate.from_dict(item, base_dir=base)
            for item in row.get("candidates", [])
            if item.get("image_path")
        ]
        mapping[str(row.get("id"))] = candidates
    return mapping


def validate_candidates(
    request_id: str,
    candidates: list[Candidate],
    expected_experts: set[str],
    allow_partial: bool,
) -> dict[str, Any]:
    valid = [
        item
        for item in candidates
        if item.image_path.exists() and not item.metadata.get("unavailable")
    ]
    valid_names = {item.name for item in valid}
    missing = sorted(expected_experts - valid_names)
    if expected_experts and missing and not allow_partial:
        raise RuntimeError(
            f"Request {request_id} is missing expected expert candidates: {missing}. "
            "Use --allow_partial_experts only for debugging or incremental runs."
        )
    if not valid:
        raise RuntimeError(f"Request {request_id} has no valid candidates")
    return {
        "valid_experts": sorted(valid_names),
        "missing_experts": missing,
        "num_valid_experts": len(valid),
    }


def build_one(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload["row"]
    request_base = Path(payload["request_base"])
    candidates = payload["candidates"]
    expected_experts = set(payload["expected_experts"])
    allow_partial = bool(payload["allow_partial"])
    output_dir = Path(payload["output_dir"])
    config = payload["config"]
    fusion_config = MoEFusionConfig.from_config(config)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    canonical_dir = output_dir / "canonical_targets"
    attribution_dir = output_dir / "attribution"
    confidence_dir = output_dir / "confidence"
    scores_dir = output_dir / "scores"

    request = EditRequest.from_dict(row, base_dir=request_base)
    if not request.target_image:
        raise RuntimeError(f"Request {request.id} has no target_image; cannot build supervised teacher")
    candidate_meta = validate_candidates(
        request.id,
        candidates,
        expected_experts=expected_experts,
        allow_partial=allow_partial,
    )
    planner = RulePlanner()
    masker = MaskGenerator.from_config(config)
    plan = planner.plan(request)
    mask_path, mask_meta = masker.generate(
        request,
        plan,
        masks_dir,
        reference_image=request.target_image,
        candidates=candidates,
    )
    out_path = images_dir / f"{request.id}.png"
    canonical_path = canonical_dir / f"{request.id}.png"
    attribution_path = attribution_dir / f"{request.id}.png"
    confidence_path = confidence_dir / f"{request.id}.png"
    fusion = build_moe_fusion_teacher(
        request=request,
        candidates=candidates,
        plan=plan,
        mask_path=mask_path,
        output_image_path=out_path,
        attribution_path=attribution_path,
        confidence_path=confidence_path,
        config=fusion_config,
        canonical_target_path=canonical_path,
    )
    score_payload = {
        "id": request.id,
        "input_image": str(request.input_image),
        "target_image": str(request.target_image),
        "output_image": str(out_path),
        "mask_image": str(mask_path),
        "canonical_target_image": str(canonical_path),
        "attribution_map": str(attribution_path),
        "confidence_map": str(confidence_path),
        **mask_meta,
        **fusion.metadata,
    }
    dump_json(scores_dir / f"{request.id}.json", score_payload)
    return EditResult(
        id=request.id,
        input_image=request.input_image,
        instruction=request.instruction,
        output_image=out_path,
        mask_image=mask_path,
        target_image=request.target_image,
        candidates=candidates,
        plan=plan,
        metadata={
            **mask_meta,
            **candidate_meta,
            **fusion.metadata,
            "canonical_target_image": str(canonical_path),
            "attribution_map": str(attribution_path),
            "confidence_map": str(confidence_path),
            "scores_json": str(scores_dir / f"{request.id}.json"),
        },
    ).to_dict()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "images")
    ensure_dir(output_dir / "masks")
    ensure_dir(output_dir / "canonical_targets")
    ensure_dir(output_dir / "attribution")
    ensure_dir(output_dir / "confidence")
    ensure_dir(output_dir / "scores")

    request_base = Path(args.requests).parent
    rows = read_jsonl(args.requests)
    if args.limit:
        rows = rows[: args.limit]
    candidate_map = load_candidate_map(args.candidates_jsonl)
    expected_experts = {item.strip() for item in args.expected_experts.split(",") if item.strip()}

    payloads = []
    for row in rows:
        request_id = str(row.get("id"))
        payloads.append(
            {
                "row": row,
                "request_base": str(request_base),
                "candidates": candidate_map.get(request_id, []),
                "expected_experts": sorted(expected_experts),
                "allow_partial": args.allow_partial_experts,
                "output_dir": str(output_dir),
                "config": config,
            }
        )

    outputs = []
    if args.num_workers <= 1:
        for payload in tqdm(payloads, desc="moe-fusion-teacher"):
            outputs.append(build_one(payload))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(build_one, payload) for payload in payloads]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"moe-fusion-teacher x{args.num_workers}"):
                outputs.append(future.result())
        outputs.sort(key=lambda item: str(item.get("id")))

    write_jsonl(output_dir / "predictions.jsonl", outputs)
    print(f"Wrote {len(outputs)} MoE-Fusion teacher rows to {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
