from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from keepedit.experts import build_expert
from keepedit.io import ensure_dir, load_yaml, read_jsonl, write_jsonl
from keepedit.schemas import Candidate, EditRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enabled editing experts and cache their candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint_interval", type=int, default=25)
    parser.add_argument("--no_resume", action="store_false", dest="resume", default=True)
    return parser.parse_args()


def existing_prediction_map(path: Path, expected_experts: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    complete = {}
    for row in rows:
        candidates = row.get("candidates") or []
        names = {str(candidate.get("name")) for candidate in candidates}
        paths_ok = all(bool(candidate.get("image_path")) and Path(candidate["image_path"]).exists() for candidate in candidates)
        if expected_experts.issubset(names) and paths_ok:
            complete[str(row.get("id"))] = row
    return complete


def cached_candidate(expert: object, request: EditRequest, sample_dir: Path) -> Candidate | None:
    name = str(getattr(expert, "name"))
    sample_dirs = [sample_dir]
    shard_parent = sample_dir.parent.parent
    if sample_dir.parent.name.startswith("out_") and shard_parent.name == "_shards":
        sample_dirs.extend(sorted(path / request.id for path in shard_parent.glob("out_*") if path != sample_dir.parent))
    for candidate_dir in sample_dirs:
        candidates = sorted(
            path for path in candidate_dir.glob(f"{request.id}_{name}.*") if path.is_file() and path.stat().st_size > 0
        )
        if candidates:
            return Candidate(name=name, image_path=candidates[0], metadata={"cache_hit": True})
    return None


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = ensure_dir(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    request_base = Path(args.requests).parent
    rows = read_jsonl(args.requests)
    if args.limit:
        rows = rows[: args.limit]

    expert_configs = config.get("experts", {})
    experts = [
        build_expert(name, expert_config)
        for name, expert_config in expert_configs.items()
        if expert_config.get("enabled", False)
    ]
    if not experts:
        raise ValueError("No experts enabled in config")

    expected_experts = {str(getattr(expert, "name")) for expert in experts}
    existing_rows = existing_prediction_map(predictions_path, expected_experts) if args.resume else {}
    if args.resume:
        print(f"Resume enabled: {len(existing_rows)} complete rows found in {predictions_path}", flush=True)

    outputs = []
    skipped_rows = 0
    cache_hits = 0
    generated = 0
    for index, row in enumerate(tqdm(rows, desc="experts"), start=1):
        request = EditRequest.from_dict(row, base_dir=request_base)
        if args.resume and request.id in existing_rows:
            outputs.append(existing_rows[request.id])
            skipped_rows += 1
            continue

        candidates = []
        sample_dir = ensure_dir(output_dir / request.id)
        for expert in experts:
            candidate = cached_candidate(expert, request, sample_dir) if args.resume else None
            if candidate is None:
                candidate = expert.generate(request, sample_dir)
                if getattr(expert, "unload_after_generate", False) and hasattr(expert, "unload"):
                    expert.unload()
                generated += 1
            else:
                cache_hits += 1
            candidates.append(candidate)
        out = request.to_dict()
        out["candidates"] = [candidate.to_dict() for candidate in candidates]
        outputs.append(out)

        if args.checkpoint_interval > 0 and index % args.checkpoint_interval == 0:
            write_jsonl(predictions_path, outputs)

    write_jsonl(predictions_path, outputs)
    print(
        f"Wrote {len(outputs)} candidate rows to {predictions_path} "
        f"(skipped_rows={skipped_rows}, cache_hits={cache_hits}, generated={generated})",
        flush=True,
    )


if __name__ == "__main__":
    main()
