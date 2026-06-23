#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keepedit.io import load_yaml


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.png"))


def count_enabled_experts(config_path: str) -> int:
    config = load_yaml(config_path)
    experts = config.get("experts", {})
    return sum(1 for expert in experts.values() if expert.get("enabled", False))


def print_progress(output_dir: Path, shard_paths: list[Path], processes: list[subprocess.Popen], expected_candidates: int) -> None:
    png_count = count_pngs(output_dir)
    shard_rows = sum(count_lines(path) for path in shard_paths)
    active = sum(1 for process in processes if process.poll() is None)
    total = f"/{expected_candidates}" if expected_candidates else ""
    percent = ""
    if expected_candidates:
        percent = f" ({png_count * 100 / expected_candidates:.2f}%)"
    statuses = ",".join("run" if process.poll() is None else str(process.returncode) for process in processes)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[experts-progress] {stamp} png={png_count}{total}{percent} "
        f"merged_shard_rows={shard_rows} active_shards={active}/{len(processes)} statuses=[{statuses}]",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard KeepEdit expert inference over multiple GPUs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress_interval", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.requests))
    if args.limit:
        rows = rows[: args.limit]

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")
    enabled_experts = count_enabled_experts(args.config)
    expected_candidates = len(rows) * enabled_experts
    print(
        f"Sharding {len(rows)} requests across {len(gpus)} GPUs with "
        f"{enabled_experts} enabled experts; expected candidate images: {expected_candidates}",
        flush=True,
    )

    shard_paths = []
    processes = []
    for index, gpu in enumerate(gpus):
        shard_rows = rows[index:: len(gpus)]
        shard_requests = shard_dir / f"requests_{index}.jsonl"
        shard_output = shard_dir / f"out_{index}"
        write_jsonl(shard_requests, shard_rows)
        shard_paths.append(shard_output / "predictions.jsonl")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            args.python,
            "-m",
            "keepedit.pipelines.run_experts",
            "--config",
            args.config,
            "--requests",
            str(shard_requests),
            "--output_dir",
            str(shard_output),
        ]
        processes.append(subprocess.Popen(cmd, env=env))

    last_progress = 0.0
    while any(process.poll() is None for process in processes):
        now = time.time()
        if now - last_progress >= args.progress_interval:
            print_progress(output_dir, shard_paths, processes, expected_candidates)
            last_progress = now
        time.sleep(5)
    print_progress(output_dir, shard_paths, processes, expected_candidates)

    failed = [(index, process.returncode) for index, process in enumerate(processes) if process.returncode != 0]
    if failed:
        raise RuntimeError(f"Expert shard failures: {failed}")

    merged: list[dict] = []
    for path in shard_paths:
        merged.extend(read_jsonl(path))
    order = {str(row.get("id")): index for index, row in enumerate(rows)}
    merged.sort(key=lambda row: order.get(str(row.get("id")), 10**12))
    write_jsonl(output_dir / "predictions.jsonl", merged)
    print(f"Wrote {len(merged)} merged rows to {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
