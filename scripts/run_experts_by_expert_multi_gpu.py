#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keepedit.io import load_yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cached_image_path(output_dir: Path, request_id: str, expert_name: str) -> Path | None:
    sample_dirs = [output_dir / request_id]
    sample_dirs.extend(sorted((output_dir / "_shards").glob(f"out_*/{request_id}")))
    for sample_dir in sample_dirs:
        matches = sorted(path for path in sample_dir.glob(f"{request_id}_{expert_name}.*") if path.is_file() and path.stat().st_size > 0)
        if matches:
            return matches[0]
    return None


def write_predictions_from_image_cache(output_dir: Path, predictions_path: Path, requests: list[dict[str, Any]], expert_name: str) -> bool:
    rows: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request.get("id"))
        image_path = cached_image_path(output_dir, request_id, expert_name)
        if image_path is None:
            return False
        row = dict(request)
        row["candidates"] = [
            {
                "name": expert_name,
                "image_path": str(image_path),
                "metadata": {"cache_hit": True, "cache_source": "image_cache"},
            }
        ]
        rows.append(row)
    write_jsonl(predictions_path, rows)
    return True


def expert_cache_complete(path: Path, requests: list[dict[str, Any]]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    rows = read_jsonl(path)
    by_id = {str(row.get("id")): row for row in rows}
    if len(by_id) < len(requests):
        return False
    for request in requests:
        row = by_id.get(str(request.get("id")))
        if not row:
            return False
        candidates = row.get("candidates") or []
        if not candidates:
            return False
        for candidate in candidates:
            image_path = candidate.get("image_path")
            if not image_path or not Path(image_path).exists():
                return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experts one expert at a time over multiple GPUs, then merge candidate rows."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress_interval", type=int, default=30)
    parser.add_argument(
        "--parallel_experts",
        action="store_true",
        help="Run enabled experts concurrently on separate GPU groups.",
    )
    parser.add_argument(
        "--expert_gpu_groups",
        default="",
        help="Optional semicolon-separated mapping, e.g. pix2pix=0;qwen_image_edit=1;editar=2,3.",
    )
    return parser.parse_args()


def enabled_experts(config: dict[str, Any]) -> list[str]:
    return [name for name, cfg in config.get("experts", {}).items() if cfg.get("enabled", False)]


def single_expert_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    cloned = json.loads(json.dumps(config))
    for expert_name, expert_cfg in cloned.get("experts", {}).items():
        expert_cfg["enabled"] = expert_name == name
    return cloned


def parse_gpu_groups(value: str) -> dict[str, str]:
    groups: dict[str, str] = {}
    if not value.strip():
        return groups
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid expert GPU group '{item}', expected name=gpu[,gpu...]")
        name, gpus = item.split("=", 1)
        groups[name.strip()] = gpus.strip()
    return groups


def default_gpu_groups(names: list[str], gpus: str) -> dict[str, str]:
    gpu_list = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_list:
        raise ValueError("No GPUs specified")
    groups: dict[str, list[str]] = {name: [] for name in names}
    for index, gpu in enumerate(gpu_list):
        target_name = names[index] if index < len(names) else names[-1]
        groups[target_name].append(gpu)
    for index, name in enumerate(names):
        if not groups[name]:
            groups[name].append(gpu_list[index % len(gpu_list)])
    return {name: ",".join(group) for name, group in groups.items()}


def expert_command(
    args: argparse.Namespace,
    expert_config_path: Path,
    expert_output: Path,
    gpu_group: str,
) -> list[str]:
    cmd = [
        args.python,
        "scripts/run_experts_multi_gpu.py",
        "--config",
        str(expert_config_path),
        "--requests",
        args.requests,
        "--output_dir",
        str(expert_output),
        "--gpus",
        gpu_group,
        "--progress_interval",
        str(args.progress_interval),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return cmd


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = load_yaml(args.config)
    names = enabled_experts(config)
    if not names:
        raise ValueError("No enabled experts")

    config_dir = output_dir / "_expert_configs"
    runs_dir = output_dir / "_expert_runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    configured_groups = parse_gpu_groups(args.expert_gpu_groups)
    gpu_groups = default_gpu_groups(names, args.gpus)
    gpu_groups.update({name: group for name, group in configured_groups.items() if name in names})
    mode = "parallel" if args.parallel_experts else "sequential"
    print(f"Running experts in {mode} mode: {names}", flush=True)
    print(f"Expert GPU groups: {gpu_groups}", flush=True)

    request_rows = read_jsonl(Path(args.requests))
    if args.limit:
        request_rows = request_rows[: args.limit]

    expert_prediction_paths: list[Path] = []
    processes: list[tuple[str, subprocess.Popen, Path]] = []
    for name in names:
        expert_config_path = config_dir / f"{name}.yaml"
        expert_config_path.write_text(yaml.safe_dump(single_expert_config(config, name), sort_keys=False), encoding="utf-8")
        expert_output = runs_dir / name
        expert_predictions = expert_output / "predictions.jsonl"
        expert_prediction_paths.append(expert_predictions)
        if expert_cache_complete(expert_predictions, request_rows):
            print(f"[expert:{name}] reuse complete cache: {expert_predictions}", flush=True)
            continue
        if write_predictions_from_image_cache(expert_output, expert_predictions, request_rows, name):
            print(f"[expert:{name}] rebuilt predictions from image cache: {expert_predictions}", flush=True)
            continue
        cmd = expert_command(args, expert_config_path, expert_output, gpu_groups[name])
        print(f"[expert:{name}] {' '.join(cmd)}", flush=True)
        if args.parallel_experts:
            processes.append((name, subprocess.Popen(cmd), expert_output))
        else:
            subprocess.run(cmd, check=True)

    if processes:
        while any(process.poll() is None for _, process, _ in processes):
            statuses = [
                f"{name}:{'run' if process.poll() is None else process.returncode}"
                for name, process, _ in processes
            ]
            print(f"[experts-by-expert-progress] {' '.join(statuses)}", flush=True)
            time.sleep(max(5, args.progress_interval))
        statuses = [
            f"{name}:{'run' if process.poll() is None else process.returncode}"
            for name, process, _ in processes
        ]
        print(f"[experts-by-expert-progress] {' '.join(statuses)}", flush=True)
        failed = [(name, process.returncode) for name, process, _ in processes if process.returncode != 0]
        if failed:
            raise RuntimeError(f"Expert failures: {failed}")

    merged_by_id = {str(row.get("id")): dict(row, candidates=[]) for row in request_rows}
    for name, path in zip(names, expert_prediction_paths):
        expert_rows = read_jsonl(path)
        for row in expert_rows:
            target = merged_by_id.get(str(row.get("id")))
            if target is None:
                continue
            candidates = row.get("candidates") or []
            target["candidates"].extend(candidates)
        print(f"[merge:{name}] rows={len(expert_rows)}", flush=True)

    merged = [merged_by_id[str(row.get("id"))] for row in request_rows]
    write_jsonl(output_dir / "predictions.jsonl", merged)
    print(f"Wrote {len(merged)} merged rows to {output_dir / 'predictions.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
