#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard final QwenEdit-LoRA inference over multiple GPUs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--diffsynth_root", default="external/DiffSynth-Studio")
    parser.add_argument("--model_base", default="checkpoints/diffsynth")
    parser.add_argument("--edit_model_id", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--text_vae_model_id", default="Qwen/Qwen-Image")
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--condition_mode", default="input_only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--denoising_strength", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--no_background_compose", action="store_true")
    parser.add_argument("--progress_interval", type=int, default=30)
    return parser.parse_args()


def shard_rows(rows: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    return [rows[index::num_shards] for index in range(num_shards)]


def main() -> None:
    args = parse_args()
    gpu_list = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_list:
        raise ValueError("No GPUs specified")

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.requests))
    if args.limit:
        rows = rows[: args.limit]
    order = {str(row.get("id")): index for index, row in enumerate(rows)}

    processes: list[tuple[int, str, subprocess.Popen[Any], Path]] = []
    for index, (gpu, shard) in enumerate(zip(gpu_list, shard_rows(rows, len(gpu_list)))):
        requests_path = shard_dir / f"requests_{index}.jsonl"
        shard_output = shard_dir / f"out_{index}"
        write_jsonl(requests_path, shard)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable,
            "-m",
            "keepedit.pipelines.run_qwen_edit_lora",
            "--config",
            args.config,
            "--requests",
            str(requests_path),
            "--diffsynth_root",
            args.diffsynth_root,
            "--model_base",
            args.model_base,
            "--edit_model_id",
            args.edit_model_id,
            "--text_vae_model_id",
            args.text_vae_model_id,
            "--lora_path",
            args.lora_path,
            "--condition_mode",
            args.condition_mode,
            "--output_dir",
            str(shard_output),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--cfg_scale",
            str(args.cfg_scale),
            "--denoising_strength",
            str(args.denoising_strength),
            "--max_pixels",
            str(args.max_pixels),
            "--seed",
            str(args.seed + index),
        ]
        if args.no_background_compose:
            cmd.append("--no_background_compose")
        print(f"[qwen-lora-shard:{index}] gpu={gpu} rows={len(shard)} cmd={' '.join(cmd)}", flush=True)
        processes.append((index, gpu, subprocess.Popen(cmd, env=env), shard_output))

    while any(process.poll() is None for _, _, process, _ in processes):
        status = " ".join(
            f"shard{index}@gpu{gpu}:{'run' if process.poll() is None else process.returncode}"
            for index, gpu, process, _ in processes
        )
        print(f"[qwen-lora-multi-progress] {status}", flush=True)
        time.sleep(max(5, args.progress_interval))

    failed = [(index, process.returncode) for index, _, process, _ in processes if process.returncode != 0]
    if failed:
        raise RuntimeError(f"QwenEdit-LoRA shard failures: {failed}")

    merged: list[dict[str, Any]] = []
    for index, _, _, shard_output in processes:
        pred_path = shard_output / "predictions.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing shard predictions: {pred_path}")
        shard_rows_out = read_jsonl(pred_path)
        print(f"[qwen-lora-merge:{index}] rows={len(shard_rows_out)}", flush=True)
        merged.extend(shard_rows_out)
    merged.sort(key=lambda row: order.get(str(row.get("id")), 10**12))
    write_jsonl(output_dir / "predictions.jsonl", merged)
    print(f"Wrote {len(merged)} merged QwenEdit-LoRA rows to {output_dir / 'predictions.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
