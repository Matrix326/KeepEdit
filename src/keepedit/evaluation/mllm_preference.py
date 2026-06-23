from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from keepedit.io import ensure_dir, read_jsonl
from keepedit.mllm.client import MLLMClient
from keepedit.mllm.qwen_vl import QwenVLBackend, preference_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 MLLM preference evaluator for image-editing outputs.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--command", help="JSON-over-stdin MLLM evaluator command.")
    parser.add_argument("--backend", choices=["command", "qwen3_vl", "none"], default="command")
    parser.add_argument("--model_path", default="checkpoints/hf/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout_s", type=int, default=120)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def fallback_score(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "preference_score": None,
        "instruction_correctness": None,
        "background_preservation": None,
        "target_similarity": None,
        "reason": "no_mllm_command_configured",
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    if args.limit:
        rows = rows[: args.limit]
    client = MLLMClient(command=args.command, timeout_s=args.timeout_s)
    backend = None
    if args.backend == "qwen3_vl":
        backend = QwenVLBackend(
            model_path=args.model_path,
            device_map=args.device_map,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
    out_path = Path(args.out_jsonl)
    ensure_dir(out_path.parent)
    completed: set[str] = set()
    if out_path.exists():
        for existing in read_jsonl(out_path):
            row_id = existing.get("id")
            if row_id is not None:
                completed.add(str(row_id))
    if completed:
        print(f"Resume MLLM preference: skip {len(completed)} existing rows from {out_path}")

    written = 0
    handle = out_path.open("a", encoding="utf-8")
    for row in tqdm(rows, desc="mllm-pref"):
        row_id = str(row.get("id"))
        if row_id in completed:
            continue
        payload = {
            "task": "image_editing_preference_eval",
            "id": row_id,
            "instruction": row.get("instruction"),
            "input_image": row.get("input_image"),
            "output_image": row.get("output_image"),
            "target_image": row.get("target_image"),
            "mask_image": row.get("mask_image"),
            "return_schema": {
                "preference_score": "0..10",
                "instruction_correctness": "0..10",
                "background_preservation": "0..10",
                "reason": "string",
            },
        }
        if backend is not None:
            try:
                images = [row.get("input_image"), row.get("output_image")]
                if row.get("target_image"):
                    images.append(row.get("target_image"))
                result = backend.ask_json(preference_prompt(str(row.get("instruction", ""))), images=[x for x in images if x])
            except Exception as exc:
                result = {"reason": f"qwen3_vl_failed={exc}"}
        elif client.enabled:
            try:
                result = client.ask_json(payload)
            except Exception as exc:
                result = {"reason": f"mllm_failed={exc}"}
        else:
            result = fallback_score(row)
        handle.write(json.dumps({"id": row_id, **result}, ensure_ascii=False) + "\n")
        handle.flush()
        written += 1
    handle.close()
    print(f"Wrote {written} new MLLM preference rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
