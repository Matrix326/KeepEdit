#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keepedit.io import ensure_dir, write_jsonl


def save_image_cell(value: Any, path: Path, mode: str = "RGB") -> str:
    ensure_dir(path.parent)
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            from io import BytesIO

            image = Image.open(BytesIO(value["bytes"])).convert(mode)
        elif value.get("path"):
            image = Image.open(value["path"]).convert(mode)
        else:
            raise ValueError(f"Unsupported image dict keys: {value.keys()}")
    elif isinstance(value, Image.Image):
        image = value.convert(mode)
    elif isinstance(value, (str, Path)):
        image = Image.open(value).convert(mode)
    else:
        raise TypeError(f"Unsupported image cell type: {type(value)}")
    image.save(path)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local MagicBrush parquet shards to KeepEdit JSONL.")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_dir = Path(args.parquet_dir)
    files = sorted((parquet_dir / "data").glob(f"{args.split}-*.parquet"))
    if not files:
        files = sorted(parquet_dir.glob(f"**/{args.split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {args.split} parquet shards found under {parquet_dir}")

    out_dir = ensure_dir(args.out_dir)
    image_dir = ensure_dir(out_dir / "images")
    mask_dir = ensure_dir(out_dir / "masks")
    rows = []
    for shard in files:
        table = pd.read_parquet(shard)
        for _, item in table.iterrows():
            if args.limit and len(rows) >= args.limit:
                break
            sample_id = f"{item['img_id']}_{int(item['turn_index'])}"
            input_path = save_image_cell(item["source_img"], image_dir / f"{sample_id}_input.png", mode="RGB")
            target_path = save_image_cell(item["target_img"], image_dir / f"{sample_id}_target.png", mode="RGB")
            mask_path = save_image_cell(item["mask_img"], mask_dir / f"{sample_id}_mask.png", mode="L")
            rows.append(
                {
                    "id": sample_id,
                    "input_image": input_path,
                    "instruction": str(item["instruction"]),
                    "target_image": target_path,
                    "mask_image": mask_path,
                    "edit_type": None,
                    "metadata": {"dataset": "osunlp/MagicBrush", "split": args.split, "shard": shard.name},
                }
            )
        if args.limit and len(rows) >= args.limit:
            break

    write_jsonl(out_dir / f"{args.split}.jsonl", rows)
    print(f"Wrote {len(rows)} rows to {out_dir / f'{args.split}.jsonl'}")


if __name__ == "__main__":
    main()
