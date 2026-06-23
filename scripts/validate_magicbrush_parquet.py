#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MagicBrush parquet shards before preprocessing.")
    parser.add_argument("--root", default="data/raw/MagicBrush/data")
    parser.add_argument("--split", action="append", choices=["train", "dev"], required=True)
    parser.add_argument("--expect-train-shards", type=int, default=51)
    parser.add_argument("--expect-dev-shards", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    failures: list[str] = []
    total_rows = 0
    expected = {"train": args.expect_train_shards, "dev": args.expect_dev_shards}
    for split in args.split:
        paths = sorted(root.glob(f"{split}-*.parquet"))
        if not paths:
            failures.append(f"{split}: no parquet shards found under {root}")
            continue
        if len(paths) != expected[split]:
            failures.append(f"{split}: expected {expected[split]} shards, found {len(paths)}")
        split_rows = 0
        for path in paths:
            try:
                if path.stat().st_size < 1024:
                    raise ValueError(f"file too small: {path.stat().st_size} bytes")
                parquet = pq.ParquetFile(path)
                split_rows += parquet.metadata.num_rows
            except Exception as exc:
                failures.append(f"{path}: {exc}")
        total_rows += split_rows
        print(f"{split}: {len(paths)} shards, {split_rows} rows")
    if failures:
        raise SystemExit("Invalid parquet shards:\n" + "\n".join(failures))
    print(f"validated {total_rows} rows")


if __name__ == "__main__":
    main()
