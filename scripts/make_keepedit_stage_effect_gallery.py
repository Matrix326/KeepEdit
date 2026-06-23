#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(path: Path) -> dict[str, dict]:
    return {str(row.get("id")): row for row in read_jsonl(path)}


def resolve(path: str | Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    resolved = candidate.resolve()
    return resolved if resolved.exists() else None


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH.")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Stage label cannot be empty.")
    return label, Path(path)


def candidate_map(path: Path, expert_name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in read_jsonl(path):
        for candidate in row.get("candidates") or []:
            if str(candidate.get("name")) != expert_name:
                continue
            image_path = resolve(candidate.get("image_path"))
            if image_path:
                result[str(row.get("id"))] = image_path
            break
    return result


def draw_tile(path: Path, label: str, size: int, label_h: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    x = (size - image.width) // 2
    y = label_h + (size - image.height) // 2
    canvas.paste(image, (x, y))
    draw.rectangle((0, 0, size, label_h), fill=(246, 248, 251))
    draw.text((8, 9), label[:38], fill=(20, 24, 31), font=font)
    return canvas


def make_strip(sample: dict, out_path: Path, size: int) -> None:
    columns = sample["columns"]
    label_h = 34
    prompt_h = 82
    gap = 6
    width = len(columns) * size + (len(columns) - 1) * gap
    height = size + label_h + prompt_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, column in enumerate(columns):
        tile = draw_tile(column["path"], column["label"], size, label_h)
        canvas.paste(tile, (index * (size + gap), 0))
    prompt = f"{sample['id']} | {sample['instruction']}"
    if len(prompt) > 280:
        prompt = prompt[:277] + "..."
    draw.rectangle((0, size + label_h, width, height), fill="white")
    draw.text((8, size + label_h + 10), prompt, fill=(20, 24, 31), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def build_overview(strip_paths: list[Path], out_path: Path, max_rows: int) -> None:
    selected = strip_paths[:max_rows]
    if not selected:
        return
    strips = [Image.open(path).convert("RGB") for path in selected]
    width = max(image.width for image in strips)
    height = sum(image.height for image in strips)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in strips:
        canvas.paste(image, (0, y))
        y += image.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def collect_samples(args: argparse.Namespace) -> list[dict]:
    requests = by_id(Path(args.requests))
    moe = by_id(Path(args.moe_predictions)) if args.moe_predictions else {}
    qwen_candidates = (
        candidate_map(Path(args.candidates_predictions), args.qwen_expert_name)
        if args.candidates_predictions
        else {}
    )
    stage_maps = [(label, by_id(path)) for label, path in args.stage]
    ids = [item.strip() for item in args.ids.split(",") if item.strip()] if args.ids else list(requests)
    samples: list[dict] = []
    for sample_id in ids:
        row = requests.get(sample_id)
        if not row:
            continue
        columns: list[tuple[str, Path | None]] = [
            ("Input", resolve(row.get("input_image"))),
            ("Target", resolve(row.get("target_image"))),
        ]
        if args.include_mask:
            columns.append(("Mask", resolve(row.get("mask_image"))))
        if qwen_candidates:
            columns.append(("Qwen expert", qwen_candidates.get(sample_id)))
        if moe:
            columns.append(("MoE teacher", resolve((moe.get(sample_id) or {}).get("output_image"))))
            canonical = ((moe.get(sample_id) or {}).get("metadata") or {}).get("canonical_target_image")
            if args.include_canonical:
                columns.append(("Canonical teacher", resolve(canonical)))
        for label, rows in stage_maps:
            columns.append((label, resolve((rows.get(sample_id) or {}).get("output_image"))))
        clean_columns = [{"label": label, "path": path} for label, path in columns if path and path.exists()]
        expected_min = 2 + len(stage_maps)
        if len(clean_columns) < expected_min:
            continue
        samples.append(
            {
                "id": sample_id,
                "instruction": row.get("instruction", ""),
                "columns": clean_columns,
            }
        )
        if len(samples) >= args.limit:
            break
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a KeepEdit per-stage visual effect gallery.")
    parser.add_argument("--requests", default="data/processed/magicbrush_dev/dev.jsonl")
    parser.add_argument("--candidates_predictions")
    parser.add_argument("--qwen_expert_name", default="qwen_image_edit")
    parser.add_argument("--moe_predictions")
    parser.add_argument("--stage", action="append", type=parse_named_path, required=True, help="LABEL=predictions.jsonl")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--thumb_size", type=int, default=220)
    parser.add_argument("--overview_rows", type=int, default=8)
    parser.add_argument("--ids", help="Optional comma-separated sample ids.")
    parser.add_argument("--include_mask", action="store_true")
    parser.add_argument("--include_canonical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    strips_dir = out_dir / "strips"
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(args)
    if not samples:
        raise SystemExit("No complete stage-effect samples found.")

    cards: list[str] = []
    manifest: list[dict] = []
    strip_paths: list[Path] = []
    for index, sample in enumerate(samples):
        strip_path = strips_dir / f"{index:04d}_{sample['id']}.jpg"
        make_strip(sample, strip_path, args.thumb_size)
        strip_paths.append(strip_path)
        copied_columns = []
        for col_index, column in enumerate(sample["columns"]):
            src = Path(column["path"])
            suffix = src.suffix or ".png"
            safe_label = column["label"].lower().replace(" ", "_").replace("/", "_")
            dst = assets_dir / f"{index:04d}_{col_index:02d}_{safe_label}{suffix}"
            shutil.copyfile(src, dst)
            copied_columns.append({"label": column["label"], "path": str(dst.relative_to(out_dir))})
        manifest.append(
            {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "strip": str(strip_path.relative_to(out_dir)),
                "columns": copied_columns,
            }
        )
        cards.append(
            f"""
            <article class="card">
              <h2>{html.escape(sample['id'])}</h2>
              <p>{html.escape(sample['instruction'])}</p>
              <img src="{strip_path.relative_to(out_dir)}" alt="KeepEdit stage strip for {html.escape(sample['id'])}">
            </article>
            """
        )

    build_overview(strip_paths, out_dir / "overview.jpg", args.overview_rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KeepEdit Stage Effect Gallery</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6f8; color: #171a21; }}
    header {{ background: #ffffff; border-bottom: 1px solid #d9dee7; padding: 24px 32px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    .sub {{ margin: 8px 0 0; color: #596273; }}
    main {{ padding: 24px 32px; display: grid; gap: 20px; }}
    .card {{ background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    h2 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 0 0 12px; line-height: 1.4; color: #384153; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #edf0f4; }}
  </style>
</head>
<body>
  <header>
    <h1>KeepEdit Stage Effect Gallery</h1>
    <p class="sub">Side-by-side inspection of source, target, expert teacher and each LoRA phase.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(f"Wrote {len(samples)} stage-effect samples to {out_dir / 'index.html'}")
    print(f"Wrote overview to {out_dir / 'overview.jpg'}")


if __name__ == "__main__":
    main()
