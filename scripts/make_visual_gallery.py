#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def thumb(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    canvas.save(dst)


def make_strip(row: dict, out_path: Path, size: int) -> None:
    paths = [
        ("Input", row["input_image"]),
        ("Output", row["output_image"]),
    ]
    if row.get("target_image"):
        paths.insert(1, ("Target", row["target_image"]))
    for candidate in row.get("candidates") or []:
        image_path = candidate.get("image_path")
        if image_path:
            paths.append((f"Candidate: {candidate.get('name', 'expert')}", image_path))
    if row.get("mask_image"):
        paths.append(("Mask", row["mask_image"]))

    tile_w = size
    label_h = 34
    prompt_h = 72
    canvas = Image.new("RGB", (tile_w * len(paths), size + label_h + prompt_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        x0 = index * tile_w + (tile_w - image.width) // 2
        y0 = label_h + (size - image.height) // 2
        canvas.paste(image, (x0, y0))
        draw.text((index * tile_w + 8, 8), label, fill="black", font=font)

    prompt = row.get("instruction", "")
    prompt = prompt[:220] + ("..." if len(prompt) > 220 else "")
    draw.text((8, size + label_h + 8), prompt, fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual result gallery for image editing outputs.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--thumb_size", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.predictions))[: args.limit]
    out_dir = Path(args.out_dir)
    strips_dir = out_dir / "strips"
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, row in enumerate(rows):
        strip_path = strips_dir / f"{index:04d}_{row.get('id', index)}.jpg"
        make_strip(row, strip_path, args.thumb_size)
        assets_dir.mkdir(parents=True, exist_ok=True)
        input_asset = assets_dir / f"{index:04d}_input{Path(row['input_image']).suffix}"
        output_asset = assets_dir / f"{index:04d}_output{Path(row['output_image']).suffix}"
        shutil.copyfile(row["input_image"], input_asset)
        shutil.copyfile(row["output_image"], output_asset)
        cards.append(
            f"""
            <article class="card">
              <h2>{html.escape(str(row.get('id', index)))}</h2>
              <p class="prompt">{html.escape(row.get('instruction', ''))}</p>
              <p class="meta">{html.escape(json.dumps(row.get('metadata', {}), ensure_ascii=False))}</p>
              <img src="{strip_path.relative_to(out_dir)}" alt="result strip">
            </article>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KeepEdit Visual Gallery</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #16181d; }}
    header {{ padding: 24px 32px; background: #ffffff; border-bottom: 1px solid #dde1e7; }}
    h1 {{ margin: 0; font-size: 24px; }}
    main {{ padding: 24px 32px; display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 20px; }}
    .card {{ background: #ffffff; border: 1px solid #dde1e7; border-radius: 8px; padding: 14px; }}
    .card h2 {{ margin: 0 0 8px; font-size: 15px; }}
    .prompt {{ min-height: 38px; margin: 0 0 12px; color: #3d4552; line-height: 1.35; }}
    .meta {{ margin: 0 0 12px; color: #697386; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #edf0f4; }}
  </style>
</head>
<body>
  <header><h1>KeepEdit Visual Gallery</h1></header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(f"Wrote gallery to {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
