#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
from tqdm import tqdm


def repo_api_url(repo_id: str) -> str:
    return f"https://huggingface.co/api/datasets/{repo_id}"


def resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"


def list_files(repo_id: str) -> list[str]:
    response = requests.get(repo_api_url(repo_id), timeout=30)
    response.raise_for_status()
    data = response.json()
    return [item["rfilename"] for item in data.get("siblings", [])]


def should_keep(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def looks_like_html(path: Path) -> bool:
    try:
        prefix = path.read_bytes()[:512].lstrip().lower()
    except OSError:
        return False
    return prefix.startswith(b"<html") or b"authentication is required" in prefix


def validate_file(path: Path) -> None:
    if path.stat().st_size < 1024:
        raise ValueError(f"{path} is too small to be a dataset shard ({path.stat().st_size} bytes)")
    if looks_like_html(path):
        raise ValueError(f"{path} is an HTML/authentication response, not a dataset file")
    if path.suffix == ".parquet":
        pq.ParquetFile(path)


def download_file(url: str, out_path: Path, force: bool = False, retries: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        try:
            validate_file(out_path)
            print(f"skip valid existing {out_path}")
            return
        except Exception as exc:
            print(f"redownload invalid existing {out_path}: {exc}", file=sys.stderr)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if force and tmp_path.exists():
        tmp_path.unlink()
    for attempt in range(1, retries + 1):
        downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as response:
                if response.status_code == 416:
                    tmp_path.rename(out_path)
                    return
                response.raise_for_status()
                remaining = int(response.headers.get("content-length") or 0)
                total = downloaded + remaining if remaining else 0
                mode = "ab" if downloaded and response.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0
                with tmp_path.open(mode) as handle, tqdm(
                    total=total,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    desc=out_path.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.update(len(chunk))
            validate_file(tmp_path)
            tmp_path.rename(out_path)
            return
        except Exception as exc:
            if attempt >= retries:
                raise
            if tmp_path.exists() and looks_like_html(tmp_path):
                tmp_path.unlink()
            wait_s = min(60, 2**attempt)
            print(f"retry {attempt}/{retries} for {out_path.name} after {exc}; sleeping {wait_s}s", file=sys.stderr)
            time.sleep(wait_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected files from a Hugging Face dataset repo.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pattern", action="append", required=True, help="Glob pattern, e.g. data/dev-*.parquet")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true", help="Redownload even if the target file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        files = list_files(args.repo_id)
    except Exception as exc:
        print(f"failed to list files: {exc}", file=sys.stderr)
        raise
    selected = [name for name in files if should_keep(name, args.pattern)]
    if not selected:
        raise ValueError(f"No files matched {args.pattern}; repo has {len(files)} files")
    out_dir = Path(args.out_dir)
    print(f"Downloading {len(selected)} files from {args.repo_id}")
    for name in selected:
        download_file(resolve_url(args.repo_id, name, args.revision), out_dir / name, force=args.force)


if __name__ == "__main__":
    main()
