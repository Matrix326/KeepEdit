#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import requests


def api_url(repo_id: str, repo_type: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/api/datasets/{repo_id}"
    if repo_type == "model":
        return f"https://huggingface.co/api/models/{repo_id}"
    raise ValueError(f"Unsupported repo type: {repo_type}")


def resolve_url(repo_id: str, repo_type: str, filename: str, revision: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def selected(name: str, includes: list[str], excludes: list[str]) -> bool:
    if includes and not any(fnmatch.fnmatch(name, pattern) for pattern in includes):
        return False
    if excludes and any(fnmatch.fnmatch(name, pattern) for pattern in excludes):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an aria2 input file for selected HF repo files.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--repo_type", choices=["model", "dataset"], default="model")
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = requests.get(api_url(args.repo_id, args.repo_type), timeout=60)
    response.raise_for_status()
    files = [item["rfilename"] for item in response.json().get("siblings", [])]
    files = [name for name in files if selected(name, args.include, args.exclude)]
    if not files:
        raise ValueError("No files selected")
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for filename in files:
            handle.write(resolve_url(args.repo_id, args.repo_type, filename, args.revision) + "\n")
            handle.write(f"  out={filename}\n")
    print(f"Wrote {len(files)} URLs to {out_path}")


if __name__ == "__main__":
    main()
