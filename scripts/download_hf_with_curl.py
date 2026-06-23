#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Hugging Face files with curl/aria2 and disabled TLS verification.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--repo_type", choices=["model", "dataset"], default="model")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern. Can be passed multiple times.")
    parser.add_argument("--file", action="append", default=[], help="Exact file path. Can be passed multiple times.")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--aria2_retries", type=int, default=12)
    parser.add_argument(
        "--lowest_speed_limit",
        default=os.environ.get("ARIA2_LOWEST_SPEED_LIMIT", "0"),
        help="Abort aria2 transfer when speed stays below this limit long enough for aria2 to retry.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def api_url(endpoint: str, repo_type: str, repo_id: str) -> str:
    quoted = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    if repo_type == "dataset":
        return f"{endpoint.rstrip('/')}/api/datasets/{quoted}"
    return f"{endpoint.rstrip('/')}/api/models/{quoted}"


def resolve_url(endpoint: str, repo_type: str, repo_id: str, revision: str, path: str) -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_path = urllib.parse.quote(path, safe="/")
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{endpoint.rstrip('/')}/{prefix}{quoted_repo}/resolve/{urllib.parse.quote(revision, safe='')}/{quoted_path}"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "keepedit-downloader"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=120) as response:
        body = response.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"HF API returned non-JSON from {url}: {body[:500]}") from exc


def select_files(siblings: list[dict], patterns: list[str], exact_files: list[str]) -> list[tuple[str, int | None]]:
    sizes = {item["rfilename"]: item.get("size") for item in siblings if "rfilename" in item}
    names = list(sizes)
    selected = set(exact_files)
    for pattern in patterns:
        selected.update(name for name in names if fnmatch.fnmatch(name, pattern))
    missing = sorted(name for name in selected if name not in names)
    if missing:
        raise FileNotFoundError(f"Files not found in repo: {missing}")
    return [(name, sizes.get(name)) for name in sorted(selected)]


def run_download(
    url: str,
    dest: Path,
    connections: int,
    expected_size: int | None = None,
    aria2_retries: int = 12,
    lowest_speed_limit: str = "512K",
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (
        dest.exists()
        and expected_size is not None
        and dest.stat().st_size == expected_size
        and not Path(str(dest) + ".aria2").exists()
    ):
        print(f"[skip] {dest}")
        return

    aria2c = shutil.which("aria2c")
    if aria2c:
        cmd = [
            aria2c,
            "--check-certificate=false",
            "--continue=true",
            f"--max-connection-per-server={connections}",
            f"--split={connections}",
            "--min-split-size=1M",
            "--summary-interval=30",
            f"--lowest-speed-limit={lowest_speed_limit}",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            f"--dir={dest.parent}",
            f"--out={dest.name}",
            url,
        ]
        for attempt in range(1, aria2_retries + 1):
            completed = subprocess.run(cmd, check=False)
            complete = dest.exists() and dest.stat().st_size > 0 and not Path(str(dest) + ".aria2").exists()
            if expected_size is not None:
                complete = complete and dest.stat().st_size == expected_size
            if completed.returncode == 0 and complete:
                return
            print(
                f"[warn] aria2c failed/incomplete for {url}; retry {attempt}/{aria2_retries}",
                file=sys.stderr,
            )
            import time

            time.sleep(min(60, 5 * attempt))
        raise RuntimeError(f"aria2c could not complete {url} after {aria2_retries} retries")

    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("Neither aria2c nor curl is available.")
    cmd = [curl, "-k", "-L", "-C", "-", "--retry", "8", "--retry-delay", "3", "-o", str(dest), url]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    if not args.pattern and not args.file:
        args.pattern = ["*"]

    metadata = fetch_json(api_url(args.endpoint, args.repo_type, args.repo_id))
    files = select_files(metadata.get("siblings", []), args.pattern, args.file)
    print(f"Selected {len(files)} files from {args.repo_type}:{args.repo_id}")
    for path, size in files:
        dest = local_dir / path
        url = resolve_url(args.endpoint, args.repo_type, args.repo_id, args.revision, path)
        print(f"[download] {path} -> {dest}")
        if not args.dry_run:
            run_download(url, dest, args.connections, size, args.aria2_retries, args.lowest_speed_limit)


if __name__ == "__main__":
    main()
