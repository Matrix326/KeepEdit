#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


FINAL_LORA_DIRS = [
    Path("checkpoints/qwen_edit_2511_keepedit_gt_onestage"),
    Path("checkpoints/qwen_edit_2511_mtp_phasea"),
    Path("checkpoints/qwen_edit_2511_moe_teacher_onestage"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload KeepEdit release weights and data to Hugging Face.")
    parser.add_argument("--weights_repo_id", required=True, help="HF model repo, e.g. username/keepedit-release-weights.")
    parser.add_argument("--data_repo_id", required=True, help="HF dataset repo, e.g. username/keepedit-release-data.")
    parser.add_argument("--data_dir", default="data", help="Local data directory to upload to the dataset repo.")
    parser.add_argument("--private", action="store_true", help="Create private repos.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include_reports", action="store_true", help="Also upload release metrics and visual galleries.")
    parser.add_argument("--token", help="HF token. If omitted, huggingface_hub uses the logged-in token.")
    return parser.parse_args()


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def upload_lora_weights(api: HfApi, repo_id: str, revision: str) -> None:
    for lora_dir in FINAL_LORA_DIRS:
        require_path(lora_dir)
        files = sorted(lora_dir.glob("*.safetensors"))
        if len(files) != 1:
            raise RuntimeError(f"{lora_dir} should contain exactly one final .safetensors file, found {len(files)}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(lora_dir),
            path_in_repo=lora_dir.name,
            revision=revision,
            commit_message=f"Upload {lora_dir.name}",
        )


def upload_release_data(api: HfApi, repo_id: str, data_dir: Path, revision: str, include_reports: bool) -> None:
    require_path(data_dir)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(data_dir),
        path_in_repo="data",
        revision=revision,
        commit_message="Upload KeepEdit release data",
    )
    if include_reports and Path("reports").exists():
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path="reports",
            path_in_repo="reports",
            revision=revision,
            commit_message="Upload KeepEdit release reports",
        )


def upload_readmes(api: HfApi, weights_repo_id: str, data_repo_id: str, revision: str) -> None:
    weights_readme = Path("hf_release/weights/README.md")
    data_readme = Path("hf_release/data/README.md")
    if weights_readme.exists():
        api.upload_file(
            repo_id=weights_repo_id,
            repo_type="model",
            path_or_fileobj=str(weights_readme),
            path_in_repo="README.md",
            revision=revision,
            commit_message="Upload model card",
        )
    if data_readme.exists():
        api.upload_file(
            repo_id=data_repo_id,
            repo_type="dataset",
            path_or_fileobj=str(data_readme),
            path_in_repo="README.md",
            revision=revision,
            commit_message="Upload dataset card",
        )


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)
    api.create_repo(args.weights_repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.create_repo(args.data_repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    upload_readmes(api, args.weights_repo_id, args.data_repo_id, args.revision)
    upload_lora_weights(api, args.weights_repo_id, args.revision)
    upload_release_data(api, args.data_repo_id, Path(args.data_dir), args.revision, args.include_reports)
    print("Uploaded KeepEdit release artifacts:")
    print(f"  weights: https://huggingface.co/{args.weights_repo_id}")
    print(f"  data:    https://huggingface.co/datasets/{args.data_repo_id}")


if __name__ == "__main__":
    main()
