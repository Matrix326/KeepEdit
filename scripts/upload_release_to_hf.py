#!/usr/bin/env python3
from __future__ import annotations

import argparse
from json import JSONDecodeError
from pathlib import Path

import httpx
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.hf_api import RepoUrl


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
    parser.add_argument(
        "--data_archive_root",
        default="hf_release/staging/hf_dataset",
        help="Directory whose archives/ subfolder contains packed release data.",
    )
    parser.add_argument("--private", action="store_true", help="Create private repos.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include_reports", action="store_true", help="Also upload release metrics and visual galleries.")
    parser.add_argument("--token", help="HF token. If omitted, huggingface_hub uses the logged-in token.")
    parser.add_argument("--skip_repo_create", action="store_true", help="Assume target repos already exist.")
    parser.add_argument("--skip_readmes", action="store_true", help="Do not upload model/dataset cards.")
    parser.add_argument("--skip_weights", action="store_true", help="Do not upload LoRA weights.")
    parser.add_argument("--skip_data", action="store_true", help="Do not upload the data directory.")
    parser.add_argument("--upload_raw_data", action="store_true", help="Upload raw data/** instead of archives/**.")
    parser.add_argument(
        "--regular_data_upload",
        action="store_true",
        help="Use upload_folder for data instead of the resumable upload_large_folder path.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Workers for large data upload.")
    parser.add_argument(
        "--disable_ssl_verification",
        action="store_true",
        help="Disable TLS certificate verification for self-signed local proxy environments.",
    )
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


def upload_release_data(
    api: HfApi,
    repo_id: str,
    data_dir: Path,
    data_archive_root: Path,
    revision: str,
    include_reports: bool,
    upload_raw_data: bool,
    regular_data_upload: bool,
    num_workers: int,
) -> None:
    if not upload_raw_data:
        require_path(data_archive_root / "archives")
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(data_archive_root),
            revision=revision,
            private=False,
            allow_patterns="archives/**",
            num_workers=num_workers,
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
        return

    require_path(data_dir)
    if regular_data_upload:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo="data",
            revision=revision,
            commit_message="Upload KeepEdit release data",
        )
    else:
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=".",
            revision=revision,
            private=False,
            allow_patterns=f"{data_dir.name}/**",
            num_workers=num_workers,
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


def create_release_repo(api: HfApi, repo_id: str, repo_type: str, private: bool) -> None:
    visibility = "private" if private else "public"
    try:
        api.create_repo(repo_id, repo_type=repo_type, visibility=visibility, exist_ok=True)
    except JSONDecodeError:
        api.repo_info(repo_id, repo_type=repo_type)
        print(f"Repository already available: {repo_id} ({repo_type})")


def repo_url_for_existing_repo(api: HfApi, repo_id: str, repo_type: str | None) -> RepoUrl:
    endpoint = api.endpoint.rstrip("/")
    if repo_type == "dataset":
        return RepoUrl(f"{endpoint}/datasets/{repo_id}", endpoint=endpoint)
    if repo_type == "space":
        return RepoUrl(f"{endpoint}/spaces/{repo_id}", endpoint=endpoint)
    return RepoUrl(f"{endpoint}/{repo_id}", endpoint=endpoint)


def patch_create_repo_json_fallback(api: HfApi) -> None:
    original_create_repo = api.create_repo

    def create_repo_with_fallback(*args, **kwargs):
        try:
            return original_create_repo(*args, **kwargs)
        except JSONDecodeError:
            repo_id = kwargs.get("repo_id") or (args[0] if args else None)
            repo_type = kwargs.get("repo_type")
            if repo_id is None:
                raise
            print(f"Repository already available, continuing after non-JSON create response: {repo_id} ({repo_type})")
            return repo_url_for_existing_repo(api, str(repo_id), repo_type)

    api.create_repo = create_repo_with_fallback  # type: ignore[method-assign]


def main() -> None:
    args = parse_args()
    if args.disable_ssl_verification:
        set_client_factory(lambda: httpx.Client(verify=False, timeout=None))
    api = HfApi(token=args.token)
    patch_create_repo_json_fallback(api)
    if not args.skip_repo_create:
        create_release_repo(api, args.weights_repo_id, repo_type="model", private=args.private)
        create_release_repo(api, args.data_repo_id, repo_type="dataset", private=args.private)
    if not args.skip_readmes:
        upload_readmes(api, args.weights_repo_id, args.data_repo_id, args.revision)
    if not args.skip_weights:
        upload_lora_weights(api, args.weights_repo_id, args.revision)
    if not args.skip_data:
        upload_release_data(
            api,
            args.data_repo_id,
            Path(args.data_dir),
            Path(args.data_archive_root),
            args.revision,
            args.include_reports,
            args.upload_raw_data,
            args.regular_data_upload,
            args.num_workers,
        )
    print("Uploaded KeepEdit release artifacts:")
    if not args.skip_weights:
        print(f"  weights: https://huggingface.co/{args.weights_repo_id}")
    if not args.skip_data:
        print(f"  data:    https://huggingface.co/datasets/{args.data_repo_id}")


if __name__ == "__main__":
    main()
