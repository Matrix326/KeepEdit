#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-Yitaallen/keepedit-release-data}"
ARCHIVE_DIR="${ARCHIVE_DIR:-hf_release/staging/hf_dataset_split/archives}"
REPO_PREFIX="${REPO_PREFIX:-archives}"
MAX_RETRIES="${MAX_RETRIES:-5}"
UPLOAD_TIMEOUT_SECONDS="${UPLOAD_TIMEOUT_SECONDS:-1800}"

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7897}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export all_proxy="${all_proxy:-${ALL_PROXY}}"

USE_XET="${USE_XET:-0}"
if [[ "${USE_XET}" == "1" ]]; then
  export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
  unset HF_HUB_DISABLE_XET || true
else
  export HF_HUB_DISABLE_XET=1
  unset HF_XET_HIGH_PERFORMANCE || true
fi

if [[ ! -d "${ARCHIVE_DIR}" ]]; then
  echo "Archive directory not found: ${ARCHIVE_DIR}" >&2
  exit 1
fi

shopt -s nullglob
parts=("${ARCHIVE_DIR}"/data_*.tar.*.part)
if (( ${#parts[@]} == 0 )); then
  echo "No split archive parts found in ${ARCHIVE_DIR}" >&2
  exit 1
fi

remote_has_file() {
  local path="$1"
  conda run --no-capture-output -n hw4diff python - "$REPO_ID" "$path" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, path = sys.argv[1], sys.argv[2]
files = set(HfApi().list_repo_files(repo_id, repo_type="dataset"))
raise SystemExit(0 if path in files else 1)
PY
}

upload_one() {
  local file="$1"
  local dest="$2"
  local label="$3"

  if remote_has_file "${dest}"; then
    echo "$(date +%F_%T) [skip] ${label} already exists"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "$(date +%F_%T) [upload] ${label} attempt=${attempt}/${MAX_RETRIES}"
    if timeout "${UPLOAD_TIMEOUT_SECONDS}" conda run --no-capture-output -n hw4diff hf upload \
      "${REPO_ID}" \
      "${file}" \
      "${dest}" \
      --repo-type dataset \
      --commit-message "Upload KeepEdit archive file ${label}" \
      --format agent; then
      echo "$(date +%F_%T) [done] ${label}"
      return 0
    fi

    echo "$(date +%F_%T) [retry] ${label} failed on attempt ${attempt}" >&2
    sleep $(( attempt * 30 ))
    attempt=$(( attempt + 1 ))
  done

  echo "$(date +%F_%T) [failed] ${label} after ${MAX_RETRIES} attempts" >&2
  return 1
}

for file in "${parts[@]}"; do
  base="$(basename "${file}")"
  upload_one "${file}" "${REPO_PREFIX}/${base}" "${base}"
done

upload_one "${ARCHIVE_DIR}/MANIFEST.sha256" "${REPO_PREFIX}/MANIFEST.sha256" "MANIFEST.sha256"
