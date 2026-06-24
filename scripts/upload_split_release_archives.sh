#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-Yitaallen/keepedit-release-data}"
ARCHIVE_DIR="${ARCHIVE_DIR:-hf_release/staging/hf_dataset_split/archives}"
REPO_PREFIX="${REPO_PREFIX:-archives}"

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

for file in "${parts[@]}"; do
  base="$(basename "${file}")"
  echo "$(date +%F_%T) [upload] ${base}"
  conda run --no-capture-output -n hw4diff hf upload \
    "${REPO_ID}" \
    "${file}" \
    "${REPO_PREFIX}/${base}" \
    --repo-type dataset \
    --commit-message "Upload KeepEdit archive part ${base}" \
    --format agent
  echo "$(date +%F_%T) [done] ${base}"
done

echo "$(date +%F_%T) [upload] MANIFEST.sha256"
conda run --no-capture-output -n hw4diff hf upload \
  "${REPO_ID}" \
  "${ARCHIVE_DIR}/MANIFEST.sha256" \
  "${REPO_PREFIX}/MANIFEST.sha256" \
  --repo-type dataset \
  --commit-message "Upload KeepEdit split archive manifest" \
  --format agent
echo "$(date +%F_%T) [done] MANIFEST.sha256"
