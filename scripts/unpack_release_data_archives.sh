#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ROOT_DIR}/archives}"

if [[ ! -d "${ARCHIVE_DIR}" ]]; then
  echo "Archive directory not found: ${ARCHIVE_DIR}" >&2
  echo "Download the dataset repo to the project root first, or set ARCHIVE_DIR=/path/to/archives." >&2
  exit 1
fi

if [[ -f "${ARCHIVE_DIR}/MANIFEST.sha256" ]]; then
  echo "[verify] ${ARCHIVE_DIR}/MANIFEST.sha256"
  (
    cd "${ARCHIVE_DIR}"
    sha256sum -c MANIFEST.sha256
  )
fi

for archive in "${ARCHIVE_DIR}"/data_*.tar; do
  [[ -e "${archive}" ]] || {
    echo "No data_*.tar files found in ${ARCHIVE_DIR}" >&2
    exit 1
  }
  echo "[extract] ${archive}"
  tar -xf "${archive}" -C "${ROOT_DIR}"
done

echo "[done] data restored under ${ROOT_DIR}/data"
