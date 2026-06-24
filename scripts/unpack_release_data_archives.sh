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

shopt -s nullglob
archives=("${ARCHIVE_DIR}"/data_*.tar)
parts=("${ARCHIVE_DIR}"/data_*.tar.*.part)

if (( ${#archives[@]} > 0 )); then
  for archive in "${archives[@]}"; do
    echo "[extract] ${archive}"
    tar -xf "${archive}" -C "${ROOT_DIR}"
  done
elif (( ${#parts[@]} > 0 )); then
  mapfile -t bases < <(
    for part in "${parts[@]}"; do
      basename "${part}" | sed -E 's/\.[0-9]+\.part$//'
    done | sort -u
  )
  for base in "${bases[@]}"; do
    echo "[extract split] ${base}"
    cat "${ARCHIVE_DIR}/${base}".*.part | tar -xf - -C "${ROOT_DIR}"
  done
else
  echo "No data_*.tar or data_*.tar.*.part files found in ${ARCHIVE_DIR}" >&2
  exit 1
fi

echo "[done] data restored under ${ROOT_DIR}/data"
