#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/hf_release/staging/hf_dataset/archives}"
FORCE="${FORCE:-0}"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

parts=(
  processed
  diffsynth
  outputs
  candidates
  teachers
)

for part in "${parts[@]}"; do
  src="data/${part}"
  dst="${OUT_DIR}/data_${part}.tar"
  if [[ ! -d "${src}" ]]; then
    echo "[missing] ${src}"
    exit 1
  fi
  if [[ -f "${dst}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${dst}"
    continue
  fi
  tmp="${dst}.tmp"
  rm -f "${tmp}"
  echo "[pack] ${src} -> ${dst}"
  tar -cf "${tmp}" "${src}"
  mv "${tmp}" "${dst}"
done

(
  cd "${OUT_DIR}"
  sha256sum data_*.tar > MANIFEST.sha256
)

echo "[done] archives written to ${OUT_DIR}"
