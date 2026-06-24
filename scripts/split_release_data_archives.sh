#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_DIR="${IN_DIR:-${ROOT_DIR}/hf_release/staging/hf_dataset/archives}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/hf_release/staging/hf_dataset_split/archives}"
CHUNK_SIZE="${CHUNK_SIZE:-5G}"

mkdir -p "${OUT_DIR}"

for archive in "${IN_DIR}"/data_*.tar; do
  [[ -e "${archive}" ]] || {
    echo "No data_*.tar files found in ${IN_DIR}" >&2
    exit 1
  }
  base="$(basename "${archive}")"
  if ls "${OUT_DIR}/${base}".*.part >/dev/null 2>&1; then
    echo "[skip] ${base}"
    continue
  fi
  echo "[split] ${base} -> ${OUT_DIR}/${base}.NNN.part"
  split -b "${CHUNK_SIZE}" -d -a 3 --additional-suffix=.part "${archive}" "${OUT_DIR}/${base}."
done

(
  cd "${OUT_DIR}"
  sha256sum data_*.tar.*.part > MANIFEST.sha256
)

echo "[done] split archives written to ${OUT_DIR}"
