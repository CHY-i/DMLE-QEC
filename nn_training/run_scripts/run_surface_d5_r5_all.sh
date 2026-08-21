#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/nn_training/surface_d5_r5}"
mkdir -p "${LOG_ROOT}"

methods=(si1000 ca rl aligned)
read -r -a gpus <<<"${GPUS:-0 1 2 3}"
if [[ "${#gpus[@]}" -ne "${#methods[@]}" ]]; then
  echo "GPUS must contain exactly ${#methods[@]} GPU indices" >&2
  exit 2
fi

for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  gpu="${gpus[$index]}"
  log_path="${LOG_ROOT}/${method}.out"
  nohup bash "${REPO_ROOT}/nn_training/run_scripts/run_surface_d5_r5.sh" \
    "${method}" "${gpu}" >"${log_path}" 2>&1 &
  echo "${method}: pid=$! gpu=${gpu} log=${log_path}"
done
