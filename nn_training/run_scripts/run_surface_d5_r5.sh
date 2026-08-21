#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_surface_d5_r5.sh <si1000|ca|rl|aligned> <gpu>}"
GPU="${2:?usage: run_surface_d5_r5.sh <si1000|ca|rl|aligned> <gpu>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEM_ROOT="${REPO_ROOT}/nn_training/dems/surface/d5_r5"
EVAL_ROOT="${EVAL_ROOT:-${PATCHDMLE_SYCAMORE_DATA:-${REPO_ROOT}/dataset/sycamore}/sample_00/d5_at_q6_5/X/r05}"
PRETRAINED="${PRETRAINED:-none}"
PYTHON="${PYTHON:-python}"

case "${METHOD}" in
  si1000) DEM_PATH="${DEM_ROOT}/si1000.dem" ;;
  ca) DEM_PATH="${DEM_ROOT}/ca.dem" ;;
  rl) DEM_PATH="${DEM_ROOT}/rl.dem" ;;
  aligned) DEM_PATH="${DEM_ROOT}/aligned.dem" ;;
  *) echo "unknown method: ${METHOD}" >&2; exit 2 ;;
esac

STEPS="${STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
KEEP_TOP_K="${KEEP_TOP_K:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/nn_training/checkpoints_surface_d5_r5}"

cd "${REPO_ROOT}"
EXTRA_ARGS=()
if [[ -n "${PRETRAINED}" && "${PRETRAINED}" != "none" ]]; then
  EXTRA_ARGS+=(--pretrained "${PRETRAINED}")
fi
exec "${PYTHON}" nn_training/train_surface_dem.py \
  --dem-path "${DEM_PATH}" \
  --run-name "surface_d5_r5_${METHOD}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "cuda:${GPU}" \
  --seed 71534 \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr 1e-4 \
  --min-lr 1e-6 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --d-model 256 \
  --nhead 8 \
  --num-layers 3 \
  --dropout 0.1 \
  --log-interval 100 \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-data-root "${EVAL_ROOT}" \
  --eval-max-shots 0 \
  --expected-eval-shots 75000 \
  --eval-batch-size 5000 \
  --keep-top-k "${KEEP_TOP_K}" \
  "${EXTRA_ARGS[@]}"
