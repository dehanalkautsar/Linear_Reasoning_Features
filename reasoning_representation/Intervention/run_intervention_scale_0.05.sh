#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_NAME="${MODEL_NAME:-Meta-Llama-3-8B-Instruct}"
LANG_CODE="${1:-${REASONMEM_LANG:-en}}"
LANG_SUFFIX=""
if [[ -n "${LANG_CODE}" ]]; then
  LANG_SUFFIX="_${LANG_CODE}"
fi

DATASET_DIR="${SCRIPT_DIR}/.."
METRICS_DIR="${METRICS_DIR:-${SCRIPT_DIR}/metrics}"
LOGS_DIR="${LOGS_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${METRICS_DIR}" "${LOGS_DIR}"

LOG_FILE="${LOGS_DIR}/intervention_scale_0.05${LANG_SUFFIX}.log"

python "${SCRIPT_DIR}/features_intervention.py" \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "ReasonMem" \
  --model_name "${MODEL_NAME}" \
  --hs_cache_dir "${DATASET_DIR}" \
  --lang "${LANG_CODE}" \
  --label_subset "reasoning" \
  --Intervention True \
  --scale 0.05 \
  --batch_size "${BATCH_SIZE:-46}" --metrics_out "${METRICS_DIR}/intervention_scale_0.05_metrics${LANG_SUFFIX}.json" \
  2>&1 | tee "${LOG_FILE}"
