#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Language toggles for hidden-state caching.
RUN_EN=true
RUN_AR=false
RUN_CH=false
RUN_IND=false

# Dataset toggles for hidden-state caching.
RUN_REASON_MEM=false
RUN_BLOOM_TAXO=true

LANGS=("en" "ar" "ch" "ind")
FLAGS=("${RUN_EN}" "${RUN_AR}" "${RUN_CH}" "${RUN_IND}")
DATASETS=("reason_mem" "bloom_taxo")
DATASET_FLAGS=("${RUN_REASON_MEM}" "${RUN_BLOOM_TAXO}")

any_lang=false
for flag in "${FLAGS[@]}"; do
  if [[ "${flag}" == "true" ]]; then
    any_lang=true
    break
  fi
done

any_dataset=false
for flag in "${DATASET_FLAGS[@]}"; do
  if [[ "${flag}" == "true" ]]; then
    any_dataset=true
    break
  fi
done

if [[ "${any_dataset}" != "true" ]]; then
  echo "No datasets enabled. Set RUN_REASON_MEM/RUN_BLOOM_TAXO to true."
fi

if [[ "${any_lang}" != "true" ]]; then
  echo "No languages enabled. Set RUN_EN/RUN_AR/RUN_CH/RUN_IND to true."
fi

if [[ "${any_dataset}" != "true" ]] || [[ "${any_lang}" != "true" ]]; then
  exit 0
fi

any_run=false
for d_idx in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$d_idx]}"
  dataset_flag="${DATASET_FLAGS[$d_idx]}"
  if [[ "${dataset_flag}" != "true" ]]; then
    continue
  fi
  for idx in "${!LANGS[@]}"; do
    lang="${LANGS[$idx]}"
    flag="${FLAGS[$idx]}"
    if [[ "${flag}" != "true" ]]; then
      continue
    fi
    any_run=true
    python "${SCRIPT_DIR}/../LiReFs_storing_hs.py" --lang "${lang}" --dataset "${dataset}"
  done
done

if [[ "${any_run}" != "true" ]]; then
  echo "No runs executed. Check dataset and language flags."
fi
