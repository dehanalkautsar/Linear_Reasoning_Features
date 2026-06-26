#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Language toggles.
RUN_EN=false
RUN_AR=false
RUN_CH=false
RUN_IND=true

# Dataset toggles.
RUN_REASON_MEM=false
RUN_BLOOM_TAXO=true

# Model toggles.
RUN_LLAMA_3_8B=false
RUN_GEMMA_2_9B=false
RUN_QWEN_3_5_4B=false
RUN_QWEN_1_5_4B=true

BATCH_SIZE=46
# BATCH_SIZE=92 # 2 GPU A40
export BATCH_SIZE

MODEL_NAMES=("Meta-Llama-3-8B-Instruct" "google/gemma-2-9b" "Qwen/Qwen3.5-4B" "Qwen/Qwen1.5-4B")
MODEL_FLAGS=("${RUN_LLAMA_3_8B}" "${RUN_GEMMA_2_9B}" "${RUN_QWEN_3_5_4B}" "${RUN_QWEN_1_5_4B}")
HS_CACHE_DIR="${SCRIPT_DIR}/../reasoning_representations_outputs"
METRICS_ROOT="${METRICS_ROOT:-${SCRIPT_DIR}/metrics}"
LOGS_ROOT="${LOGS_ROOT:-${SCRIPT_DIR}/logs}"

sanitize_dir() {
  local name="${1##*/}"
  name="$(echo "${name}" | tr -cs 'A-Za-z0-9._-' '_')"
  echo "${name}"
}

dataset_dir_name() {
  case "${1}" in
    bloom_taxo) echo "bloom_taxo" ;;
    reason_mem) echo "reason_mem" ;;
    *) sanitize_dir "${1}" ;;
  esac
}

SCRIPTS_REASON_MEM=(
  "${SCRIPT_DIR}/run_baseline.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.05.sh"
  "${SCRIPT_DIR}/run_intervention_scale_-0.05.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.10.sh"
  "${SCRIPT_DIR}/run_intervention_scale_-0.10.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.15.sh"
  "${SCRIPT_DIR}/run_intervention_scale_-0.15.sh"
)
SCRIPTS_REASON_MEM_FLAGS=(
  false
  false
  false
  false
  false
  false
  false
)
SCRIPTS_BLOOM_TAXO=(
  "${SCRIPT_DIR}/run_baseline_BloomTaxo_Remember.sh"
  "${SCRIPT_DIR}/run_baseline_BloomTaxo_Understand.sh"
  "${SCRIPT_DIR}/run_baseline_BloomTaxo_Apply.sh"
  "${SCRIPT_DIR}/run_baseline_BloomTaxo_Analyze.sh"
  "${SCRIPT_DIR}/run_baseline_BloomTaxo_Evaluate.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.05_Remember.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.10_Remember.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.15_Remember.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.05_Understand.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.10_Understand.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.15_Understand.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.05_Apply.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.10_Apply.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.15_Apply.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.05_Analyze.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.10_Analyze.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.15_Analyze.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.05_Evaluate.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.10_Evaluate.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.15_Evaluate.sh"
)
SCRIPTS_BLOOM_TAXO_FLAGS=(
  false
  false
  false
  false
  false
  false
  false
  false
  false
  false
  false
  false
  false
  false
  false
  true
  false
  false
  false
  false
)

if [[ ${#SCRIPTS_REASON_MEM[@]} -ne ${#SCRIPTS_REASON_MEM_FLAGS[@]} ]]; then
  echo "Mismatch: SCRIPTS_REASON_MEM and SCRIPTS_REASON_MEM_FLAGS lengths differ."
  exit 1
fi

if [[ ${#SCRIPTS_BLOOM_TAXO[@]} -ne ${#SCRIPTS_BLOOM_TAXO_FLAGS[@]} ]]; then
  echo "Mismatch: SCRIPTS_BLOOM_TAXO and SCRIPTS_BLOOM_TAXO_FLAGS lengths differ."
  exit 1
fi

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

any_model=false
for flag in "${MODEL_FLAGS[@]}"; do
  if [[ "${flag}" == "true" ]]; then
    any_model=true
    break
  fi
done

if [[ "${any_model}" != "true" ]]; then
  echo "No models enabled. Set RUN_LLAMA_3_8B/RUN_GEMMA_2_9B to true."
fi

if [[ "${any_dataset}" != "true" ]] || [[ "${any_lang}" != "true" ]] || [[ "${any_model}" != "true" ]]; then
  exit 0
fi

any_run=false
for m_idx in "${!MODEL_NAMES[@]}"; do
  model_flag="${MODEL_FLAGS[$m_idx]}"
  if [[ "${model_flag}" != "true" ]]; then
    continue
  fi

  model_name="${MODEL_NAMES[$m_idx]}"
  model_tag="${model_name##*/}"
  model_dir="$(sanitize_dir "${model_tag}")"
  export MODEL_NAME="${model_name}"

  for d_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$d_idx]}"
    dataset_flag="${DATASET_FLAGS[$d_idx]}"
    if [[ "${dataset_flag}" != "true" ]]; then
      continue
    fi

    dataset_suffix=""
    dataset_dir="$(dataset_dir_name "${dataset}")"
    scripts_to_run=()
    if [[ "${dataset}" == "bloom_taxo" ]]; then
      dataset_suffix="_bloom_taxo"
      for i in "${!SCRIPTS_BLOOM_TAXO[@]}"; do
        if [[ "${SCRIPTS_BLOOM_TAXO_FLAGS[$i]}" == "true" ]]; then
          scripts_to_run+=("${SCRIPTS_BLOOM_TAXO[$i]}")
        fi
      done
    else
      for i in "${!SCRIPTS_REASON_MEM[@]}"; do
        if [[ "${SCRIPTS_REASON_MEM_FLAGS[$i]}" == "true" ]]; then
          scripts_to_run+=("${SCRIPTS_REASON_MEM[$i]}")
        fi
      done
    fi

    for idx in "${!LANGS[@]}"; do
      lang="${LANGS[$idx]}"
      flag="${FLAGS[$idx]}"
      if [[ "${flag}" != "true" ]]; then
        continue
      fi
      any_run=true
      lang_dir="$(sanitize_dir "${lang}")"
      export METRICS_DIR="${METRICS_ROOT}/${dataset_dir}/${model_dir}/${lang_dir}"
      export LOGS_DIR="${LOGS_ROOT}/${dataset_dir}/${model_dir}/${lang_dir}"
      lang_suffix="_${lang}"
      if [[ "${dataset}" == "reason_mem" ]]; then
        hs_cache_file="${HS_CACHE_DIR}/${model_tag}-base_hs_cache_no_cot_all${lang_suffix}.pt"
      else
        hs_cache_file="${HS_CACHE_DIR}/${model_tag}-base_hs_cache_no_cot_all${dataset_suffix}${lang_suffix}.pt"
      fi

      if [[ -n "${SLURM_PROCID:-}" && "${SLURM_NTASKS:-1}" -gt 1 ]]; then
        if [[ "${SLURM_PROCID}" -eq 0 ]]; then
          if [[ ! -f "${hs_cache_file}" ]]; then
            python "${SCRIPT_DIR}/../LiReFs_storing_hs.py" --lang "${lang}" --dataset "${dataset}"
          fi
        fi
        while [[ ! -f "${hs_cache_file}" ]]; do
          sleep 30
        done
        if [[ ${#scripts_to_run[@]} -gt 0 ]]; then
          for ((i=SLURM_PROCID; i<${#scripts_to_run[@]}; i+=SLURM_NTASKS)); do
            "${scripts_to_run[i]}" "${lang}"
          done
        elif [[ "${SLURM_PROCID}" -eq 0 ]]; then
          echo "No intervention scripts configured for ${dataset}; cached only."
        fi
      else
        if [[ ! -f "${hs_cache_file}" ]]; then
          python "${SCRIPT_DIR}/../LiReFs_storing_hs.py" --lang "${lang}" --dataset "${dataset}"
        fi
        if [[ ${#scripts_to_run[@]} -gt 0 ]]; then
          for script in "${scripts_to_run[@]}"; do
            "${script}" "${lang}"
          done
        else
          echo "No intervention scripts configured for ${dataset}; cached only."
        fi
      fi
    done
  done
done

if [[ "${any_run}" != "true" ]]; then
  echo "No runs executed. Check dataset and language flags."
fi
