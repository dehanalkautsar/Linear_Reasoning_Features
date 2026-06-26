#!/bin/bash
#SBATCH --job-name=run_eval_intervention
#SBATCH --time=72:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /home/besher.hassan/miniconda3/etc/profile.d/conda.sh
conda activate /home/besher.hassan/miniconda3/envs/jais-env

# Use a stable path because Slurm copies the job script to a temp dir.
SCRIPT_DIR="/home/besher.hassan/educational_reasoning/bloom_taxonomy/Linear_Reasoning_Features/reasoning_representation/Intervention"

SCRIPTS=(
  "${SCRIPT_DIR}/run_intervention_scale_0.05_Evaluate.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.10_Evaluate.sh"
  "${SCRIPT_DIR}/run_intervention_scale_0.15_Evaluate.sh"
)

pids=()
for script in "${SCRIPTS[@]}"; do
  # One script per node.
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --exclusive /bin/bash "${script}" en &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
