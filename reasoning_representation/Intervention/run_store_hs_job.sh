#!/bin/bash
#SBATCH --job-name=store_hs_langs
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /home/besher.hassan/miniconda3/etc/profile.d/conda.sh
conda activate /home/besher.hassan/miniconda3/envs/jais-env

srun --ntasks="${SLURM_NTASKS:-1}" --ntasks-per-node=1 /bin/bash /home/besher.hassan/educational_reasoning/bloom_taxonomy/Linear_Reasoning_Features/reasoning_representation/Intervention/run_store_hs.sh
