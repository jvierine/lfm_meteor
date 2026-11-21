#!/usr/bin/env bash
#SBATCH --job-name=nov17-mass-refit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=nov17-mass-refit-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
/opt/soft/anaconda3/bin/conda run -n base python -u campaign_20261117/overlap.py --workers "$SLURM_CPUS_PER_TASK"
bash campaign_20261117/physical.slurm
bash campaign_20261117/figures.slurm
