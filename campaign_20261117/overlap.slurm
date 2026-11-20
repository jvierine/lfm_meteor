#!/usr/bin/env bash
#SBATCH --job-name=nov17-overlap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=8G
#SBATCH --output=nov17-overlap-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export LC_ALL=C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/soft/anaconda3/bin/conda run -n base python -u campaign_20261117/overlap.py --workers "$SLURM_CPUS_PER_TASK"
bash campaign_20261117/figures.slurm
