#!/usr/bin/env bash
#SBATCH --job-name=nov17-mono-clean
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=nov17-mono-clean-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/monostatic_selection.py --workers "$SLURM_CPUS_PER_TASK"
bash campaign_20261117/figures.slurm
