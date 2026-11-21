#!/usr/bin/env bash
#SBATCH --job-name=nov17-rcs-point
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=nov17-rcs-point-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export PYTHONPATH="$HOME/.local/syisr-campaign-python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/soft/anaconda3/bin/conda run --no-capture-output -p /opt/soft/anaconda3/envs/py311 python -u campaign_20261117/derived.py \
    --rcs-only --workers "$SLURM_CPUS_PER_TASK" \
    --lookup /gpfs/pub/data_other/juha/20261117_tristatic/calibration/lookup.h5
bash campaign_20261117/figures.slurm
