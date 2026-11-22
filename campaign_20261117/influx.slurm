#!/usr/bin/env bash
#SBATCH --job-name=nov17-influx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=nov17-influx-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/influx.py
