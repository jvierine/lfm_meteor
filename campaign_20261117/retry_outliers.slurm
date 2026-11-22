#!/usr/bin/env bash
#SBATCH --job-name=nov17-outliers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=nov17-outliers-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export PYTHONPATH="$HOME/.local/syisr-campaign-python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/soft/anaconda3/bin/conda run --no-capture-output -p /opt/soft/anaconda3/envs/py311 python -u campaign_20261117/physical.py \
    --workers "$SLURM_CPUS_PER_TASK" --retry-quality-failed \
    --library "$PWD/tristatic_native/target/release/libtristatic_native.so"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/quality.py
/opt/soft/anaconda3/bin/conda run --no-capture-output -p /opt/soft/anaconda3/envs/py311 python -u campaign_20261117/derived.py \
    --workers "$SLURM_CPUS_PER_TASK" --lookup /gpfs/pub/data_other/juha/20261117_tristatic/calibration/lookup.h5
bash campaign_20261117/figures.slurm
