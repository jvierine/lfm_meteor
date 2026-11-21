#!/usr/bin/env bash
#SBATCH --job-name=nov17-extreme-review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G
#SBATCH --output=nov17-extreme-review-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export LC_ALL=C
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/review_extremes.py --library tristatic_native/target/release/libtristatic_native.so --workers "$SLURM_CPUS_PER_TASK"
