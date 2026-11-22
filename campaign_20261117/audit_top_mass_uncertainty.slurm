#!/usr/bin/env bash
#SBATCH --job-name=nov17-top-mass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=nov17-top-mass-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/audit_top_mass_uncertainty.py \
    --workers "$SLURM_CPUS_PER_TASK" --simulations 32 \
    --library "$PWD/tristatic_native/target/release/libtristatic_native.so"
