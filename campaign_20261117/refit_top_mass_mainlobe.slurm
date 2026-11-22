#!/usr/bin/env bash
#SBATCH --job-name=nov17-mass-refit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=nov17-mass-refit-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/audit_top_mass_uncertainty.py \
    --workers 2 --simulations 0 --kinds mainlobe_corrected \
    --library "$PWD/tristatic_native/target/release/libtristatic_native.so"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/revise_top_mass_influx.py
