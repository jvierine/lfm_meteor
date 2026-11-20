#!/usr/bin/env bash
#SBATCH --job-name=nov17-physical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=nov17-physical-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export LC_ALL=C
export PATH="$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH"
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$HOME/.local/syisr-campaign-python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
conda=/opt/soft/anaconda3/bin/conda
# The base environment is Python 3.8; current MSIS wheels require newer Python.
environment=/opt/soft/anaconda3/envs/py311
"$conda" run -p "$environment" python -m pip install --target "$HOME/.local/syisr-campaign-python" --no-deps \
    --no-index --find-links /gpfs/pub/data_other/juha/20261117_tristatic/calibration \
    pymsis==0.10.0 rebound==4.4.11
"$conda" run -p "$environment" python -u campaign_20261117/physical.py \
    --workers "$SLURM_CPUS_PER_TASK" --library "$PWD/tristatic_native/target/release/libtristatic_native.so" "$@"
"$conda" run -p "$environment" python -u campaign_20261117/derived.py \
    --workers "$SLURM_CPUS_PER_TASK" \
    --lookup /gpfs/pub/data_other/juha/20261117_tristatic/calibration/lookup.h5 "$@"
