#!/usr/bin/env bash
#SBATCH --job-name=nov17-tristatic
#SBATCH --nodes=1
#SBATCH --nodelist=node03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=nov17-tristatic-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export LC_ALL=C
export PATH="$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH"
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_DIR="$PWD/syisr_native/target/hdf5-sdk"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
workers="${SLURM_CPUS_PER_TASK:-128}"
conda=/opt/soft/anaconda3/bin/conda
cargo build --release --locked --manifest-path tristatic_native/Cargo.toml --lib -j "$workers"
"$conda" run -n base python -u campaign_20261117/analyze.py associate
"$conda" run -n base python -u campaign_20261117/analyze.py fit --workers "$workers" "$@"
