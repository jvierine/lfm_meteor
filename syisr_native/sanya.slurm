#!/usr/bin/env bash
#SBATCH --job-name=meteor-sanya
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=meteor-sanya-%j.log

set -euo pipefail
cd /home/juha/src/lfm_meteor/syisr_native
export PATH="$HOME/texlive/bin/x86_64-linux:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
binary=./target/release/syisr-native
output=/gpfs/pub/data_other/juha/20261117_sanya
workers="${SLURM_CPUS_PER_TASK:-64}"
mode="${1:-process}"

# Fail before touching products when an old executable is still installed.
"$binary" plot --help | grep -q -- --workers || {
    echo "Old executable: git pull, then bash build.sh before submitting." >&2
    exit 1
}
case "$mode" in
    process)
        resume=()
        if [[ -f "$output/run.h5" ]]; then resume=(--resume); fi
        exec "$binary" process --config sanya.toml \
            --output "$output" --workers "$workers" "${resume[@]}"
        ;;
    plot)
        # Optional second argument selects an existing event-cut directory.
        exec "$binary" plot "${2:-$output/event_cuts}" --workers "$workers"
        ;;
    *)
        echo "Usage: sbatch sanya.slurm [process|plot [EVENT_DIRECTORY]]" >&2
        exit 1
        ;;
esac
