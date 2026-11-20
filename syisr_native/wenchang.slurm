#!/usr/bin/env bash
#SBATCH --job-name=meteor-wenchang
#SBATCH --nodes=1
#SBATCH --nodelist=node03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=meteor-wenchang-%j.log

set -euo pipefail
cd /home/juha/src/lfm_meteor/syisr_native
export PATH="$HOME/texlive/bin/x86_64-linux:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
binary=./target/release/syisr-native
output=/gpfs/pub/data_other/juha/20261117_wenchang
workers="${SLURM_CPUS_PER_TASK:-64}"
mode="${1:-process}"

# Fail before touching products when an old executable is still installed.
if ! help=$("$binary" plot --help); then
    echo "Executable cannot start. Run bash build.sh on symaster to bundle runtime libraries." >&2
    exit 1
fi
grep -q -- --workers <<< "$help" || {
    echo "Old executable: git pull, then bash build.sh before submitting." >&2
    exit 1
}
case "$mode" in
    process)
        set --
        if [[ -f "$output/run.h5" ]]; then set -- --resume; fi
        exec "$binary" process --config wenchang.toml \
            --output "$output" --workers "$workers" "$@"
        ;;
    plot)
        # Optional second argument selects an existing event-cut directory.
        exec "$binary" plot "${2:-$output/event_cuts}" --workers "$workers"
        ;;
    *)
        echo "Usage: sbatch wenchang.slurm [process|plot [EVENT_DIRECTORY]]" >&2
        exit 1
        ;;
esac
