#!/usr/bin/env bash
#SBATCH --job-name=nov17-events
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=8G
#SBATCH --output=nov17-events-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export LC_ALL=C
export PATH="$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH"
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_DIR="$PWD/syisr_native/target/hdf5-sdk"
cargo build --release --locked --manifest-path campaign_20261117/figures/Cargo.toml -j "$SLURM_CPUS_PER_TASK"
export NOVEMBER_FIGURE_BINARY="$PWD/campaign_20261117/figures/target/release/november-memo-figures"
export NOVEMBER_FIGURE_OUTPUT=/gpfs/pub/data_other/juha/20261117_tristatic/figures/events
find /gpfs/pub/data_other/juha/20261117_tristatic/events -mindepth 2 -maxdepth 2 -name event_plot.h5 -print0 |
    xargs -0 -r -P "$SLURM_CPUS_PER_TASK" -I '{}' bash -c '
        input="$1"; name=$(basename "$(dirname "$input")"); out="$NOVEMBER_FIGURE_OUTPUT/$name"
        mkdir -p "$out"
        if [[ ! -s "$out/event.png" || ! -s "$out/event.pdf" || -s "$out/render.log" ]]; then
            "$NOVEMBER_FIGURE_BINARY" event "$input" "$out/event" >"$out/render.log" 2>&1
        fi
    ' _ '{}'
