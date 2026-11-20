#!/usr/bin/env bash
#SBATCH --job-name=nov17-figures
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --output=nov17-figures-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
export LC_ALL=C
export PATH="$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH"
export LD_LIBRARY_PATH="$PWD/syisr_native/target/release/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_DIR="$PWD/syisr_native/target/hdf5-sdk"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/soft/anaconda3/bin/conda run -n base python -u campaign_20261117/prepare_figures.py --workers "$SLURM_CPUS_PER_TASK"
link_flags=()
if [[ -f "$PWD/syisr_native/target/centos7-link/libc.so" ]]; then
    link_flags=(-L "native=$PWD/syisr_native/target/centos7-link")
fi
cargo rustc --release --locked --manifest-path campaign_20261117/figures/Cargo.toml -j "$SLURM_CPUS_PER_TASK" -- "${link_flags[@]}"
root=/gpfs/pub/data_other/juha/20261117_tristatic
failed=0
for number in 2 3 4 7 8 9 10; do
    file="$root/figure_data/figure$number.h5"
    if [[ -f "$file" ]]; then
        if ! campaign_20261117/figures/target/release/november-memo-figures \
            "figure$number" "$file" "$root/figures/figure$number"; then failed=1; fi
    fi
done
export NOVEMBER_FIGURE_BINARY="$PWD/campaign_20261117/figures/target/release/november-memo-figures"
export NOVEMBER_FIGURE_OUTPUT="$root/figures/events"
find "$root/events" -mindepth 2 -maxdepth 2 -name event_plot.h5 -print0 | \
    xargs -0 -r -P "$SLURM_CPUS_PER_TASK" -I '{}' bash -c '
        input="$1"; name=$(basename "$(dirname "$input")")
        mkdir -p "$NOVEMBER_FIGURE_OUTPUT/$name"
        "$NOVEMBER_FIGURE_BINARY" event "$input" "$NOVEMBER_FIGURE_OUTPUT/$name/event" >"$NOVEMBER_FIGURE_OUTPUT/$name/render.log" 2>&1
    ' _ '{}' || failed=1
exit "$failed"
