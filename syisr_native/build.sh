#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
features=()
if [[ "${1:-}" == "--mpi" ]]; then
    command -v mpicc >/dev/null || { echo "Load an MPI module providing mpicc" >&2; exit 1; }
    export MPI_CC="${MPI_CC:-$(command -v mpicc)}"
    features=(--features mpi)
elif [[ $# -ne 0 ]]; then
    echo "Usage: ./build.sh [--mpi]" >&2; exit 1
fi
command -v cargo >/dev/null || { echo "Install Rust with rustup or load the cluster Rust module" >&2; exit 1; }
pkg-config --exists hdf5 || { echo "Load HDF5 development libraries; set PKG_CONFIG_PATH if needed" >&2; exit 1; }
export SYISR_GIT_REVISION="$(git rev-parse HEAD)$(test -z "$(git status --porcelain)" || printf '+dirty')"
cargo build --release --locked "${features[@]}"
echo "Built $(pwd)/target/release/syisr-native"
