#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
with_mpi=0
if [[ "${1:-}" == "--mpi" ]]; then
    command -v mpicc >/dev/null || { echo "Load an MPI module providing mpicc" >&2; exit 1; }
    export MPI_CC="${MPI_CC:-$(command -v mpicc)}"
    with_mpi=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: ./build.sh [--mpi]" >&2; exit 1
fi
command -v cargo >/dev/null || { echo "Install Rust with rustup or load the cluster Rust module" >&2; exit 1; }
pkg-config --exists hdf5 || { echo "Load HDF5 development libraries; set PKG_CONFIG_PATH if needed" >&2; exit 1; }
export SYISR_GIT_REVISION="$(git rev-parse HEAD)$(test -z "$(git status --porcelain)" || printf '+dirty')"
if [[ "$with_mpi" == 1 ]]; then
    cargo build --release --locked --features mpi
else
    cargo build --release --locked
fi
if [[ "$(uname -s)" == Linux ]]; then
    # Compute nodes may lack the login node's HDF5 installation. Bundle resolved
    # non-glibc dependencies in shared storage, without replacing the loader.
    libdir=target/release/runtime-lib
    mkdir -p "$libdir"
    dependencies=$(ldd target/release/syisr-native)
    if grep -q 'not found' <<< "$dependencies"; then
        printf '%s\n' "$dependencies" >&2
        exit 1
    fi
    while read -r library; do
        case "$(basename "$library")" in
            libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|libutil.so.*|ld-linux*) continue ;;
        esac
        cp -L "$library" "$libdir/$(basename "$library")"
    done < <(awk '$2 == "=>" && $3 ~ /^\// {print $3}' <<< "$dependencies")
    echo "Bundled runtime libraries in $(pwd)/$libdir"
fi
echo "Built $(pwd)/target/release/syisr-native"
