# Linux Quick Start

The commands below are for Ubuntu/Debian. On other Linux distributions, install
the equivalent compiler, HDF5 development libraries, pkg-config and TeX packages.
Python and MATLAB are not needed. An internet connection is needed for the build.

## Install and Build

```sh
sudo apt update
sudo apt install -y git curl build-essential pkg-config libhdf5-dev \
  texlive-latex-extra texlive-fonts-recommended dvisvgm

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"

git clone https://github.com/jvierine/lfm_meteor.git
cd lfm_meteor/syisr_native
bash build.sh
```

Build on the Linux computer; do not transfer the macOS executable.

## Configure and Run

Replace the input and output paths with your own. Keep the input pattern quoted.

```sh
target/release/syisr-native init --station sanya \
  --input '/data/Sanya/20261117/*.dat*' \
  --output /data/products/sanya-20261117 --config sanya.toml

target/release/syisr-native process --config sanya.toml
```

Before running, review `sanya.toml`: paths, search interval, recording clock
(assumed UTC+8) and hardware-delay correction (zero until calibrated).
For remote receivers, use `--station danzhou` or `--station wenchang` instead.

Raw-voltage cuts appear in `event_cuts/`, with PNGs in `event_cuts/plots/`,
as processing progresses. The final `noise.h5` contains the full-dataset noise
power series; `catalogue.h5` is written when processing finishes.

To resume after an interruption, use the same configuration and executable:

```sh
target/release/syisr-native process --config sanya.toml --resume
```

## Optional MPI

For data on fast storage, install MPI and rebuild:

```sh
sudo apt install -y libopenmpi-dev openmpi-bin libclang-dev
bash build.sh --mpi
mpirun -np 4 target/release/syisr-native process --config sanya.toml
```

Use the single-process command above for a single mechanical hard disk; multiple
readers can slow it down. On a cluster, build and run with the same MPI installation.

See [README.md](README.md) for output details and additional options. The code
has been tested locally on macOS; this Linux recipe has not yet been run on Linux.
