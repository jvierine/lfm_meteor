# SYISR Meteor Processing

Rust processing for Sanya, Danzhou and Wenchang binary recordings. Detects
meteors and saves raw-voltage HDF5 cuts, event plots and full-dataset noise
power. Python and MATLAB are not required.

Each station is processed independently. This program does not perform the
joint three-station trajectory, mass or orbit fit.

## Install and Compile

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git curl build-essential pkg-config libhdf5-dev \
  texlive-latex-extra texlive-fonts-recommended dvisvgm

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
  sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

git clone https://github.com/jvierine/lfm_meteor.git
cd lfm_meteor/syisr_native
bash build.sh
```

Build on the computer where you will run the program. Other Linux distributions
need equivalent packages; clusters may provide these through environment modules.

## Analyze a Directory

Replace the paths, keep the input pattern quoted, and use a new output directory:

```bash
target/release/syisr-native init --station sanya \
  --input '/INPUT/DIRECTORY/*.dat*' \
  --output /OUTPUT/DIRECTORY \
  --config sanya.toml

target/release/syisr-native process --config sanya.toml
```

To use 64 local workers without MPI:

```bash
target/release/syisr-native process --config sanya.toml --workers 64
```

The default is `--workers 1`. Workers process different input files while event
cuts and plots are published during processing. Do not combine local workers
with a multi-rank MPI launch.

Before running, check the generated configuration's clock assumption (UTC+8),
search interval and hardware-delay correction (zero until calibrated).
For remote receivers, replace `sanya` with `danzhou` or `wenchang`.

Outputs in the selected directory:

- `event_cuts/`: raw-voltage HDF5 cuts, including available pulses between
  detections, with default padding of 0.1 s and 20 km around the fitted track.
- `event_cuts/plots/`: event PNGs, appearing during processing.
- `noise.h5`: full-dataset noise power in ADC-count-squared units.
- `catalogue.h5`: final event catalogue, written when processing finishes.

Resume an interrupted run with the same configuration and executable:

```bash
target/release/syisr-native process --config sanya.toml --resume
```

## Optional MPI

For fast storage:

```bash
sudo apt install -y libopenmpi-dev openmpi-bin libclang-dev
bash build.sh --mpi
mpirun -np 4 target/release/syisr-native process --config sanya.toml
```

Use one process for a single mechanical hard disk. Build and launch with the
same MPI installation.

Tested locally on macOS with real Sanya data. The Linux installation recipe has
not yet been tested on Linux.

See [processing details](PROCESSING.md) for additional options and ionospheric
examples, and [file format](FORMAT.md) for metadata and HDF5 conventions.
