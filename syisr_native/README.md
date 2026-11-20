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

Inputs are checked before processing and ordered by recorded UTC, not filename.
Files with the same start time are excluded only after a complete byte-for-byte
comparison confirms duplication; the mapping is saved in `run.h5`. Conflicting
recordings fail explicitly. If raw files were replaced after a run began, keep
the previous output and start a new output directory rather than forcing resume.
If copied blocks or clock jumps overlap across files, extraction switches to a
chronological merge after scanning. Repeated pulses are removed only when their
raw headers and voltage words match exactly. `input_reconciliation.h5` records
both source locations. Earlier previews are archived before the corrected
catalogue and plots are rebuilt; normal recordings still publish incrementally.

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

## Plot Existing Event Cuts

```bash
target/release/syisr-native plot /home/juha/data/20261117_sanya/event_cuts --workers 64
```

This does not reprocess raw recordings. Plot workers default to 1 and run in
independent processes.

## Symaster Slurm Jobs

Build once on the login node with `bash build.sh` (no MPI), then submit:
On Linux the build also copies resolved non-glibc shared libraries, including
HDF5, into `target/release/runtime-lib`. The Slurm jobs load this shared-directory
bundle, so HDF5 need not be installed on each compute node. Login and compute
nodes must have compatible architecture and system C libraries.

```bash
sbatch sanya.slurm
sbatch danzhou.slurm
sbatch wenchang.slurm
squeue -u "$USER"
```

These request 64 CPUs and 64 GB per job: Sanya on node01, Danzhou on node02,
Wenchang on node03. Outputs go to `/gpfs/pub/data_other/juha/20261117_STATION`.
Existing runs resume with configuration/build verification; incompatible runs
fail rather than being overwritten. Logs are written in the submission directory.

For plots only, including existing products in your home directory:

```bash
sbatch sanya.slurm plot /home/juha/data/20261117_sanya/event_cuts
sbatch danzhou.slurm plot /home/juha/data/20261117_danzhou/event_cuts
sbatch wenchang.slurm plot /home/juha/data/20261117_wenchang/event_cuts
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
