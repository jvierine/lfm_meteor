# SYISR Station Meteor Processing

One Rust program reads station binary recordings, detects linear meteor tracks,
saves continuous voltage cuts, calculates system-noise power, and makes event
plots. Use the same program at Sanya, Danzhou and Wenchang. Python and MATLAB are
not required. The older tri-static analysis elsewhere in this repository is
unchanged.

**Start here: [Linux Quick Start](LINUX.md)** for installation, build and run commands.

Cuts and plots are published **during processing**, once enough neighbouring
data are ready to cover an event and its padding. MPI rank zero coordinates
extraction/plotting while the other ranks scan files. A serial run publishes
events between files. Neither waits for the full dataset before plotting.

## Build

Install current stable Rust, a C compiler, `pkg-config` and HDF5 development
libraries. MPI also needs `mpicc` and libclang. Plots use Rustplotlib, fetched
automatically at the pinned version in Cargo.lock, plus `latex` and `dvisvgm`.

```sh
cd syisr_native
bash build.sh --mpi             # omit --mpi for a serial-only build
cargo test --locked --features mpi
```

Build on the CentOS cluster after loading its compiler, Rust, HDF5, MPI and LLVM
modules. Module names differ by cluster. Set `PKG_CONFIG_PATH` for HDF5 and
`LIBCLANG_PATH` for LLVM if necessary. Build and launch with the same MPI
installation. No parallel-HDF5 installation is needed. Do not copy a macOS
executable to Linux. The cluster itself has not yet been tested.

For offline builds, prepare `cargo vendor --locked vendor` and its printed Cargo
source configuration on a connected machine before transferring the checkout.
No separate Rustplotlib checkout or machine-specific path dependency is needed.

## Run a Dataset

Create a configuration, quoting the input glob so the program expands it:

```sh
target/release/syisr-native init --station sanya \
  --input '/data/Sanya/20261117/*.dat*' \
  --output /data/products/sanya-20261117 --config sanya.toml
```

Use `--station danzhou` or `--station wenchang` for a remote receiver. Generated
remote configurations use total transmitter-target-receiver path length, not
monostatic range. Review the input/output paths, clock offset, search interval
and hardware-delay calibration. All remaining settings have working defaults.
The assumed header clock is Beijing time (UTC+8); confirm with the operator.

```sh
# Check the recording metadata first.
target/release/syisr-native inspect /data/Sanya/20261117/first-file.dat1-1-0

# Full dataset: no --max-files or --max-frames limits.
target/release/syisr-native process --config sanya.toml
```

For fast storage and an MPI build, use
`mpirun -np 4 target/release/syisr-native process --config sanya.toml`.
Prefer a single process when reading one mechanical hard disk.

A bounded test can add `--max-files 3 --output /data/products/pilot`.
`--event-output /another/disk/event-cuts` places voltage cuts separately from
the summaries and noise series. Output directories must be new; nothing is
overwritten in a fresh run. To resume an interrupted job, repeat the same command
with `--resume`, using the same executable: completed file products are verified
against the configuration and source file size/mtime before reuse. Do not change
processing settings during a resume. `--allow-build-change` is an explicit compatibility override for a
verified non-numerical fix; do not use it to mix different processing methods.
`cluster.slurm` is a launcher example. Workers need the same input paths and
shared output directories. The final `catalogue.h5` is written
only after processing and extraction finish. Failed jobs exit nonzero.

## Noise and Plots

Experimental ionospheric pilot (first 60 seconds only, one-second display bins):

```sh
cargo run --release --locked --example ionosphere -- /data/products/sanya-20261117 ionosphere
cargo run --release --locked --example plot_ionosphere_rti -- ionosphere.h5 ionosphere_rti.png
```

The RTI uses Storm colours and total matched-filter power divided by noise,
without background subtraction. Noise comes from the last 200 us of valid
full-pulse matched-filter lags available in each recording, near the end of the
IPP. Meteor intervals and heuristic coherent-scatter candidate blocks are
excluded. Range bins are 2 km; plotted altitude uses recorded elevation and a
6371 km spherical Earth, with site height zero and no refraction. The HDF5
retains the power, counts, rejection masks, noise-window delays and pulse-pair
diagnostics. The pilot does not establish a de-aliased ionospheric velocity.

Full experimental reduction with spectral noise estimation and whitening:

```sh
SYISR_WHITEN=1 SYISR_NO_PROFILE_PLOTS=1 cargo run --release --locked \
  --example ionosphere_full -- /data/products/sanya-20261117 /data/products/ionosphere-whitened
```

Use a new output directory. This is separate from the meteor catalogue.
Late-window PSD median/ln(2) and IQR/ln(3) estimates are checked for consistency;
the noise spectrum is propagated through the actual compression filter.
Voltage and chirp receive identical finite-FIR whitening, with DC/Nyquist
removal and edge guards. Localized interference and moving-scatter masks are
saved in minute files. The RTI updates every minute. Power is not calibrated
temperature or electron density.

To independently replay a raw-voltage cut and compare zero-Doppler filtering
with Doppler predicted from its fitted range migration:

```sh
cargo run --release --locked --example replay_cut -- event_cuts/event_000001.h5 replay
```

This writes `replay.h5` and `replay.png` (blue: zero Doppler; orange: migration
correction), and checks the replay against the cached complex response.
It also saves a full delay/Doppler bank for the strongest pulse as
`replay_bank.png` and in the HDF5 output. The orange line on that bank is the
migration-derived Doppler. Peak-power comparisons refine delay to 1/32 sample
to reduce sampling-grid scalloping; integer-grid powers are retained separately.
Range acceleration can bias the migration-derived Doppler through LFM coupling;
the example is a diagnostic, not a trajectory refit.

Run the deterministic noisy moving-echo recovery check with:

```sh
cargo run --release --locked --example validate_migration -- validation.h5
```

Every normal run calculates noise for **every processed pulse**, not only meteor
detections. `noise.h5` includes the per-pulse series and 100-pulse averages for
fitting a sky-noise model. Raw power is in ADC-count-squared units, not kelvin;
no sky fit or receiver-temperature calibration is applied by this program.

```sh
# Faster full-dataset noise run: skips matched filtering and detection.
mpirun -np 8 target/release/syisr-native process --config sanya.toml \
  --noise-only --output /data/products/sanya-noise

# A cluster without TeX can extract first, then plot the HDF5 files elsewhere.
mpirun -np 8 target/release/syisr-native process --config sanya.toml --no-plots
target/release/syisr-native plot /data/products/sanya-20261117/event_cuts
```

Noise-only output marks matched-filter quantities as unavailable (NaN), rather
than reporting zero power. A build without plot dependencies is also possible:
`cargo build --release --locked --no-default-features --features mpi`; use
`--no-plots` with that executable.

## What Is Saved

- `work/pulses_NNNNNN.h5`: internal resumable indexes containing pulse times, binary offsets, gate-start delays,
  matched-filter peaks, noise powers and transmitter/receiver pointing.
- `noise.h5`: all-pulse noise series and averages, including receiver pointing
  and UTC for sky-noise calibration. `partial_run` flags use of pilot limits.
- `catalogue.h5`: accepted event times, detection counts and summary values.
- `event_cuts/event_NNNNNN.h5`: **all available pulses** from 0.1 s before to
  0.1 s after the event, whether detected or not, across input file boundaries.
  Raw and complex matched-filter cuts follow the first-order RANSAC track with
  a default half-width of 20 km. Raw cuts also include the pulse envelope.
- `event_cuts/plots/event_NNNNNN.png`: range-time matched-filter SNR plots.
  The default display zooms to the track plus 2 km; the saved cuts retain 20 km.

The range and time padding are configurable. Data outside the recorder's gate
cannot be recovered: truncated padding and real missing frames are explicitly
flagged. Weak pulses are retained; missing measurements are not interpolated.
These remain candidate meteor events, requiring screening for clutter and other
targets. No RCS, three-dimensional trajectory, mass or orbit is inferred here.

See [FORMAT.md](FORMAT.md) for gate-delay, IQ, units and HDF5 conventions.
