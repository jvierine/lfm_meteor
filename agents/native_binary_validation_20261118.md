# Native Monostatic Binary Validation

Scope: `syisr_native`, branch `codex/native-syisr-binary`. No changes to the
existing MATLAB/Python tri-static reduction. No RCS calculations restarted.

## Local Environment

- macOS arm64; rustc 1.97.1.
- Open MPI 5.0.11; HDF5 2.2.0 from Homebrew.
- Production processing, filtering, grouping, MPI and HDF5 output are Rust only.
- An independent local NumPy/h5py comparison used base conda for verification;
  neither is a runtime dependency or part of the delivered processing program.
- CentOS has not been tested on an actual cluster. Build there with its own MPI
  and HDF5 libraries; do not transfer the macOS binary.

## Format Findings

The supplied MATLAB and Python readers both interpret recorded int32 pairs as
complex Q + i I. Valid headers have start 0xAAAA5555, end 0xAA5555AA, 1024 bytes.
The first files declare 16811 IQ samples, but frames are 135520 bytes: eight
extra bytes follow the declared payload. Headers report sampling/bandwidth
4 MHz, pulse 200 microseconds, IPP 5 ms and pointing approximately azimuth
15 degrees, elevation 75 degrees. Header code-width 199 is retained in the
original headers but is not substituted for the 200-microsecond pulse duration.
The analytic reference follows the existing matched_filter.py detection code.

The example configuration assumes header wall clock UTC+8 and zero additional
hardware-delay correction. Both need operator/calibration confirmation before
absolute timing or range interpretation. No April 2024 calibration is copied.

File SY_2026_11_17_19_44_50.dat1-1-1 has a damaged boundary at byte 437458560.
The next validated header is at 437593056: recovery skips 134496 bytes. The
program records this in HDF5; strict mode (max_resync_bytes=0) stops instead.
Timing discontinuities are separately recorded. Recovery does not invent pulses.

## Tests and Pilot

`cargo test --locked` and `cargo test --locked --features mpi`: 10 tests pass.
`cargo clippy --locked --all-targets --features mpi -- -D warnings`: passes.
Tests cover clocks, unsupported modes, false magic in IQ, truncation, bounded
recovery and its limits, direct versus FFT correlation, injected lag/amplitude,
Doppler sign and overlap loss, Gaussian-noise power, HDF5 empty arrays and
end-to-end extraction of an event spanning two input files.

Real-data pilot used the first three complete input files, not the full 912 GB
directory. Commands, from syisr_native:

```sh
bash build.sh --mpi
target/release/syisr-native process --config config.toml --max-files 3 \
  --output output-serial-recovery
mpirun -np 2 target/release/syisr-native process --config config.toml \
  --max-files 3 --output output-mpi-pilot
```

Processed pulse counts: 7885, 7884, 4053; total 19822. There are 49 threshold
crossings and three candidate events containing 26, 6 and 11 retained pulses.
Peak median-referenced SNRs: 23.92, 17.39 and 20.46 dB. Median apparent ranges
before hardware-delay calibration: 89.94, 91.17 and 88.78 km. These are candidate
detections, not independently classified or calibrated meteor trajectories.

All 154 datasets compared between single-process and two-rank output agree
exactly (excluding configuration bytes because the output directory differs).
Original frame headers and 51600 extracted IQ pairs agree byte-for-byte with
the recordings. Independent direct complex correlation at every retained peak
has maximum relative difference 1.89e-7 from Rust output.

Output directories and raw data are ignored/not committed. A failed initial
strict pilot remains locally at output-serial-pilot; it has no catalogue.h5.
The two successful directories contain the final catalogues and event products.
