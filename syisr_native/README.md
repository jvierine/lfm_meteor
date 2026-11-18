# Sanya Monostatic LFM Processor

Rust-only binary reader, analytic LFM matched filter, pulse detector and meteor-event
extractor for the November 2026 Sanya recordings. No MATLAB, Python or intermediate
MAT files are needed. MPI distributes input files; events are joined across file
and rank boundaries after processing.

## Build

Install a current stable Rust toolchain, a C compiler, `pkg-config`, and HDF5
development libraries. MPI builds also need MPI development libraries, `mpicc`,
and libclang (used when generating the MPI bindings).

```sh
cd syisr_native
bash build.sh                    # serial executable
cargo test --locked
bash build.sh --mpi              # replaces it with the MPI-enabled executable
cargo test --locked --features mpi
```

Build on the CentOS cluster itself, after loading its compiler, HDF5, Rust and
MPI modules. Do not copy a macOS executable there. Use the same MPI installation
to build and launch. An MPI-enabled executable also works without `mpirun`.
No parallel-HDF5 installation is required: each worker writes separate files.

For example, adapting module names to the cluster:

```sh
module load gcc rust hdf5 openmpi llvm
export MPI_CC="$(command -v mpicc)"
# If necessary, point these at the cluster's installation directories:
# export PKG_CONFIG_PATH=/path/to/hdf5/lib/pkgconfig:$PKG_CONFIG_PATH
# export LIBCLANG_PATH=/path/to/llvm/lib
# export LD_LIBRARY_PATH=/path/to/hdf5/lib:/path/to/mpi/lib:$LD_LIBRARY_PATH
bash build.sh --mpi
```

The lockfile fixes dependency versions; the build does not enable host-specific
CPU instructions. Local tests use macOS; a CentOS build still needs to be checked
on the target cluster. For a cluster without internet access, run `cargo vendor
--locked vendor` on a connected machine and follow its printed Cargo source
configuration instructions before transferring the checkout and vendor directory.

## Configure and Run

Edit `config.toml`. It is the complete configuration; unknown keys are errors.
Relative paths in it resolve against the configuration file. Set the input glob
and a new output directory. Existing output directories are never overwritten.

```sh
target/release/syisr-native inspect \
  /Volumes/LaCie/SYISR_data/20261117/SY_2026_11_17_19_44_50.dat1-1-0

# A bounded pilot: two files, at most 1000 pulses per file.
target/release/syisr-native process --config config.toml \
  --max-files 2 --max-frames 1000 --output output-pilot

# Full recording, distributed by file. Remove the pilot limits.
mpirun -np 8 target/release/syisr-native process --config config.toml
```

For Slurm, use the cluster's recommended MPI launcher (`srun` or `mpirun`).
`cluster.slurm` is a starting point. All ranks need the same input paths and a
shared output directory. A serial build refuses a multi-task launcher. Worker
errors abort the MPI job rather than leaving other ranks at a barrier. A failed
run has no final `catalogue.h5`; rerun into a fresh output directory.

The shipped example assumes a Beijing-time header clock (UTC+8). Confirm this
with the operator; filenames are not timestamps. The delay correction defaults
to zero and is explicitly marked **uncalibrated**. Set a measured hardware-delay
correction before interpreting absolute ranges. The old experiment's correction
is not automatically applied.

## Products

All numerical products are HDF5. Configuration, version, source paths and
scientific conventions are embedded as metadata datasets, not plotted credits.
Names ending `_utf8` contain UTF-8 bytes. Times are signed UTC nanoseconds since
1970-01-01. Indexes and sample numbers are zero-based.

- `pulses_NNNNNN.h5`: every processed pulse's timestamp, source byte offset,
  peak lag, apparent delay/range, SNR, noise power, IPP and pointing. These files
  also record input size and a SHA-256 of the processed headers, not the full IQ.
- `catalogue.h5`: event start/stop, pulse counts, peak SNR and median apparent
  range, input manifest, processed counts and rejected overlong-track count.
  Its presence means the run completed, including event extraction.
- `events/event_NNNNNN.h5`: retained detections, signed raw IQ cuts with padding,
  exact frame headers, and complex matched-filter cuts around the peak.

In event files, `raw/iq_words_iq` alternates recorder I then Q int32 words.
The supplied reference readers define complex voltage as **Q + i I**; so does
this program. `raw/pulse_offsets` indexes IQ pairs, not individual int32 words.
For pulse `k`, its cut occupies pairs `[offset[k], offset[k+1])`; the first
original sample is `raw/first_sample[k]`. `raw/frame_headers_u8` contains 1024
bytes per retained pulse. The matched-filter arrays use their own ragged offsets
and first-lag samples. Raw cut padding is clipped at the recording boundaries.
The offsets describe the actual lengths; do not assume all cuts are equal.

## Measurement Conventions

Supported layout: little-endian 1024-byte headers, one DBF beam, 430 MHz carrier,
4 MHz sampling, 4 MHz LFM bandwidth and 200 microsecond pulses. The chirp direction
is explicit in the configuration because the header does not establish it.
The observed November files have an extra eight bytes after the declared IQ
payload. These bytes are skipped, not counted as declared samples. The setting
is configurable. Unexpected markers trigger a bounded search for the next header
with valid start and end markers; skipped byte ranges and subsequent timing gaps
are recorded in each pulse file. Set `max_resync_bytes` to zero for strict
processing. Unrecoverable boundaries, unsupported modes, invalid clocks and
truncated frames stop processing. IQ values resembling a header marker alone
do not create spurious frames.

The analytic reference matches the 200-microsecond detection reference in
`matched_filter.py`:
`s(t) = exp[2 pi i (B t/2 - B t^2/(2 T))]`, inside the pulse, zero outside.
The valid correlation is `y[k] = sum_j z[k+j] conj(s[j])`; lag `k` is the
echo-envelope start, with no hidden convolution centering correction. The
measured delay is `gate_front_us + k/fs + delay_correction_us` (with consistent
units), and apparent **monostatic range** is `c*delay/2`, never altitude.
Doppler displaces this zero-Doppler correlation peak; these outputs do not yet
correct range-Doppler ambiguity or estimate a three-dimensional trajectory.

SNR is `10 log10(peak power / median power)` over the configured valid search
lags. This retains the legacy median-noise convention; it is not a mean-noise
SNR. The default threshold is 14 dB. Noise power is in unnormalized
matched-filter ADC-count-squared units. Each pulse contributes its largest peak;
simultaneous weaker echoes are not searched separately.

Consecutive detections form a candidate event if their time separation,
apparent-range change, IPP and pointing satisfy the configured limits. Tracks
shorter than the minimum pulse count or longer than the maximum duration are
discarded. Grouping uses UTC order across all files, not MPI rank. These are
**candidate meteor events**: persistent clutter, interference and satellites
still need scientific screening. No calibrated RCS, mass or orbit is inferred
from monostatic detection alone.
