# Continuous Station Processing Validation

The active native program is syisr_native on main, not a new processing fork.
Its event renderer is integrated; the prototype monostatic-events plotting
binary in rustplotlib_figures was removed. Rustplotlib is pinned to a published
Git revision, not a local checkout. No Python is required by the pipeline.

Header gate front at byte 766 is microseconds (explicit in the supplied MATLAB
reader). Both readers agree. It is 466 us in the first files. Detection and cut
origins include it; no old hardware-delay calibration is applied. Remote
station configurations use full bistatic path, not c*tau/2 or altitude.

The new cuts contain all available pulses, with +/-0.1 s time padding and
+/-20 km around a deterministic first-order RANSAC track. Original signed int32
IQ pairs are retained directly. Extra matched-filter arrays serve plotting.
Clipped recorder gates, clipped temporal padding and actual time gaps are
flagged. No values or missing pulses are synthesized.

Pilot: first three files, 19822 pulses, 49 threshold crossings, three accepted
tracks. The new cuts contain 72, 53 and 51 pulses, at uninterrupted 5 ms spacing;
their RANSAC inlier counts are 26, 6 and 11. The requested 20 km half-width
reaches the recording gate edge for some pulses, which is explicitly flagged.
327324 saved original IQ pairs were checked byte-for-byte against raw files;
all match. Cut first_delay_us equals gate_front_us + first_sample/4.

Noise product covers all 19822 pilot pulses, with 200 averaging blocks whose
counts sum to 19822. Serial and two-rank MPI runs agree exactly across 275
compared datasets (excluding output/plot configuration bytes). Noise-only raw
noise and normal-processing raw noise agree exactly, pulse-level and averaged.
The added instrument/source-mtime metadata do not alter the numerical arrays.

12 unit/integration tests pass with MPI/plots and without default features.
Clippy all-targets with MPI passes without warnings. Tests include a weak
undetected pulse retained inside a two-file cut, remote path coordinates,
RANSAC outliers, noise normalization, and refusing resume when source or
configuration changes. Resume also checks the original executable revision.

The inspected delivered pilot plot is
syisr_native/output-continuous-mpi/event_cuts/plots/event_000001.png.
It has no detection-induced white columns and shows the requested time padding.
Actual remote recordings and CentOS have not been tested; support is validated
against the common format and synthetic remote-coordinate tests.
