# Data Conventions

## Binary Input and Delay Origin

The supported format is little endian: 1024-byte header, declared IQ payload,
then configurable trailing bytes (eight in the supplied November recordings).
Frames require both synchronization words, one DBF output beam, 430 MHz carrier,
4 MHz sampling, 4 MHz LFM bandwidth and 200 microsecond pulses. Unsupported
layouts are rejected. Configured bounded resynchronization records skipped byte
ranges; it does not silently treat payload synchronization-like words as frames.

With `allow_orphaned_tail=true`, an unheadered or incomplete EOF block is
reported with byte offset/count and left in the original recording, not
assigned a guessed pulse time. A complete declared IQ payload is retained even
if only the unused trailing bytes are missing. Set the policy to false for
strict EOF errors. Unsupported valid-header experiment layouts still fail.

**Header bytes 766-767 contain the unsigned gate-start delay in microseconds.**
This is 466 microseconds in the first supplied files. The first received sample
therefore starts at this delay, not zero. For original sample or correlation lag
`k` (zero-based),

```text
delay_us = gate_front_us + k / 4 + delay_correction_us
total_path_length_km = 0.299792458 * delay_us
monostatic_range_km = total_path_length_km / 2
```

The supplied MATLAB reader explicitly labels the gate-start field `us`; the
Python reference uses the same origin in its range calculation. Every frame's
gate-start delay is preserved as `pulses/gate_front_us`. A voltage cut's first
absolute delay is `raw/first_delay_us`; the matched-filter cut's first absolute
delay is `matched_filter/first_delay_us`. Both include the original header gate
origin and the cut's sample offset. The hardware correction is separate,
defaults to zero, and is marked uncalibrated. No old experiment's correction is
silently applied.

Monostatic range is not altitude. At remote receivers the default coordinate is
the full bistatic path length, not distance from that receiver to the meteor.
Range padding and RANSAC tolerances use the chosen coordinate's kilometres.
The corresponding coordinate convention and conversion are stored in HDF5.

## Waveform and Power

Recorder words alternate signed int32 I, Q. Both supplied readers use complex
voltage **Q + i I**; this program preserves that ordering and sign convention.
The analytic downchirp is `exp[2 pi i (B t/2 - B t^2/(2 T))]` within the pulse,
zero outside. Its direction is explicitly configured, not guessed from the
header. Correlation is `y[k] = sum_j z[k+j] conj(s[j])`, with lag `k` marking
the echo-envelope start. There is no hidden convolution-centering correction.
Positive Doppler displaces this downchirp correlation peak to larger delay.
The detected range remains apparent, without a Doppler correction.

Detection SNR is peak matched-filter power divided by median matched-filter
power in the configured search window, expressed in dB. The default threshold
is 14 dB. A deterministic RANSAC first-order fit removes range outliers before
event acceptance; `range_slack_km`, `ransac_min_fraction` and the minimum pulse
count control acceptance. Both inlier flags and the fitted line are retained.
One strongest peak per pulse is searched; simultaneous weaker targets are not
independently detected.

Per-pulse raw noise is `median(|z|^2)/ln(2)`, using all recorded samples unless
absolute noise-delay bounds are configured. This estimates mean complex noise
power for zero-mean circular Gaussian noise. The raw arithmetic mean power is
also saved, including signals/interference. The matched-filter noise series is
the matched-filter median divided by ln(2), in unnormalized correlated-output
ADC-count-squared units. Noise-only mode skips this latter measurement.
100-pulse averages split at true time gaps, IPP changes, receiver pointing
changes or gate-start changes; they are not restricted to detected events.

## Event Files

Times are UTC nanoseconds since Unix epoch. Original header clock UTC offset is
in the configuration. Never derive observation times from filenames. Names
ending `_utf8` contain UTF-8 bytes. Original frame headers are concatenated in
`raw/frame_headers_u8`, 1024 bytes per saved pulse.

`raw/iq_words_iq` retains signed recorder words. `raw/pulse_offsets` indexes IQ
pairs, not individual words; pulse `k` spans pairs `[offset[k],offset[k+1])`.
`raw/first_sample[k]` is the first original recorder sample in that cut.
Matched-filter real and imaginary arrays have their own `pulse_offsets` and
`first_lag_sample`. Their support is never inferred from the detected lag of a
weak pulse; it follows the fitted track.

`ransac_inlier` identifies detection pulses used to define the track.
`gap_before_ns` records actual time gaps; zero means no such gap before that
pulse. `range_padding_truncated` flags gate-edge clipping per pulse, and
`time_padding_truncated` flags missing coverage at the two requested ends.
Original counts and arrays are not clipped to hide missing or negative data.

Some recorder files contain a future-dated first header followed by a regular
pulse sequence. `timing_valid` flags a maximum strictly-increasing subsequence
within each input file; rejected timestamps are preserved, not reconstructed.
These frames remain in the per-file power/index product and all-pulse noise
table, but are excluded from event timing and averaged sky-noise measurements.

Configuration, build version, processed-header hashes and source paths are
metadata only. No script paths or generator credits are rendered in figures.
