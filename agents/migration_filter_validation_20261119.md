# Raw-cut replay and migration-derived Doppler validation

Native Rust diagnostics; original recordings are not opened by the replay.

- `cargo test --locked --bin syisr-native`: all 18 processor tests pass,
  including full delay/Doppler bank recovery for both chirps and Doppler signs.
- The fractional-delay search separately recovers a known fractional echo.
- `validate_migration`: 768 deterministic noisy moving-echo trials, 12 pulses
  per trial, both chirp signs, Dopplers +/-100 and +/-200 kHz, fractional delays,
  independently random pulse phases, and coherent matched-filter SNRs 20, 30,
  and 60 dB. Circular complex Gaussian noise has E|n|^2=1.
- Fit the zero-Doppler integer peak lag against time, convert its slope to
  Doppler, demodulate each original voltage pulse, and search delay again.
- Maximum Doppler error: 1012.2 Hz. Maximum corrected range RMS: 14.85 m.
  Mean uncorrected/corrected RMS at 20 dB: 1124.18/11.29 m; mean gain 0.250 dB.
  At 30 dB: 1124.09/11.14 m, 0.257 dB. At 60 dB: 1124.34/11.20 m, 0.260 dB.
  These are recovery errors on the integer sample grid, not confidence intervals.
- Real events 1, 100, 1000, 3000, 6000: all cached complex matched-filter
  responses reproduced within 2.2e-7 relative RMS using cuts alone.
- Migration Dopplers: 32.000, 4.521, 125.314, 91.281, 158.310 kHz respectively.
- Strongest-pulse gains after fractional-delay refinement: 0.143, 0.000,
  0.140, 0.229, 0.230 dB. Integer-grid gains can be negative due to scalloping;
  do not interpret those as intrinsic performance loss.
- For each event a full strongest-pulse delay/Doppler bank covers -250 to
  +250 kHz in 5 kHz steps, with the normal 4 MHz delay sample spacing.
  Rendered bank shows the expected LFM ambiguity ridge. The orange line is
  the migration-derived Doppler, not a free per-pulse Doppler estimate.

Outputs live beside the full-run noise product on the external disk:
`migration_validation.h5`, `replay_event_NNNNNN.h5`, comparison PNGs and bank PNGs.
The diagnostic HDF5 files contain both integer and refined peak powers, peak
range coordinates, times, the bank, and replay error.

This is a constant-range-rate diagnostic. Acceleration affects the derivative
of the LFM range bias. It does not validate absolute receiver delay calibration,
phase coherence across pulses, deceleration, or dynamic masses. It does not
change the original detection catalogue or automatically process all events.
