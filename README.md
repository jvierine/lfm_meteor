# lfm_meteor

Processing scripts for the Sanya tri-static LFM meteor head-echo data.

## Raw MATLAB Files

The raw experiment files are MATLAB v7.3/HDF5 files under directories such as:

```text
/mnt/data/juha/SANYA/Juha/20240422/Sanya/
/mnt/data/juha/SANYA/Juha/20240422/Danzhou/
/mnt/data/juha/SANYA/Juha/20240422/Wenchang/
```

Each raw file contains three important variables:

- `data_raw`: raw complex IQ voltage data indexed by range sample and pulse time.
- `para`: experiment configuration parameters.
- `time`: Beijing local time, UTC+8, with shape `[7, N]`.

The `time` rows are:

```text
[year_since_2000, month, day, hour, minute, second, code]
```

Important: raw `time` is not UTC. Processing code must subtract 8 hours before using timestamps for celestial coordinates, GCRS/ITRS transforms, solar longitude, or radiant calculations.

Useful `para` entries:

```text
para[6]   azimuth, deg
para[7]   elevation, deg
para[10]  LFM pulse width, us
para[11]  IPP, us
para[12]  raw gate start, km
para[13]  raw gate end, km
para[14]  sampling rate, MHz
para[15]  bandwidth, MHz
```

## Time Convention

Processed event files should use:

```text
times_ns = UTC nanoseconds since Unix epoch
beijing_local_time_ns = original raw MATLAB local-time nanoseconds
```

New files written by `explore.py` or `matched_filter.py` include HDF5 attributes:

```text
times_ns_time_scale = "UTC"
source_time_zone = "Beijing local time (UTC+8)"
source_time_correction = "times_ns = raw MATLAB time - 8 hours"
source_timezone_offset_hours = 8
```

Legacy local tri-static files that lacked this metadata were patched with:

```bash
python3 fix_local_tristatic_time_metadata.py
```

That script is idempotent. It skips files already marked with `times_ns_time_scale = "UTC"`.

## Event HDF5 Files

Matched-filter event files live in either:

```text
results/head_echoes/<site>/<site>_<event>.h5
results/tristatic_head_echoes/<site>/<site>_<event>.h5
```

Common datasets:

```text
times_ns                 UTC time for each detected pulse, ns
beijing_local_time_ns    original raw local time, ns
relative_time_s          seconds from event start
echoes                   matched-filtered echo snippets
raw                      raw voltage snippets
range_gate               selected range-gate index per pulse
range_km                 selected half-path/range coordinate per pulse
ranges_km_axis           range coordinate for all gates
snr_peak_db              peak SNR-like matched-filter power, dB
az, el                   pointing azimuth/elevation, deg
r0, r1                   processed range-axis start/end, km
sr_mhz                   sample rate, MHz
bw_mhz                   LFM bandwidth, MHz
ipp_us                   interpulse period, us
pulse_length_us          LFM pulse length, us
source_file              raw MATLAB file path
event_id                 event identifier
rti_png                  diagnostic RTI plot path
```

For current processing, `times_ns` is UTC. If an older file has no `times_ns_time_scale` attribute, treat `times_ns` as legacy Beijing-local-derived time and subtract 8 hours before inertial or celestial calculations.

## Tri-static Event Index

The selected tri-static index is:

```text
results/tristatic_event_index.h5
```

Key datasets:

```text
sanya_event_id, danzhou_event_id, wenchang_event_id
sanya_event_h5, danzhou_event_h5, wenchang_event_h5
sanya_dt0_ns, danzhou_dt0_ns, wenchang_dt0_ns
sanya_dt0_ns_beijing_local
danzhou_dt0_ns_beijing_local
wenchang_dt0_ns_beijing_local
sanya_delay_us, danzhou_delay_us, wenchang_delay_us
```

The `*_dt0_ns` datasets are UTC in the patched/current index. The `*_dt0_ns_beijing_local` datasets preserve the original local timestamps.

## Delays and Path Coordinates

The LFM matched-filter delay coordinate is converted to a total propagation path for fitting:

```text
total path = c * delay
```

The trajectory fit uses total transmitter-target-receiver path length, not monostatic one-way range.

For Sanya monostatic:

```text
L_S = |target - Sanya| + |target - Sanya|
```

For Danzhou/Wenchang bistatic links:

```text
L_D = |target - Sanya| + |target - Danzhou|
L_W = |target - Sanya| + |target - Wenchang|
```

The diagnostic half-path coordinate is only:

```text
half_path = total_path / 2
```

Do not interpret the remote half-path coordinate as a physical one-way range.

Current first-sample delays:

```text
Sanya     466.32 us
Danzhou   438.426 us
Wenchang  430.906 us
```

## GCRS Trajectory Fit Product

The main trajectory product is:

```text
results/gcrs_trajectory_fits_lfm_ambiguity_v20260610.h5
```

It is produced by:

```bash
python3 fit_gcrs_trajectories_lfm_ambiguity.py
```

Root-level datasets:

```text
event_id
t0_ns, t0_utc
t0_beijing_local_ns, t0_beijing_local
n_points, duration_s
r0_gcrs_m, v0_gcrs_mps
r0_prior_gcrs_m, v0_prior_gcrs_mps
speed_km_s, prior_speed_km_s
start_alt_km, end_alt_km
rms_total_path_residual_m
median_abs_total_path_residual_m
rms_half_path_diagnostic_residual_m
median_abs_half_path_diagnostic_residual_m
optimizer_success, optimizer_nfev
link_names
link_tx_positions_m
link_rx_positions_m
```

Per-trajectory groups live under:

```text
points/<event_id>/
```

Common per-point datasets:

```text
time_ns                    UTC pulse time, ns
beijing_local_time_ns       original local pulse time, ns
t_rel_s
itrs_fit_m
itrs_fit_v_mps
prior_points_ecef_m
prior_points_gcrs_m
lat_deg, lon_deg, alt_km
measured_total_paths_m
predicted_total_paths_m
total_path_residuals_m
measured_half_path_diagnostic_m
predicted_half_path_diagnostic_m
half_path_diagnostic_residuals_m
```

Relevant attributes:

```text
fit_residual_coordinate = "total propagation path length"
coordinate_frame = "GCRS"
model = "constant velocity, no deceleration; LFM range-Doppler ambiguity included"
source_time_zone = "Beijing local time, UTC+8"
source_time_correction = "UTC time_ns = raw MATLAB local time_ns - 8 hours"
```

## Radiant Product

The sun-centered radiant product is:

```text
results/sun_centered_ecliptic_radiants_v20260610.h5
```

It is produced by:

```bash
python3 plot_sun_centered_ecliptic_radiants.py
```

Datasets:

```text
event_id
t0_ns
speed_km_s
rms_total_path_residual_m
n_points
lambda_ecliptic_deg
beta_ecliptic_deg
sun_lambda_ecliptic_deg
lambda_minus_sun_deg
```

Conventions:

```text
radiant direction = -v0_gcrs
ecliptic frame = GeocentricTrueEcliptic
lambda_minus_sun_deg = lambda_radiant - lambda_sun, wrapped to [0, 360) deg
```

In this convention:

```text
Helion       0 deg
Antihelion  180 deg
Anti-apex   90 deg
Apex        270 deg
```

Figures:

```text
results/sun_centered_ecliptic_radiants.png
results/sun_centered_ecliptic_radiants_diagnostic.png
```

The plot is centered on apex longitude 270 deg and the horizontal axis is mirrored for the chosen meteor-radiant display convention.

## Velocity Distribution Product

The geocentric speed distribution product is:

```text
results/geocentric_velocity_distribution_v20260611.h5
```

It is produced by:

```bash
python3 plot_geocentric_velocity_distribution.py
```

Datasets:

```text
event_id
speed_km_s
n_points
rms_total_path_residual_m
histogram_bin_edges_km_s
histogram_counts
```

The speeds are fitted GCRS geocentric velocity magnitudes.

## Useful Scripts

```text
explore.py
matched_filter.py
select_tristatic_events.py
fix_local_tristatic_time_metadata.py
fit_gcrs_trajectories_lfm_ambiguity.py
plot_sun_centered_ecliptic_radiants.py
plot_geocentric_velocity_distribution.py
plot_lfm_corrected_vs_uncorrected_positions.py
simulate_lfm_range_doppler_test.py
```

Use the local conda environment when running analysis scripts:

```bash
source /opt/anaconda3/bin/activate base
```
