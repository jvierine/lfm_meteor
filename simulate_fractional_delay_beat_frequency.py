"""Validate dechirped beat-frequency recovery of fractional LFM delay offsets."""

import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.signal as sig

import fit_event_joint_delay_doppler_fft as joint
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import sanya_opts as sc


SCRIPT_VERSION = "v20260629a"
OUTPUT_BASE = os.path.join("results", f"memo25_fractional_delay_beat_frequency_{SCRIPT_VERSION}")
PAPER_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/memos/figures"
PAPER_FIGURE_BASE = os.path.join(PAPER_FIGURE_DIR, f"memo25_fractional_delay_beat_frequency_{SCRIPT_VERSION}")
PAPER_TRAJECTORY_FIGURE_BASE = os.path.join(
    PAPER_FIGURE_DIR,
    f"memo25_constant_radial_velocity_joint_fit_{SCRIPT_VERSION}",
)
PAPER_NOISY_TRAJECTORY_FIGURE_BASE = os.path.join(
    PAPER_FIGURE_DIR,
    f"memo25_constant_radial_velocity_joint_fit_snr18db_{SCRIPT_VERSION}",
)

TRUTH_SR_MHZ = 40.0
MEASUREMENT_SR_MHZ = 4.0
DOWNSAMPLE = int(round(TRUTH_SR_MHZ / MEASUREMENT_SR_MHZ))
PULSE_LENGTH_US = 199.0
BANDWIDTH_MHZ = 4.0
REFERENCE_CHIRP_RATE_SCALE = gfit.REFERENCE_CHIRP_RATE_SCALE
ZERO_PAD_FACTOR = 512
GATE_UPSAMPLE_FACTOR = 32
TIME_PAD_US = 50.0
REFERENCE_GATE = 500.0
FRACTIONAL_OFFSETS_4MHZ = np.asarray(
    [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
    dtype=np.float64,
)
N_TRAJECTORY_PULSES = 20
IPP_S = 0.005
SANYA_RADIAL_VELOCITY_MPS = 50.0e3
SANYA_TOTAL_PATH_RATE_MPS = 2.0 * SANYA_RADIAL_VELOCITY_MPS
TRAJECTORY_INITIAL_OFFSET_SAMPLES = -0.25
PATH_SIGMA_M = 0.25
BEAT_SIGMA_HZ = 25.0
NOISY_TRAJECTORY_SNR_DB = 18.0
NOISY_TRAJECTORY_RNG_SEED = 20260629


def high_rate_lfm(length_us=PULSE_LENGTH_US, sr_mhz=TRUTH_SR_MHZ, bandwidth_mhz=BANDWIDTH_MHZ):
    n = int(round(float(length_us) * float(sr_mhz)))
    t_s = np.arange(n, dtype=np.float64) / (float(sr_mhz) * 1e6)
    bandwidth_hz = float(bandwidth_mhz) * 1e6
    sweep_rate = (
        bandwidth_hz
        * 1e6
        / float(length_us)
        / 2.0
        * float(REFERENCE_CHIRP_RATE_SCALE)
    )
    phase_cycles = t_s * bandwidth_hz / 2.0 - sweep_rate * t_s**2.0
    return np.exp(1j * 2.0 * np.pi * phase_cycles).astype(np.complex128)


def continuous_lfm(t_s):
    t = np.asarray(t_s, dtype=np.float64)
    bandwidth_hz = BANDWIDTH_MHZ * 1e6
    sweep_rate = (
        bandwidth_hz
        * 1e6
        / PULSE_LENGTH_US
        / 2.0
        * REFERENCE_CHIRP_RATE_SCALE
    )
    phase_cycles = t * bandwidth_hz / 2.0 - sweep_rate * t**2.0
    return np.exp(1j * 2.0 * np.pi * phase_cycles)


def simulate_measurement(fractional_offset_4mhz):
    """Return a 4 MHz measurement made from a 40 MHz delayed zero-Doppler echo."""
    code40 = high_rate_lfm()
    n_code40 = len(code40)
    n_code4 = int(round(PULSE_LENGTH_US * MEASUREMENT_SR_MHZ))
    reference_gate40 = int(round(REFERENCE_GATE * DOWNSAMPLE))
    reference_start40 = reference_gate40 - n_code40 // 2
    true_start40 = reference_start40 + int(round(float(fractional_offset_4mhz) * DOWNSAMPLE))
    pad40 = int(round(120.0 * TRUTH_SR_MHZ))
    n_raw40 = reference_start40 + n_code40 + pad40
    raw40 = np.zeros(n_raw40, dtype=np.complex128)
    raw40[true_start40 : true_start40 + n_code40] = code40
    raw4 = raw40[::DOWNSAMPLE].astype(np.complex128)
    return raw4


def simulate_doppler_measurement(total_path_offset_m, total_path_rate_mps):
    """Simulate one Sanya zero-acceleration pulse on a 40 MHz truth grid."""
    n_code40 = int(round(PULSE_LENGTH_US * TRUTH_SR_MHZ))
    reference_gate40 = int(round(REFERENCE_GATE * DOWNSAMPLE))
    reference_start40 = reference_gate40 - n_code40 // 2
    pad40 = int(round(160.0 * TRUTH_SR_MHZ))
    n_raw40 = reference_start40 + n_code40 + pad40
    n = np.arange(n_raw40, dtype=np.float64)
    t_ref_s = (n - float(reference_start40)) / (TRUTH_SR_MHZ * 1e6)
    delay_s = float(total_path_offset_m) / gfit.C
    pulse_t_s = t_ref_s - delay_s
    raw40 = np.zeros(n_raw40, dtype=np.complex128)
    keep = (pulse_t_s >= 0.0) & (pulse_t_s < PULSE_LENGTH_US * 1e-6)
    doppler_hz = -float(total_path_rate_mps) / sc.RADAR_WAVELENGTH_M
    raw40[keep] = continuous_lfm(pulse_t_s[keep]) * np.exp(1j * 2.0 * np.pi * doppler_hz * t_ref_s[keep])
    return raw40[::DOWNSAMPLE].astype(np.complex128), doppler_hz


def matched_filter_noise_power_for_snr(snr_db):
    n_code = int(round(PULSE_LENGTH_US * MEASUREMENT_SR_MHZ))
    return float(n_code) / (10.0 ** (float(snr_db) / 10.0))


def add_complex_voltage_noise(row4, rng, snr_db):
    noise_power = matched_filter_noise_power_for_snr(snr_db)
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(row4)) + 1j * rng.standard_normal(len(row4)))
    return np.asarray(row4, dtype=np.complex128) + noise.astype(np.complex128)


def measurement_time_axis_s(n_samples):
    reference_gate40 = int(round(REFERENCE_GATE * DOWNSAMPLE))
    reference_start40 = reference_gate40 - int(round(PULSE_LENGTH_US * TRUTH_SR_MHZ)) // 2
    sample40 = np.arange(n_samples, dtype=np.float64) * DOWNSAMPLE
    return (sample40 - float(reference_start40)) / (TRUTH_SR_MHZ * 1e6)


def doppler_model_row(t_ref_s, total_path_offset_m, total_path_rate_mps):
    delay_s = float(total_path_offset_m) / gfit.C
    pulse_t_s = t_ref_s - delay_s
    model = np.zeros(len(t_ref_s), dtype=np.complex128)
    keep = (pulse_t_s >= 0.0) & (pulse_t_s < PULSE_LENGTH_US * 1e-6)
    doppler_hz = -float(total_path_rate_mps) / sc.RADAR_WAVELENGTH_M
    model[keep] = continuous_lfm(pulse_t_s[keep]) * np.exp(1j * 2.0 * np.pi * doppler_hz * t_ref_s[keep])
    return model


def matched_filter_gate(row4):
    n_code = int(round(PULSE_LENGTH_US * MEASUREMENT_SR_MHZ))
    code = joint.lfm_reference_for_offsets(
        np.arange(n_code, dtype=np.float64),
        MEASUREMENT_SR_MHZ,
        BANDWIDTH_MHZ * 1e6,
        PULSE_LENGTH_US,
        chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE,
    )
    corr = sig.fftconvolve(np.asarray(row4, dtype=np.complex128), np.conj(code), mode="same")
    power = np.abs(corr) ** 2.0
    lo = max(0, int(round(REFERENCE_GATE)) - 120)
    hi = min(len(power), int(round(REFERENCE_GATE)) + 220)
    peak_idx = int(lo + np.nanargmax(power[lo:hi]))
    gate = float(peak_idx)
    if 0 < peak_idx < len(power) - 1:
        ym1, y0, yp1 = np.log(np.maximum(power[peak_idx - 1 : peak_idx + 2], 1e-300))
        denom = ym1 - 2.0 * y0 + yp1
        if np.isfinite(denom) and abs(denom) > 1e-30:
            gate += float(np.clip(0.5 * (ym1 - yp1) / denom, -1.0, 1.0))
    return gate


def dechirped_diagnostic(row4):
    beat_hz, fft_bin_hz, fourier_resolution_hz, prominence_db = joint.dechirped_fft_offset_hz(
        row4,
        REFERENCE_GATE,
        MEASUREMENT_SR_MHZ,
        BANDWIDTH_MHZ,
        PULSE_LENGTH_US,
        ZERO_PAD_FACTOR,
        gate_upsample_factor=GATE_UPSAMPLE_FACTOR,
        time_pad_us=TIME_PAD_US,
        chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE,
    )
    chirp_rate_hz_per_s = gfit.NOMINAL_CHIRP_RATE_HZ_PER_S * REFERENCE_CHIRP_RATE_SCALE
    recovered_delay_s = beat_hz / chirp_rate_hz_per_s
    recovered_offset_samples = recovered_delay_s * MEASUREMENT_SR_MHZ * 1e6
    recovered_path_m = gfit.C * recovered_delay_s
    return beat_hz, recovered_offset_samples, recovered_path_m, fft_bin_hz, fourier_resolution_hz, prominence_db


def run_constant_velocity_fit(noise_snr_db=None, rng_seed=NOISY_TRAJECTORY_RNG_SEED):
    times_s = np.arange(N_TRAJECTORY_PULSES, dtype=np.float64) * IPP_S
    true_initial_offset_m = gfit.C * TRAJECTORY_INITIAL_OFFSET_SAMPLES / (MEASUREMENT_SR_MHZ * 1e6)
    true_path_m = true_initial_offset_m + SANYA_TOTAL_PATH_RATE_MPS * times_s
    rng = np.random.default_rng(int(rng_seed)) if noise_snr_db is not None else None
    noise_power = matched_filter_noise_power_for_snr(noise_snr_db) if noise_snr_db is not None else 0.0
    rows4 = []
    gate = []
    beat_hz = []
    fft_bin_hz = []
    prominence_db = []
    for path_m in true_path_m:
        row4, _doppler_hz = simulate_doppler_measurement(path_m, SANYA_TOTAL_PATH_RATE_MPS)
        if rng is not None:
            row4 = add_complex_voltage_noise(row4, rng, noise_snr_db)
        rows4.append(row4)
        peak_gate = matched_filter_gate(row4)
        gate.append(peak_gate)
        beat, fft_bin, _fourier_res, prom = joint.dechirped_fft_offset_hz(
            row4,
            peak_gate,
            MEASUREMENT_SR_MHZ,
            BANDWIDTH_MHZ,
            PULSE_LENGTH_US,
            ZERO_PAD_FACTOR,
            gate_upsample_factor=GATE_UPSAMPLE_FACTOR,
            time_pad_us=TIME_PAD_US,
            chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE,
        )
        beat_hz.append(beat)
        fft_bin_hz.append(fft_bin)
        prominence_db.append(prom)
    gate = np.asarray(gate, dtype=np.float64)
    beat_hz = np.asarray(beat_hz, dtype=np.float64)
    fft_bin_hz = np.asarray(fft_bin_hz, dtype=np.float64)
    prominence_db = np.asarray(prominence_db, dtype=np.float64)
    measured_path_m = gfit.C * (gate - REFERENCE_GATE) / (MEASUREMENT_SR_MHZ * 1e6)
    chirp_rate_hz_per_s = gfit.NOMINAL_CHIRP_RATE_HZ_PER_S * REFERENCE_CHIRP_RATE_SCALE
    t_ref_s = measurement_time_axis_s(len(rows4[0]))
    rows4 = [np.asarray(row, dtype=np.complex128) for row in rows4]

    def residual(params):
        initial_path_m, total_path_rate_mps = params
        parts = []
        for pulse_time_s, row in zip(times_s, rows4):
            path_m = initial_path_m + total_path_rate_mps * pulse_time_s
            model = doppler_model_row(t_ref_s, path_m, total_path_rate_mps)
            norm = np.vdot(model, model).real
            if norm <= 0.0:
                parts.append(np.ones(2 * len(row), dtype=np.float64) * 1e6)
                continue
            amp = np.vdot(model, row) / norm
            diff = amp * model - row
            parts.append(np.r_[diff.real, diff.imag] / np.sqrt(norm))
        return np.concatenate(parts)

    slope, intercept = np.polyfit(times_s, measured_path_m, 1)
    doppler_guess_hz = -slope / sc.RADAR_WAVELENGTH_M
    ambiguity_corrected_intercept = intercept - gfit.C * doppler_guess_hz / chirp_rate_hz_per_s
    rate_grid = slope * np.asarray([0.98, 1.0, 1.02], dtype=np.float64)
    intercept_grid = ambiguity_corrected_intercept + np.arange(-2500.0, 2500.1, 50.0)
    best_seed = np.asarray([intercept, slope], dtype=np.float64)
    best_seed_cost = np.inf
    for rate0 in rate_grid:
        for intercept0 in intercept_grid:
            trial = np.asarray([intercept0, rate0], dtype=np.float64)
            trial_residual = residual(trial)
            trial_cost = 0.5 * float(np.dot(trial_residual, trial_residual))
            if trial_cost < best_seed_cost:
                best_seed_cost = trial_cost
                best_seed = trial

    result = so.least_squares(
        residual,
        best_seed,
        x_scale=np.asarray([100.0, 100.0e3], dtype=np.float64),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    fit_geo_m = result.x[0] + result.x[1] * times_s
    fit_total_path_rate_mps = float(result.x[1])
    fit_radial_velocity_mps = 0.5 * fit_total_path_rate_mps
    fit_doppler_hz = float(-fit_total_path_rate_mps / sc.RADAR_WAVELENGTH_M)
    beat_at_fit_range_hz = []
    for row, path_m in zip(rows4, fit_geo_m):
        gate_fit = REFERENCE_GATE + path_m * (MEASUREMENT_SR_MHZ * 1e6) / gfit.C
        beat, _fft_bin, _fourier_res, _prom = joint.dechirped_fft_offset_hz(
            row,
            gate_fit,
            MEASUREMENT_SR_MHZ,
            BANDWIDTH_MHZ,
            PULSE_LENGTH_US,
            ZERO_PAD_FACTOR,
            gate_upsample_factor=GATE_UPSAMPLE_FACTOR,
            time_pad_us=TIME_PAD_US,
            chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE,
        )
        beat_at_fit_range_hz.append(beat)
    beat_at_fit_range_hz = np.asarray(beat_at_fit_range_hz, dtype=np.float64)
    fit_beat_hz = np.full_like(times_s, fit_doppler_hz, dtype=np.float64)
    return {
        "time_s": times_s,
        "true_initial_offset_m": float(true_initial_offset_m),
        "true_total_path_m": true_path_m,
        "true_total_path_rate_mps": float(SANYA_TOTAL_PATH_RATE_MPS),
        "true_radial_velocity_mps": float(SANYA_RADIAL_VELOCITY_MPS),
        "true_doppler_hz": float(-SANYA_TOTAL_PATH_RATE_MPS / sc.RADAR_WAVELENGTH_M),
        "noise_snr_db": float(noise_snr_db) if noise_snr_db is not None else np.nan,
        "rng_seed": int(rng_seed) if noise_snr_db is not None else -1,
        "complex_voltage_noise_power": float(noise_power),
        "gate": gate,
        "matched_filter_path_m": measured_path_m,
        "matched_filter_beat_hz": beat_hz,
        "observed_beat_hz": beat_at_fit_range_hz,
        "fft_bin_hz": fft_bin_hz,
        "prominence_db": prominence_db,
        "fit_params": result.x,
        "fit_total_path_m": fit_geo_m,
        "fit_beat_hz": fit_beat_hz,
        "fit_total_path_rate_mps": fit_total_path_rate_mps,
        "fit_radial_velocity_mps": fit_radial_velocity_mps,
        "fit_doppler_hz": fit_doppler_hz,
        "seed_initial_path_m": float(best_seed[0]),
        "seed_total_path_rate_mps": float(best_seed[1]),
        "seed_cost": float(best_seed_cost),
        "optimizer_cost": float(result.cost),
        "optimizer_success": bool(result.success),
        "path_residual_m": fit_geo_m - true_path_m,
        "beat_residual_hz": fit_beat_hz - beat_at_fit_range_hz,
    }


def one_spectrum(row4, freq_limit_khz=40.0):
    if GATE_UPSAMPLE_FACTOR > 1:
        import scipy.signal as sig

        row_work = sig.resample_poly(row4, GATE_UPSAMPLE_FACTOR, 1).astype(np.complex128)
        sr_work_mhz = MEASUREMENT_SR_MHZ * GATE_UPSAMPLE_FACTOR
        center = int(round(REFERENCE_GATE * GATE_UPSAMPLE_FACTOR))
    else:
        row_work = np.asarray(row4, dtype=np.complex128)
        sr_work_mhz = MEASUREMENT_SR_MHZ
        center = int(round(REFERENCE_GATE))
    n_code = int(round(PULSE_LENGTH_US * sr_work_mhz))
    pulse_start = center - n_code // 2
    pad_samples = int(round(TIME_PAD_US * sr_work_mhz))
    start = max(0, pulse_start - pad_samples)
    stop = min(len(row_work), pulse_start + n_code + pad_samples)
    segment = row_work[start:stop]
    sample_offsets = np.arange(start, stop, dtype=np.float64) - float(pulse_start)
    reference = joint.lfm_reference_for_offsets(
        sample_offsets,
        sr_work_mhz,
        BANDWIDTH_MHZ * 1e6,
        PULSE_LENGTH_US,
        chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE,
    )
    deramped = segment * np.conj(reference)
    n_fft = 1
    while n_fft < ZERO_PAD_FACTOR * len(deramped):
        n_fft *= 2
    windowed = deramped * np.hanning(len(deramped))
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / (sr_work_mhz * 1e6)))
    power_db = 10.0 * np.log10(np.maximum(np.abs(spectrum) ** 2.0, 1e-300))
    power_db -= np.nanmax(power_db)
    keep = np.abs(freq_hz) <= freq_limit_khz * 1e3
    return deramped, freq_hz[keep], power_db[keep]


def write_trajectory_group(parent, name, trajectory):
    tg = parent.create_group(name)
    for key, value in trajectory.items():
        if np.isscalar(value) or isinstance(value, (str, bytes, bool)):
            tg.attrs[key] = value
        else:
            tg[key] = value


def plot_trajectory_fit(trajectory, output_base, noisy=False):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, constrained_layout=True)
    time_ms = trajectory["time_s"] * 1e3
    axes[0].plot(
        time_ms,
        trajectory["matched_filter_path_m"],
        "o",
        ms=3.5,
        color="#1f77b4",
        alpha=0.65,
        label="1-D matched-filter peak",
    )
    axes[0].plot(time_ms, trajectory["fit_total_path_m"], "-", color="0.2", lw=1.2, label="joint raw-voltage fit")
    axes[0].plot(time_ms, trajectory["true_total_path_m"], "--", color="#2ca02c", lw=1.0, label="true geometric path")
    axes[0].set_ylabel("Total-path offset (m)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    title = (
        f"20 pulses, SNR {trajectory['noise_snr_db']:.0f} dB: "
        if noisy
        else "20 noiseless pulses: "
    )
    axes[0].set_title(
        title
        + f"radial speed {SANYA_RADIAL_VELOCITY_MPS/1e3:.0f} km/s, "
        + f"fit {trajectory['fit_radial_velocity_mps']/1e3:.3f} km/s",
        fontsize=10,
    )

    axes[1].plot(time_ms, trajectory["observed_beat_hz"] / 1e3, "o", ms=4, color="#d62728", label="FFT at fitted range")
    axes[1].plot(time_ms, trajectory["fit_beat_hz"] / 1e3, "-", color="0.2", lw=1.2, label="joint fit")
    axes[1].axhline(trajectory["true_doppler_hz"] / 1e3, color="#2ca02c", lw=1.0, ls="--", label="true Doppler")
    axes[1].set_xlabel("Pulse time (ms)")
    axes[1].set_ylabel("Beat frequency (kHz)")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Joint range and constant-Doppler recovery", fontsize=11)
    fig.savefig(f"{output_base}.png", dpi=220)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    os.makedirs(PAPER_FIGURE_DIR, exist_ok=True)

    rows4 = []
    beat_hz = []
    recovered_samples = []
    recovered_path_m = []
    fft_bin_hz = []
    fourier_resolution_hz = []
    prominence_db = []
    for offset in FRACTIONAL_OFFSETS_4MHZ:
        row4 = simulate_measurement(offset)
        rows4.append(row4)
        result = dechirped_diagnostic(row4)
        beat_hz.append(result[0])
        recovered_samples.append(result[1])
        recovered_path_m.append(result[2])
        fft_bin_hz.append(result[3])
        fourier_resolution_hz.append(result[4])
        prominence_db.append(result[5])

    beat_hz = np.asarray(beat_hz, dtype=np.float64)
    recovered_samples = np.asarray(recovered_samples, dtype=np.float64)
    recovered_path_m = np.asarray(recovered_path_m, dtype=np.float64)
    fft_bin_hz = np.asarray(fft_bin_hz, dtype=np.float64)
    fourier_resolution_hz = np.asarray(fourier_resolution_hz, dtype=np.float64)
    prominence_db = np.asarray(prominence_db, dtype=np.float64)
    true_path_m = gfit.C * FRACTIONAL_OFFSETS_4MHZ / (MEASUREMENT_SR_MHZ * 1e6)
    expected_beat_hz = (
        gfit.NOMINAL_CHIRP_RATE_HZ_PER_S
        * REFERENCE_CHIRP_RATE_SCALE
        * FRACTIONAL_OFFSETS_4MHZ
        / (MEASUREMENT_SR_MHZ * 1e6)
    )
    sample_error = recovered_samples - FRACTIONAL_OFFSETS_4MHZ
    path_error_m = recovered_path_m - true_path_m
    trajectory = run_constant_velocity_fit()
    noisy_trajectory = run_constant_velocity_fit(noise_snr_db=NOISY_TRAJECTORY_SNR_DB)

    with h5py.File(f"{OUTPUT_BASE}.h5", "w") as h:
        h.attrs["source_script"] = "/Users/jvi019/src/lfm_meteor/simulate_fractional_delay_beat_frequency.py"
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["truth_sample_rate_hz"] = TRUTH_SR_MHZ * 1e6
        h.attrs["measurement_sample_rate_hz"] = MEASUREMENT_SR_MHZ * 1e6
        h.attrs["pulse_length_us"] = PULSE_LENGTH_US
        h.attrs["bandwidth_hz"] = BANDWIDTH_MHZ * 1e6
        h.attrs["reference_chirp_rate_scale"] = REFERENCE_CHIRP_RATE_SCALE
        h.attrs["zero_doppler"] = True
        h["true_fractional_offset_samples_4mhz"] = FRACTIONAL_OFFSETS_4MHZ
        h["true_total_path_offset_m"] = true_path_m
        h["expected_beat_hz"] = expected_beat_hz
        h["estimated_beat_hz"] = beat_hz
        h["recovered_fractional_offset_samples_4mhz"] = recovered_samples
        h["recovered_total_path_offset_m"] = recovered_path_m
        h["fractional_offset_error_samples_4mhz"] = sample_error
        h["total_path_offset_error_m"] = path_error_m
        h["fft_bin_hz"] = fft_bin_hz
        h["fourier_resolution_hz"] = fourier_resolution_hz
        h["prominence_db"] = prominence_db
        write_trajectory_group(h, "constant_radial_velocity", trajectory)
        write_trajectory_group(h, "constant_radial_velocity_snr18db", noisy_trajectory)

    example_offset = 0.3
    example_index = int(np.where(np.isclose(FRACTIONAL_OFFSETS_4MHZ, example_offset))[0][0])
    deramped, freq_hz, power_db = one_spectrum(rows4[example_index])
    t_us = np.arange(len(deramped), dtype=np.float64) / (MEASUREMENT_SR_MHZ * GATE_UPSAMPLE_FACTOR)
    t_us -= np.nanmean(t_us)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    axes[0, 0].plot(FRACTIONAL_OFFSETS_4MHZ, recovered_samples, "o", color="#1f77b4", ms=5)
    lim = max(np.max(np.abs(FRACTIONAL_OFFSETS_4MHZ)), np.max(np.abs(recovered_samples))) * 1.08
    axes[0, 0].plot([-lim, lim], [-lim, lim], color="0.25", lw=1.0, ls="--")
    axes[0, 0].set_xlim(-lim, lim)
    axes[0, 0].set_ylim(-lim, lim)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_xlabel("True delay offset (4 MHz samples)")
    axes[0, 0].set_ylabel("Recovered offset (4 MHz samples)")
    axes[0, 0].set_title("Fractional-gate recovery")

    axes[0, 1].plot(FRACTIONAL_OFFSETS_4MHZ, path_error_m, "o-", color="#d62728", lw=1.2, ms=4)
    axes[0, 1].axhline(0.0, color="0.25", lw=1.0, ls="--")
    axes[0, 1].set_xlabel("True delay offset (4 MHz samples)")
    axes[0, 1].set_ylabel("Recovery error (m)")
    axes[0, 1].set_title("Total-path error")

    axes[1, 0].plot(t_us, deramped.real / np.nanmax(np.abs(deramped)), color="#1f77b4", lw=0.9, label="real")
    axes[1, 0].plot(t_us, deramped.imag / np.nanmax(np.abs(deramped)), color="#ff7f0e", lw=0.9, label="imag")
    axes[1, 0].set_xlim(-135.0, 135.0)
    axes[1, 0].set_xlabel("Time relative to pulse center (us)")
    axes[1, 0].set_ylabel("Dechirped voltage")
    axes[1, 0].set_title("Example offset +0.3 samples")
    axes[1, 0].legend(loc="upper right", frameon=False, fontsize=8)

    axes[1, 1].plot(freq_hz / 1e3, power_db, color="#2ca02c", lw=1.1)
    axes[1, 1].axvline(beat_hz[example_index] / 1e3, color="#d62728", lw=1.0, ls="--", label="FFT peak")
    axes[1, 1].axvline(expected_beat_hz[example_index] / 1e3, color="0.25", lw=1.0, ls=":", label="expected")
    axes[1, 1].set_ylim(-80.0, 3.0)
    axes[1, 1].set_xlabel("Dechirped beat frequency (kHz)")
    axes[1, 1].set_ylabel("Power (dB rel. peak)")
    axes[1, 1].set_title("Example beat spectrum")
    axes[1, 1].legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle("Zero-Doppler fractional-delay LFM beat-frequency simulation", fontsize=11)
    png = f"{OUTPUT_BASE}.png"
    pdf = f"{OUTPUT_BASE}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    for ext in ("png", "pdf"):
        shutil.copy2(f"{OUTPUT_BASE}.{ext}", f"{PAPER_FIGURE_BASE}.{ext}")

    print(f"output_h5={OUTPUT_BASE}.h5")
    print(f"output_png={png}")
    print(f"paper_pdf={PAPER_FIGURE_BASE}.pdf")
    print(f"max_abs_sample_error={np.nanmax(np.abs(sample_error)):.6e}")
    print(f"max_abs_path_error_m={np.nanmax(np.abs(path_error_m)):.6e}")
    for offset, rec, freq, err_m in zip(FRACTIONAL_OFFSETS_4MHZ, recovered_samples, beat_hz, path_error_m):
        print(f"offset={offset:+.3f} recovered={rec:+.6f} beat_hz={freq:+.3f} path_error_m={err_m:+.6e}")

    trajectory_png = f"{OUTPUT_BASE}_constant_radial_velocity.png"
    trajectory_pdf = f"{OUTPUT_BASE}_constant_radial_velocity.pdf"
    plot_trajectory_fit(trajectory, f"{OUTPUT_BASE}_constant_radial_velocity")
    for ext in ("png", "pdf"):
        shutil.copy2(f"{OUTPUT_BASE}_constant_radial_velocity.{ext}", f"{PAPER_TRAJECTORY_FIGURE_BASE}.{ext}")
    plot_trajectory_fit(noisy_trajectory, f"{OUTPUT_BASE}_constant_radial_velocity_snr18db", noisy=True)
    for ext in ("png", "pdf"):
        shutil.copy2(
            f"{OUTPUT_BASE}_constant_radial_velocity_snr18db.{ext}",
            f"{PAPER_NOISY_TRAJECTORY_FIGURE_BASE}.{ext}",
        )
    print(f"trajectory_paper_pdf={PAPER_TRAJECTORY_FIGURE_BASE}.pdf")
    print(f"true_radial_velocity_mps={trajectory['true_radial_velocity_mps']:.6f}")
    print(f"fit_radial_velocity_mps={trajectory['fit_radial_velocity_mps']:.6f}")
    print(f"radial_velocity_error_mps={trajectory['fit_radial_velocity_mps'] - trajectory['true_radial_velocity_mps']:+.6f}")
    print(f"true_doppler_hz={trajectory['true_doppler_hz']:.6f}")
    print(f"fit_doppler_hz={trajectory['fit_doppler_hz']:.6f}")
    print(f"path_rms_m={np.sqrt(np.mean(trajectory['path_residual_m']**2.0)):.6e}")
    print(f"beat_rms_hz={np.sqrt(np.mean(trajectory['beat_residual_hz']**2.0)):.6e}")
    print(f"noisy_trajectory_paper_pdf={PAPER_NOISY_TRAJECTORY_FIGURE_BASE}.pdf")
    print(f"noisy_snr_db={noisy_trajectory['noise_snr_db']:.3f}")
    print(f"noisy_fit_radial_velocity_mps={noisy_trajectory['fit_radial_velocity_mps']:.6f}")
    print(
        "noisy_radial_velocity_error_mps="
        f"{noisy_trajectory['fit_radial_velocity_mps'] - noisy_trajectory['true_radial_velocity_mps']:+.6f}"
    )
    print(f"noisy_fit_doppler_hz={noisy_trajectory['fit_doppler_hz']:.6f}")
    print(f"noisy_path_rms_m={np.sqrt(np.mean(noisy_trajectory['path_residual_m']**2.0)):.6e}")
    print(f"noisy_beat_rms_hz={np.sqrt(np.mean(noisy_trajectory['beat_residual_hz']**2.0)):.6e}")


if __name__ == "__main__":
    main()
