import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so

import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260618a"
DEFAULT_OUTPUT_BASE = os.path.join(
    "results", f"single_pulse_dechirped_fft_doppler_{SCRIPT_VERSION}"
)
DEFAULT_EVENT_GLOB = os.path.join("results", "tristatic_head_echoes", "*", "*.h5")
DEFAULT_ZERO_PAD_FACTOR = 64
DEFAULT_TIME_PAD_US = 50.0
DEFAULT_REFERENCE_CHIRP_RATE_SCALE = interp.REFERENCE_CHIRP_RATE_SCALE


def find_highest_snr_echo(pattern):
    best = None
    for path in glob.glob(pattern):
        try:
            with h5py.File(path, "r") as h:
                if "raw" not in h or "snr_peak_db" not in h:
                    continue
                snr_db = np.asarray(h["snr_peak_db"][()], dtype=np.float64)
                if snr_db.size == 0 or not np.any(np.isfinite(snr_db)):
                    continue
                idx = int(np.nanargmax(snr_db))
                candidate = (float(snr_db[idx]), path, idx)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        except OSError:
            continue
    if best is None:
        raise RuntimeError(f"No usable event files found with pattern {pattern!r}")
    return best[1], best[2]


def quadratic_peak_frequency(freq_hz, power_db, idx):
    if idx <= 0 or idx >= len(power_db) - 1:
        return float(freq_hz[idx]), 0.0
    ym1 = float(power_db[idx - 1])
    y0 = float(power_db[idx])
    yp1 = float(power_db[idx + 1])
    denom = ym1 - 2.0 * y0 + yp1
    if not np.isfinite(denom) or abs(denom) < 1e-30:
        return float(freq_hz[idx]), 0.0
    delta = 0.5 * (ym1 - yp1) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    df = float(freq_hz[1] - freq_hz[0])
    return float(freq_hz[idx] + delta * df), delta


def lfm_reference_for_offsets(
    sample_offsets,
    sr_mhz,
    bandwidth_hz,
    pulse_length_us,
    chirp_rate_scale=DEFAULT_REFERENCE_CHIRP_RATE_SCALE,
):
    t_s = np.asarray(sample_offsets, dtype=np.float64) / (float(sr_mhz) * 1e6)
    sweep_rate = (
        float(bandwidth_hz)
        * 1e6
        / float(pulse_length_us)
        / 2.0
        * float(chirp_rate_scale)
    )
    code = np.exp(
        1j
        * 2.0
        * np.pi
        * (t_s * float(bandwidth_hz) / 2.0 - sweep_rate * t_s**2.0)
    )
    return code.astype(np.complex128), t_s


def fft_power_metrics(deramped, sr_hz, zero_pad_factor):
    n_analysis = int(len(deramped))
    window = np.hanning(n_analysis)
    y = deramped * window

    n_fft = 1
    target = max(n_analysis, int(zero_pad_factor) * n_analysis)
    while n_fft < target:
        n_fft *= 2
    spectrum = np.fft.fftshift(np.fft.fft(y, n=n_fft))
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sr_hz))
    power = np.abs(spectrum) ** 2.0
    power_db = 10.0 * np.log10(np.maximum(power, 1e-300))
    peak_idx = int(np.nanargmax(power_db))
    peak_freq_hz, peak_delta_bins = quadratic_peak_frequency(freq_hz, power_db, peak_idx)
    power_db_rel = power_db - float(power_db[peak_idx])

    half_power = power_db_rel >= -3.0
    peak_region = np.flatnonzero(half_power)
    if peak_region.size:
        width_3db_hz = float(freq_hz[peak_region[-1]] - freq_hz[peak_region[0]])
    else:
        width_3db_hz = np.nan
    median_floor_db = float(np.nanmedian(power_db_rel))
    prominence_db = float(-median_floor_db)

    local = np.abs(freq_hz - peak_freq_hz) <= 80e3
    weights = np.maximum(10.0 ** (power_db_rel[local] / 10.0) - 10.0 ** (-45.0 / 10.0), 0.0)
    if np.sum(weights) > 0.0:
        local_freq = freq_hz[local]
        centroid_hz = float(np.sum(weights * local_freq) / np.sum(weights))
        rms_width_hz = float(
            np.sqrt(np.sum(weights * (local_freq - centroid_hz) ** 2.0) / np.sum(weights))
        )
    else:
        rms_width_hz = np.nan

    return {
        "n_fft": n_fft,
        "freq_hz": freq_hz,
        "power_db_rel": power_db_rel,
        "fft_bin_hz": float(sr_hz / n_fft),
        "peak_bin_frequency_hz": float(freq_hz[peak_idx]),
        "peak_frequency_hz": peak_freq_hz,
        "peak_delta_bins": peak_delta_bins,
        "width_3db_hz": width_3db_hz,
        "rms_width_hz": rms_width_hz,
        "prominence_db": prominence_db,
    }


def dechirped_fft_estimate(
    path,
    pulse_index,
    zero_pad_factor,
    time_pad_us,
    chirp_rate_scale=DEFAULT_REFERENCE_CHIRP_RATE_SCALE,
    optimize_chirp_rate=False,
):
    with h5py.File(path, "r") as h:
        row = np.asarray(h["raw"][pulse_index], dtype=np.complex128)
        gate = float(np.asarray(h["range_gate"][pulse_index]))
        snr_db = float(np.asarray(h["snr_peak_db"][pulse_index]))
        sr_mhz = float(np.asarray(h["sr_mhz"]))
        bw_mhz = float(np.asarray(h["bw_mhz"]))
        pulse_length_us = float(np.asarray(h["pulse_length_us"]))
        time_ns = int(np.asarray(h["times_ns"][pulse_index]))
        site = h["site"][()]
        event_id = h["event_id"][()]
        if isinstance(site, bytes):
            site = site.decode("utf-8")
        if isinstance(event_id, bytes):
            event_id = event_id.decode("utf-8")

    code, _ = interp.lfm(
        length_us=pulse_length_us,
        sr_mhz=sr_mhz,
        bandwidth_hz=bw_mhz * 1e6,
    )
    n_code = int(len(code))
    center = int(round(gate))
    pulse_start = center - n_code // 2
    pulse_stop = pulse_start + n_code
    pad_samples_requested = int(round(float(time_pad_us) * sr_mhz))
    start = max(0, pulse_start - pad_samples_requested)
    stop = min(len(row), pulse_stop + pad_samples_requested)
    if pulse_start < 0 or pulse_stop > len(row):
        raise ValueError(
            f"Pulse segment [{pulse_start}, {pulse_stop}) falls outside raw row length {len(row)}"
        )
    if start >= stop:
        raise ValueError(f"Empty padded segment [{start}, {stop})")

    segment = row[start:stop]
    sample_offsets = np.arange(start, stop, dtype=np.float64) - float(pulse_start)
    sr_hz = sr_mhz * 1e6

    def deramp_with_scale(scale):
        reference, t_s_local = lfm_reference_for_offsets(
            sample_offsets,
            sr_mhz=sr_mhz,
            bandwidth_hz=bw_mhz * 1e6,
            pulse_length_us=pulse_length_us,
            chirp_rate_scale=scale,
        )
        return segment * np.conj(reference), t_s_local

    optimized = {
        "chirp_rate_optimized": False,
        "chirp_rate_scale_initial": float(chirp_rate_scale),
    }
    if optimize_chirp_rate:
        def objective(scale_array):
            scale = float(np.atleast_1d(scale_array)[0])
            deramped_try, _ = deramp_with_scale(scale)
            metrics_try = fft_power_metrics(deramped_try, sr_hz, zero_pad_factor)
            return float(metrics_try["rms_width_hz"])

        opt = so.minimize_scalar(
            objective,
            bounds=(0.90, 1.10),
            method="bounded",
            options={"xatol": 1e-7, "maxiter": 120},
        )
        if opt.success and np.isfinite(opt.fun):
            chirp_rate_scale = float(opt.x)
            optimized.update(
                {
                    "chirp_rate_optimized": True,
                    "chirp_rate_optimizer_success": bool(opt.success),
                    "chirp_rate_optimizer_nfev": int(opt.nfev),
                    "chirp_rate_optimizer_score_hz": float(opt.fun),
                }
            )

    reference, t_s = lfm_reference_for_offsets(
        sample_offsets,
        sr_mhz=sr_mhz,
        bandwidth_hz=bw_mhz * 1e6,
        pulse_length_us=pulse_length_us,
        chirp_rate_scale=chirp_rate_scale,
    )
    deramped = segment * np.conj(reference)
    n_analysis = int(len(deramped))
    metrics = fft_power_metrics(deramped, sr_hz, zero_pad_factor)

    result = {
        "path": path,
        "site": site,
        "event_id": event_id,
        "pulse_index": int(pulse_index),
        "time_ns": time_ns,
        "range_gate": gate,
        "snr_peak_db": snr_db,
        "sr_mhz": sr_mhz,
        "bw_mhz": bw_mhz,
        "pulse_length_us": pulse_length_us,
        "n_code": n_code,
        "n_analysis": n_analysis,
        "n_fft": metrics["n_fft"],
        "zero_pad_factor_requested": int(zero_pad_factor),
        "time_pad_us_requested": float(time_pad_us),
        "reference_chirp_rate_scale": float(chirp_rate_scale),
        "reference_sweep_rate_hz_per_s": float(
            (bw_mhz * 1e6) * 1e6 / pulse_length_us * chirp_rate_scale
        ),
        "pad_samples_requested": pad_samples_requested,
        "pre_pulse_samples": int(pulse_start - start),
        "post_pulse_samples": int(stop - pulse_stop),
        "analysis_length_us": float(n_analysis / sr_mhz),
        "fft_bin_hz": metrics["fft_bin_hz"],
        "fourier_resolution_hz": float(1.0 / (n_analysis / sr_hz)),
        "peak_bin_frequency_hz": metrics["peak_bin_frequency_hz"],
        "peak_frequency_hz": metrics["peak_frequency_hz"],
        "peak_delta_bins": metrics["peak_delta_bins"],
        "width_3db_hz": metrics["width_3db_hz"],
        "rms_width_hz": metrics["rms_width_hz"],
        "prominence_db": metrics["prominence_db"],
        "t_us": t_s * 1e6,
        "deramped": deramped,
        "freq_hz": metrics["freq_hz"],
        "power_db_rel": metrics["power_db_rel"],
    }
    result.update(optimized)
    return result


def write_h5(result, output_base):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    with h5py.File(f"{output_base}.h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["method"] = (
            "Single raw pulse plus symmetric raw-voltage time padding; "
            "dechirp by the LFM code at the detected gate; "
            "Hann window; zero-padded FFT; quadratic interpolation of the "
            "log-power spectral peak."
        )
        for key, value in result.items():
            if key in {"path", "site", "event_id"}:
                h.attrs[key] = value
            elif np.isscalar(value):
                h.attrs[key] = value
            elif key == "deramped":
                h.create_dataset(key, data=np.asarray(value, dtype=np.complex64))
            else:
                h.create_dataset(key, data=np.asarray(value))


def plot_result(result, output_base):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)

    deramped = np.asarray(result["deramped"], dtype=np.complex128)
    norm = float(np.nanmax(np.abs(deramped)))
    if not np.isfinite(norm) or norm <= 0.0:
        norm = 1.0
    axes[0].plot(result["t_us"], np.real(deramped) / norm, color="#1f77b4", lw=1.0, label="Real")
    axes[0].plot(result["t_us"], np.imag(deramped) / norm, color="#d95f02", lw=1.0, label="Imag.")
    axes[0].axvline(0.0, color="0.45", lw=0.8, ls="--")
    axes[0].axvline(result["pulse_length_us"], color="0.45", lw=0.8, ls="--")
    axes[0].set_xlabel("Time relative to nominal pulse start (us)")
    axes[0].set_ylabel("Dechirped voltage\n(normalized)")
    axes[0].set_title(
        f"{result['site']} pulse {result['pulse_index']}, "
        f"SNR={result['snr_peak_db']:.1f} dB"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)

    freq_khz = result["freq_hz"] / 1e3
    peak_khz = result["peak_frequency_hz"] / 1e3
    keep = np.abs(freq_khz - peak_khz) <= 120.0
    axes[1].plot(freq_khz[keep], result["power_db_rel"][keep], color="#222222", lw=1.1)
    axes[1].axvline(peak_khz, color="#c43c39", lw=1.3)
    axes[1].set_ylim(-70, 3)
    axes[1].set_xlabel("Frequency offset after dechirp (kHz)")
    axes[1].set_ylabel("Relative power (dB)")
    axes[1].grid(True, alpha=0.25)
    axes[1].text(
        0.02,
        0.96,
        (
            f"peak = {peak_khz:.3f} kHz\n"
            f"FFT bin = {result['fft_bin_hz']:.1f} Hz\n"
            f"1/T = {result['fourier_resolution_hz'] / 1e3:.2f} kHz\n"
            f"RMS width = {result['rms_width_hz'] / 1e3:.2f} kHz\n"
            f"chirp scale = {result['reference_chirp_rate_scale']:.6f}\n"
            f"raw pad = {result['time_pad_us_requested']:.0f} us/side\n"
            f"zero pad = {result['n_fft'] / result['n_analysis']:.0f}x"
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    fig.suptitle("Single-pulse dechirped FFT Doppler-offset diagnostic", fontsize=12)
    fig.savefig(f"{output_base}.png", dpi=240)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a single-pulse Doppler offset from a dechirped, zero-padded FFT."
    )
    parser.add_argument("--event", default=None, help="Event HDF5 path. Default: highest-SNR echo.")
    parser.add_argument("--pulse-index", type=int, default=None, help="Pulse index. Default: max SNR in event.")
    parser.add_argument("--event-glob", default=DEFAULT_EVENT_GLOB)
    parser.add_argument("--zero-pad-factor", type=int, default=DEFAULT_ZERO_PAD_FACTOR)
    parser.add_argument(
        "--time-pad-us",
        type=float,
        default=DEFAULT_TIME_PAD_US,
        help="Raw-voltage samples to include before and after the nominal pulse.",
    )
    parser.add_argument(
        "--chirp-rate-scale",
        type=float,
        default=DEFAULT_REFERENCE_CHIRP_RATE_SCALE,
        help="Scale factor applied to the nominal LFM chirp rate.",
    )
    parser.add_argument(
        "--optimize-chirp-rate",
        action="store_true",
        help="Fit the reference chirp-rate scale by minimizing FFT spectral width.",
    )
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()

    if args.event is None:
        path, pulse_index = find_highest_snr_echo(args.event_glob)
    else:
        path = args.event
        if args.pulse_index is None:
            with h5py.File(path, "r") as h:
                pulse_index = int(np.nanargmax(np.asarray(h["snr_peak_db"][()], dtype=np.float64)))
        else:
            pulse_index = int(args.pulse_index)

    result = dechirped_fft_estimate(
        path,
        pulse_index,
        args.zero_pad_factor,
        args.time_pad_us,
        chirp_rate_scale=args.chirp_rate_scale,
        optimize_chirp_rate=args.optimize_chirp_rate,
    )
    write_h5(result, args.output_base)
    plot_result(result, args.output_base)

    print(f"event_file={path}")
    print(f"pulse_index={pulse_index}")
    print(f"site={result['site']}")
    print(f"event_id={result['event_id']}")
    print(f"snr_peak_db={result['snr_peak_db']:.2f}")
    print(f"peak_frequency_hz={result['peak_frequency_hz']:.3f}")
    print(f"rms_width_hz={result['rms_width_hz']:.3f}")
    print(f"width_3db_hz={result['width_3db_hz']:.3f}")
    print(f"reference_chirp_rate_scale={result['reference_chirp_rate_scale']:.9f}")
    print(f"chirp_rate_optimized={result['chirp_rate_optimized']}")
    print(f"fft_bin_hz={result['fft_bin_hz']:.3f}")
    print(f"fourier_resolution_hz={result['fourier_resolution_hz']:.3f}")
    print(f"time_pad_us_requested={result['time_pad_us_requested']:.3f}")
    print(f"analysis_length_us={result['analysis_length_us']:.3f}")
    print(f"output_h5={args.output_base}.h5")
    print(f"output_png={args.output_base}.png")


if __name__ == "__main__":
    main()
