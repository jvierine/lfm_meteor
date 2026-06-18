import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260618a"
DEFAULT_OUTPUT_BASE = os.path.join(
    "results", f"single_pulse_dechirped_fft_doppler_{SCRIPT_VERSION}"
)
DEFAULT_EVENT_GLOB = os.path.join("results", "tristatic_head_echoes", "*", "*.h5")
DEFAULT_ZERO_PAD_FACTOR = 64


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


def dechirped_fft_estimate(path, pulse_index, zero_pad_factor):
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

    code, t_s = interp.lfm(
        length_us=pulse_length_us,
        sr_mhz=sr_mhz,
        bandwidth_hz=bw_mhz * 1e6,
    )
    n_code = int(len(code))
    center = int(round(gate))
    start = center - n_code // 2
    stop = start + n_code
    if start < 0 or stop > len(row):
        raise ValueError(
            f"Pulse segment [{start}, {stop}) falls outside raw row length {len(row)}"
        )

    segment = row[start:stop]
    deramped = segment * np.conj(code.astype(np.complex128))
    window = np.hanning(n_code)
    y = deramped * window

    n_fft = 1
    target = max(n_code, int(zero_pad_factor) * n_code)
    while n_fft < target:
        n_fft *= 2
    sr_hz = sr_mhz * 1e6
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

    return {
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
        "n_fft": n_fft,
        "zero_pad_factor_requested": int(zero_pad_factor),
        "fft_bin_hz": float(sr_hz / n_fft),
        "fourier_resolution_hz": float(1.0 / (pulse_length_us * 1e-6)),
        "peak_bin_frequency_hz": float(freq_hz[peak_idx]),
        "peak_frequency_hz": peak_freq_hz,
        "peak_delta_bins": peak_delta_bins,
        "width_3db_hz": width_3db_hz,
        "prominence_db": prominence_db,
        "t_us": t_s * 1e6,
        "deramped": deramped,
        "freq_hz": freq_hz,
        "power_db_rel": power_db_rel,
    }


def write_h5(result, output_base):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    with h5py.File(f"{output_base}.h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["method"] = (
            "Single raw pulse; dechirp by the LFM code at the detected gate; "
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

    amp = np.abs(result["deramped"])
    axes[0].plot(result["t_us"], amp / np.nanmax(amp), color="#3b6ea8", lw=1.2)
    axes[0].set_xlabel("Time in pulse (us)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].set_title(
        f"{result['site']} pulse {result['pulse_index']}, "
        f"SNR={result['snr_peak_db']:.1f} dB"
    )
    axes[0].grid(True, alpha=0.25)

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
            f"zero pad = {result['n_fft'] / result['n_code']:.0f}x"
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

    result = dechirped_fft_estimate(path, pulse_index, args.zero_pad_factor)
    write_h5(result, args.output_base)
    plot_result(result, args.output_base)

    print(f"event_file={path}")
    print(f"pulse_index={pulse_index}")
    print(f"site={result['site']}")
    print(f"event_id={result['event_id']}")
    print(f"snr_peak_db={result['snr_peak_db']:.2f}")
    print(f"peak_frequency_hz={result['peak_frequency_hz']:.3f}")
    print(f"fft_bin_hz={result['fft_bin_hz']:.3f}")
    print(f"fourier_resolution_hz={result['fourier_resolution_hz']:.3f}")
    print(f"output_h5={args.output_base}.h5")
    print(f"output_png={args.output_base}.png")


if __name__ == "__main__":
    main()
