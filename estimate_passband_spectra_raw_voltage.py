#!/usr/bin/env python3
"""Estimate Hann-windowed raw-voltage passband spectra for quiet and RFI periods.

This script is intended to run on revontuli, close to the raw voltage files.
It uses the reduced 100-pulse system-noise product to select, for each station,
one quiet interval and one high-noise/RFI interval.  It then uses the
pulse-level product as an index into the raw MATLAB v7.3 HDF5 files and
averages Hann-windowed FFT power spectra over raw complex voltages.

The output is HDF5 only.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


SITE_ORDER = ("Sanya", "Danzhou", "Wenchang")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-pulse-h5",
        default="/mnt/data/juha/sanya/system_noise_power_4mhz/sanya_4mhz_system_noise_power_per_pulse.h5",
        help="Pulse-level raw noise-power HDF5 product.",
    )
    p.add_argument(
        "--low-rate-h5",
        default="/mnt/data/juha/sanya/system_noise_power_4mhz/sanya_4mhz_system_noise_power_100pulse.h5",
        help="Reduced 100-pulse raw noise-power HDF5 product.",
    )
    p.add_argument(
        "--output-h5",
        default="/mnt/data/juha/sanya/system_noise_power_4mhz/sanya_passband_spectra_quiet_rfi.h5",
        help="Output HDF5 file.",
    )
    p.add_argument("--sample-rate-hz", type=float, default=4.0e6)
    p.add_argument(
        "--window-seconds",
        type=float,
        default=20.0,
        help="Raw-voltage interval length for each selected spectrum.",
    )
    p.add_argument(
        "--fft-block-pulses",
        type=int,
        default=256,
        help="Number of pulses to read and FFT at a time.",
    )
    return p.parse_args()


def ns_to_iso(ns: int) -> str:
    return str(np.datetime64(int(ns), "ns")).replace("T", " ")


def decode_strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(np.float64), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def select_periods(low_rate_h5: str, window_seconds: float) -> dict[tuple[str, str], dict]:
    selections: dict[tuple[str, str], dict] = {}
    with h5py.File(low_rate_h5, "r") as h:
        names = decode_strings(h["site_names"][:])
        station_id = h["bins/station_id"][:]
        time_ns = h["bins/time_utc_mid_ns"][:].astype(np.int64)
        power = h["bins/noise_power_mean_raw_voltage"][:].astype(np.float64)

    for sid, site in enumerate(names):
        m = (station_id == sid) & np.isfinite(power) & (power > 0.0)
        if not np.any(m):
            raise RuntimeError(f"no finite low-rate power samples for {site}")
        t = time_ns[m]
        pwr = power[m]
        order = np.argsort(t)
        t = t[order]
        pwr = pwr[order]
        dt = np.nanmedian(np.diff(t).astype(np.float64)) / 1.0e9
        bins_per_window = max(1, int(round(window_seconds / dt))) if np.isfinite(dt) and dt > 0 else 1
        smooth = rolling_mean(np.log(pwr), bins_per_window)

        quiet_idx = int(np.nanargmin(smooth))
        rfi_idx = int(np.nanargmax(smooth))
        for label, idx in (("quiet", quiet_idx), ("rfi", rfi_idx)):
            center = int(t[idx])
            half_ns = int(round(0.5 * window_seconds * 1.0e9))
            lo = center - half_ns
            hi = center + half_ns
            in_window = (t >= lo) & (t <= hi)
            selections[(site, label)] = {
                "site": site,
                "period": label,
                "center_utc_ns": center,
                "start_utc_ns": lo,
                "end_utc_ns": hi,
                "low_rate_window_mean_power": float(np.nanmean(pwr[in_window])),
                "low_rate_window_median_power": float(np.nanmedian(pwr[in_window])),
                "low_rate_window_min_power": float(np.nanmin(pwr[in_window])),
                "low_rate_window_max_power": float(np.nanmax(pwr[in_window])),
                "low_rate_bins": int(np.count_nonzero(in_window)),
            }
    return selections


def read_complex_block(raw: h5py.Dataset, pulse_indices: np.ndarray) -> np.ndarray:
    block = raw[:, pulse_indices]
    real = block["real"].astype(np.float64, copy=False)
    imag = block["imag"].astype(np.float64, copy=False)
    voltage = real + 1j * imag
    voltage -= np.mean(voltage, axis=0, keepdims=True)
    return voltage


def estimate_spectrum_for_selection(
    per_pulse: h5py.File,
    selection: dict,
    sample_rate_hz: float,
    fft_block_pulses: int,
) -> tuple[np.ndarray, dict]:
    files = per_pulse["files/source_file"]
    file_index_all = per_pulse["pulses/file_index"][:]
    station_all = per_pulse["pulses/station_id"][:]
    pulse_index_all = per_pulse["pulses/pulse_index"][:]
    time_all = per_pulse["pulses/time_utc_ns"][:].astype(np.int64)

    site = selection["site"]
    sid = SITE_ORDER.index(site)
    m = (
        (station_all == sid)
        & (time_all >= int(selection["start_utc_ns"]))
        & (time_all <= int(selection["end_utc_ns"]))
    )
    indices = np.flatnonzero(m)
    if indices.size == 0:
        raise RuntimeError(f"no raw pulses found for {site} {selection['period']} at {ns_to_iso(selection['center_utc_ns'])}")

    by_file: dict[int, list[int]] = defaultdict(list)
    by_file_time: dict[int, list[int]] = defaultdict(list)
    for global_idx in indices:
        fi = int(file_index_all[global_idx])
        by_file[fi].append(int(pulse_index_all[global_idx]))
        by_file_time[fi].append(int(time_all[global_idx]))

    sum_power: np.ndarray | None = None
    n_accum = 0
    n_range = None
    source_files: list[str] = []
    actual_start = np.iinfo(np.int64).max
    actual_end = np.iinfo(np.int64).min

    for fi in sorted(by_file):
        source_file = files[fi]
        source_path = source_file.decode("utf-8") if isinstance(source_file, bytes) else str(source_file)
        source_files.append(source_path)
        pulse_indices = np.asarray(by_file[fi], dtype=np.int64)
        times = np.asarray(by_file_time[fi], dtype=np.int64)
        order = np.argsort(pulse_indices)
        pulse_indices = pulse_indices[order]
        times = times[order]
        actual_start = min(actual_start, int(times[0]))
        actual_end = max(actual_end, int(times[-1]))

        with h5py.File(source_path, "r") as raw_h:
            raw = raw_h["data_raw"]
            if n_range is None:
                n_range = int(raw.shape[0])
                window = np.hanning(n_range).astype(np.float64)
                window_norm = float(np.sum(window * window))
                sum_power = np.zeros(n_range, dtype=np.float64)
            elif int(raw.shape[0]) != n_range:
                raise RuntimeError(f"range-sample mismatch in {source_path}")

            for start in range(0, pulse_indices.size, fft_block_pulses):
                stop = min(start + fft_block_pulses, pulse_indices.size)
                block_indices = pulse_indices[start:stop]
                x = read_complex_block(raw, block_indices)
                spec = np.fft.fftshift(np.fft.fft(x * window[:, None], axis=0), axes=0)
                sum_power += np.sum(np.abs(spec) ** 2, axis=1) / window_norm
                n_accum += x.shape[1]

    if sum_power is None or n_range is None or n_accum == 0:
        raise RuntimeError(f"no samples accumulated for {site} {selection['period']}")

    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_range, d=1.0 / sample_rate_hz))
    spectrum = sum_power / float(n_accum)
    info = dict(selection)
    info.update(
        {
            "n_raw_pulses": int(n_accum),
            "n_range_samples": int(n_range),
            "actual_start_utc_ns": int(actual_start),
            "actual_end_utc_ns": int(actual_end),
            "source_files": source_files,
        }
    )
    return spectrum.astype(np.float32), info | {"frequency_hz": freq_hz}


def write_output(output_h5: str, spectra: dict[tuple[str, str], tuple[np.ndarray, dict]], args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    first_info = next(iter(spectra.values()))[1]
    frequency_hz = first_info["frequency_hz"]
    with h5py.File(output_h5, "w") as h:
        h.attrs["description"] = "DC-removed, Hann-windowed raw-voltage passband spectra for quiet and high-noise/RFI intervals."
        h.attrs["per_pulse_h5"] = args.per_pulse_h5
        h.attrs["low_rate_h5"] = args.low_rate_h5
        h.attrs["sample_rate_hz"] = args.sample_rate_hz
        h.attrs["window_seconds"] = args.window_seconds
        h.attrs["fft_window"] = "Hann over raw range samples"
        h.attrs["dc_removal"] = "For each pulse, subtract mean complex raw voltage over range samples before applying the Hann window."
        h.attrs["power_definition"] = "mean over selected pulses of abs(fftshift(fft(Hann * (complex raw voltage - per-pulse range mean))))^2 / sum(Hann^2)"
        h["site_names"] = np.asarray(SITE_ORDER, dtype=string_dtype)
        h["period_names"] = np.asarray(("quiet", "rfi"), dtype=string_dtype)
        h["frequency_hz"] = frequency_hz.astype(np.float64)
        spec_group = h.create_group("spectra")
        sel_group = h.create_group("selections")

        row_site = []
        row_period = []
        row_center = []
        row_start = []
        row_end = []
        row_actual_start = []
        row_actual_end = []
        row_pulses = []
        row_low_median = []
        row_low_mean = []

        for (site, period), (spectrum, info) in spectra.items():
            g = spec_group.require_group(site).create_group(period)
            g["power_spectrum"] = spectrum
            g["source_files"] = np.asarray(info["source_files"], dtype=string_dtype)
            for key in (
                "center_utc_ns",
                "start_utc_ns",
                "end_utc_ns",
                "actual_start_utc_ns",
                "actual_end_utc_ns",
                "n_raw_pulses",
                "n_range_samples",
                "low_rate_bins",
                "low_rate_window_mean_power",
                "low_rate_window_median_power",
                "low_rate_window_min_power",
                "low_rate_window_max_power",
            ):
                g.attrs[key] = info[key]

            row_site.append(site)
            row_period.append(period)
            row_center.append(info["center_utc_ns"])
            row_start.append(info["start_utc_ns"])
            row_end.append(info["end_utc_ns"])
            row_actual_start.append(info["actual_start_utc_ns"])
            row_actual_end.append(info["actual_end_utc_ns"])
            row_pulses.append(info["n_raw_pulses"])
            row_low_median.append(info["low_rate_window_median_power"])
            row_low_mean.append(info["low_rate_window_mean_power"])

        sel_group["site"] = np.asarray(row_site, dtype=string_dtype)
        sel_group["period"] = np.asarray(row_period, dtype=string_dtype)
        sel_group["center_utc_ns"] = np.asarray(row_center, dtype=np.int64)
        sel_group["start_utc_ns"] = np.asarray(row_start, dtype=np.int64)
        sel_group["end_utc_ns"] = np.asarray(row_end, dtype=np.int64)
        sel_group["actual_start_utc_ns"] = np.asarray(row_actual_start, dtype=np.int64)
        sel_group["actual_end_utc_ns"] = np.asarray(row_actual_end, dtype=np.int64)
        sel_group["n_raw_pulses"] = np.asarray(row_pulses, dtype=np.int64)
        sel_group["low_rate_window_median_power"] = np.asarray(row_low_median, dtype=np.float64)
        sel_group["low_rate_window_mean_power"] = np.asarray(row_low_mean, dtype=np.float64)


def main() -> None:
    args = parse_args()
    selections = select_periods(args.low_rate_h5, args.window_seconds)
    spectra: dict[tuple[str, str], tuple[np.ndarray, dict]] = {}
    with h5py.File(args.per_pulse_h5, "r") as per_pulse:
        for site in SITE_ORDER:
            for period in ("quiet", "rfi"):
                spectrum, info = estimate_spectrum_for_selection(
                    per_pulse,
                    selections[(site, period)],
                    args.sample_rate_hz,
                    args.fft_block_pulses,
                )
                spectra[(site, period)] = (spectrum, info)
                print(
                    f"{site:8s} {period:5s} {info['n_raw_pulses']:5d} pulses "
                    f"{ns_to_iso(info['actual_start_utc_ns'])} to {ns_to_iso(info['actual_end_utc_ns'])}",
                    flush=True,
                )
    write_output(args.output_h5, spectra, args)
    print(f"wrote {args.output_h5}")


if __name__ == "__main__":
    main()
