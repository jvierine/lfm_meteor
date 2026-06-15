#!/usr/bin/env python3
"""Reduce per-pulse system-noise power to robust 100-pulse averages.

Input and output are HDF5.  Each output row summarizes one contiguous 100-pulse
block within a source raw file.  Obvious outliers are rejected with a robust
median/MAD rule before averaging, and power summaries are stored as float32.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-h5", default="/mnt/data/juha/sanya/system_noise_power_4mhz/sanya_4mhz_system_noise_power_per_pulse.h5")
    p.add_argument("--output-h5", default="/mnt/data/juha/sanya/system_noise_power_4mhz/sanya_4mhz_system_noise_power_100pulse.h5")
    p.add_argument("--bin-size", type=int, default=100)
    p.add_argument("--mad-sigma", type=float, default=5.0)
    return p.parse_args()


def robust_block(values: np.ndarray, mad_sigma: float) -> tuple[float, float, float, int, int]:
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    n_total = int(vals.size)
    if finite.size == 0:
        return np.nan, np.nan, np.nan, 0, n_total
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if np.isfinite(mad) and mad > 0.0:
        sigma = 1.4826 * mad
        keep = np.abs(finite - med) <= mad_sigma * sigma
        used = finite[keep]
    else:
        used = finite
    if used.size == 0:
        used = finite
    return float(np.mean(used)), float(np.median(used)), float(np.std(used)), int(used.size), n_total


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_h5), exist_ok=True)
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive")

    with h5py.File(args.input_h5, "r") as src, h5py.File(args.output_h5, "w") as dst:
        dst.attrs["description"] = "Robust 100-pulse averages of Sanya 4 MHz raw-voltage system noise power."
        dst.attrs["input_h5"] = args.input_h5
        dst.attrs["bin_size_pulses"] = args.bin_size
        dst.attrs["outlier_rule"] = f"Within each bin reject values farther than {args.mad_sigma} * 1.4826 * MAD from the median."
        dst.attrs["power_definition"] = src.attrs.get("definition", "mean(real(data_raw)^2 + imag(data_raw)^2) over range samples")
        dst.attrs["units"] = src.attrs.get("units", "raw ADC voltage-squared units")
        dst.attrs["storage_note"] = "Power summary datasets are stored as float32 to keep the product compact."

        dst["site_names"] = src["site_names"][:]
        files = dst.create_group("files")
        for key in src["files"].keys():
            files.create_dataset(key, data=src["files"][key][:])

        n_pulses_by_file = src["files/n_pulses"][:].astype(np.int64)
        n_files = int(n_pulses_by_file.size)
        n_bins = int(np.sum((n_pulses_by_file + args.bin_size - 1) // args.bin_size))
        power = src["pulses/noise_power_mean_raw_voltage"]
        utc = src["pulses/time_utc_ns"]
        local = src["pulses/time_beijing_local_ns"]
        station = src["pulses/station_id"]
        pulse_index = src["pulses/pulse_index"]

        out = {
            "file_index": np.empty(n_bins, dtype=np.uint32),
            "station_id": np.empty(n_bins, dtype=np.uint8),
            "pulse_index_start": np.empty(n_bins, dtype=np.uint32),
            "pulse_index_end": np.empty(n_bins, dtype=np.uint32),
            "time_utc_start_ns": np.empty(n_bins, dtype=np.int64),
            "time_utc_end_ns": np.empty(n_bins, dtype=np.int64),
            "time_utc_mid_ns": np.empty(n_bins, dtype=np.int64),
            "time_beijing_local_start_ns": np.empty(n_bins, dtype=np.int64),
            "time_beijing_local_end_ns": np.empty(n_bins, dtype=np.int64),
            "noise_power_mean_raw_voltage": np.empty(n_bins, dtype=np.float32),
            "noise_power_median_raw_voltage": np.empty(n_bins, dtype=np.float32),
            "noise_power_std_raw_voltage": np.empty(n_bins, dtype=np.float32),
            "n_total": np.empty(n_bins, dtype=np.uint16),
            "n_used": np.empty(n_bins, dtype=np.uint16),
            "n_rejected": np.empty(n_bins, dtype=np.uint16),
        }

        pulse_cursor = 0
        bin_cursor = 0
        for fi in range(n_files):
            n_file_pulses = int(n_pulses_by_file[fi])
            file_start = pulse_cursor
            file_stop = file_start + n_file_pulses
            pulse_cursor = file_stop
            if n_file_pulses == 0:
                continue

            file_power = power[file_start:file_stop]
            file_utc = utc[file_start:file_stop]
            file_local = local[file_start:file_stop]
            file_station = station[file_start:file_stop]
            file_pulse_index = pulse_index[file_start:file_stop]

            for block_start in range(0, n_file_pulses, args.bin_size):
                block_stop = min(block_start + args.bin_size, n_file_pulses)
                mean, med, std, n_used, n_total = robust_block(file_power[block_start:block_stop], args.mad_sigma)
                t0 = int(file_utc[block_start])
                t1 = int(file_utc[block_stop - 1])
                out["file_index"][bin_cursor] = fi
                out["station_id"][bin_cursor] = int(file_station[block_start])
                out["pulse_index_start"][bin_cursor] = int(file_pulse_index[block_start])
                out["pulse_index_end"][bin_cursor] = int(file_pulse_index[block_stop - 1])
                out["time_utc_start_ns"][bin_cursor] = t0
                out["time_utc_end_ns"][bin_cursor] = t1
                out["time_utc_mid_ns"][bin_cursor] = t0 + (t1 - t0) // 2
                out["time_beijing_local_start_ns"][bin_cursor] = int(file_local[block_start])
                out["time_beijing_local_end_ns"][bin_cursor] = int(file_local[block_stop - 1])
                out["noise_power_mean_raw_voltage"][bin_cursor] = mean
                out["noise_power_median_raw_voltage"][bin_cursor] = med
                out["noise_power_std_raw_voltage"][bin_cursor] = std
                out["n_total"][bin_cursor] = n_total
                out["n_used"][bin_cursor] = n_used
                out["n_rejected"][bin_cursor] = n_total - n_used
                bin_cursor += 1

        if bin_cursor != n_bins:
            raise RuntimeError(f"internal bin count mismatch: wrote {bin_cursor}, expected {n_bins}")

        bins = dst.create_group("bins")
        chunk = min(max(1, n_bins), 8192)
        for key, values in out.items():
            bins.create_dataset(key, data=values, chunks=(chunk,))

        errors = dst.create_group("errors")
        for key in src["errors"].keys():
            errors.create_dataset(key, data=src["errors"][key][:])
        print(f"wrote {args.output_h5}")
        print(f"bins {n_bins}")


if __name__ == "__main__":
    main()
