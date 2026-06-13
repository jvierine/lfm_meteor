#!/usr/bin/env python3
"""Generate corrected Sanya monostatic range/height products."""

from __future__ import annotations

import argparse
import os

import h5py
import jcoord
import numpy as np

import sanya_opts as sc


SCRIPT_VERSION = "v20260613b"
DEFAULT_INPUT = os.path.join("results", "sanya_monostatic_ranges_v20260610.h5")
DEFAULT_OUTPUT = os.path.join("results", f"sanya_monostatic_ranges_{SCRIPT_VERSION}.h5")
SANYA_AZ_DEG = 15.0
SANYA_EL_DEG = 75.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT, help="Old Sanya monostatic cache with uncorrected ranges.")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Corrected output HDF5 path.")
    p.add_argument("--az-deg", type=float, default=SANYA_AZ_DEG)
    p.add_argument("--el-deg", type=float, default=SANYA_EL_DEG)
    return p.parse_args()


def slant_ranges_to_heights_km(ranges_km: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    heights = np.full(np.asarray(ranges_km).shape, np.nan, dtype=np.float64)
    for i, range_km in enumerate(np.asarray(ranges_km, dtype=np.float64)):
        if not np.isfinite(range_km):
            continue
        llh = jcoord.az_el_r2geodetic(
            sc.lat0[0],
            sc.lon0[0],
            sc.alt0[0] * 1e3,
            float(az_deg),
            float(el_deg),
            float(range_km) * 1e3,
        )
        heights[i] = float(llh[2] / 1e3)
    return heights


def copy_optional_dataset(src: h5py.File, dst: h5py.File, name: str) -> None:
    if name in src:
        dst.create_dataset(name, data=src[name][()])


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    with h5py.File(args.input, "r") as h:
        raw_range_km = np.asarray(h["range_km"][()], dtype=np.float64)
        corrected_range_km = raw_range_km + sc.SANYA_RANGE_CORRECTION_KM
        height_km = slant_ranges_to_heights_km(corrected_range_km, args.az_deg, args.el_deg)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with h5py.File(args.output, "w") as out:
            out.create_dataset("raw_range_km", data=raw_range_km)
            out.create_dataset("range_km", data=corrected_range_km)
            out.create_dataset("height_km", data=height_km)
            for name in ("event_id", "range_gate", "snr_peak_db", "source_file", "time_ns"):
                copy_optional_dataset(h, out, name)

            out.attrs["script"] = os.path.basename(__file__)
            out.attrs["script_version"] = SCRIPT_VERSION
            out.attrs["input_h5"] = args.input
            out.attrs["n_detections"] = int(corrected_range_km.size)
            out.attrs["sanya_range_correction_km"] = float(sc.SANYA_RANGE_CORRECTION_KM)
            out.attrs["range_correction_sign"] = "range_km = raw_range_km + sanya_range_correction_km"
            out.attrs["sanya_first_sample_delay_us"] = float(sc.SANYA_FIRST_SAMPLE_DELAY_US)
            out.attrs["sanya_corrected_txrx_delay_us"] = float(sc.SANYA_CORRECTED_TXRX_DELAY_US)
            out.attrs["sanya_zero_gate_txrx_path_km"] = float(sc.SANYA_CORRECTED_TXRX_DELAY_US * 1e-6 * 299792458.0 / 1e3)
            out.attrs["az_deg"] = float(args.az_deg)
            out.attrs["el_deg"] = float(args.el_deg)
            out.attrs["height_method"] = "jcoord.az_el_r2geodetic using corrected Sanya slant range along fixed beam"

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"detections: {corrected_range_km.size}")
    print(f"range correction: {sc.SANYA_RANGE_CORRECTION_KM:+.4f} km")
    print(f"corrected range min/max: {np.nanmin(corrected_range_km):.3f} / {np.nanmax(corrected_range_km):.3f} km")
    print(f"height min/max: {np.nanmin(height_km):.3f} / {np.nanmax(height_km):.3f} km")
    print(f"height mean/median: {np.nanmean(height_km):.3f} / {np.nanmedian(height_km):.3f} km")


if __name__ == "__main__":
    main()
