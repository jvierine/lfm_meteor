#!/usr/bin/env python3
"""Write compact HDF5 raw-voltage cuts for selected Sanya head-echo events.

The raw MATLAB files are available on the processing server.  This script reads
the full head-echo event index, applies the same event-selection gates used for
the all-detections time-delay memo, force-includes any events listed in the
tri-static index, and writes one cut file per event.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import h5py
import jcoord
import numpy as np

import sanya_opts as sc


C_MPS = 299792458.0
UTC8_NS = int(8 * 3600 * 1e9)
SOURCE_TIMEZONE_NAME = "Beijing local time"
SOURCE_TIMEZONE_OFFSET_HOURS = 8
SOURCE_TIMEZONE_OFFSET = np.timedelta64(SOURCE_TIMEZONE_OFFSET_HOURS, "h")
EPOCH = np.datetime64("1970-01-01T00:00:00", "ns")

SITE_ORDER = ("sanya", "danzhou", "wenchang")
SITE_DIRS = {"sanya": "Sanya", "danzhou": "Danzhou", "wenchang": "Wenchang"}
SITE_FIRST_SAMPLE_DELAY_US = {"sanya": 466.32, "danzhou": 438.426, "wenchang": 430.906}
SANYA_RANGE_CORRECTION_KM = -16.0186
SANYA_AZ_DEG = 15.0
SANYA_EL_DEG = 75.0
SANYA_LOW_HEIGHT_KM = 80.0
SANYA_HIGH_HEIGHT_KM = 120.0
SANYA_OUTSIDE_HEIGHT_MIN_ABS_VELOCITY_KM_S = 10.0
BISTATIC_MIN_DELAY_US = 800.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/mnt/data/juha/SANYA/Juha/20240422")
    p.add_argument("--head-echo-root", default="results/head_echoes")
    p.add_argument("--tristatic-index", default="results/tristatic_event_index.h5")
    p.add_argument("--output-root", default="/mnt/data/juha/sanya/replication_data/raw_voltage_cuts")
    p.add_argument("--manifest", default="/mnt/data/juha/sanya/replication_data/raw_voltage_cut_index.h5")
    p.add_argument("--manifest-csv", default="/mnt/data/juha/sanya/replication_data/raw_voltage_cut_index.csv")
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--monostatic-max-rms-m", type=float, default=100.0)
    p.add_argument("--bistatic-min-max-snr-db", type=float, default=15.0)
    p.add_argument("--bistatic-max-rms-m", type=float, default=100.0)
    p.add_argument("--range-padding-km", type=float, default=5.0)
    p.add_argument("--time-padding-s", type=float, default=0.05)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--site", choices=("all", "sanya", "danzhou", "wenchang"), default="all")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def delay_us_to_range_km(delay_us: float | np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(delay_us, dtype=np.float64) * 1e-6 * C_MPS / 1e3


def site_first_sample_r0_km(site: str) -> float:
    return float(delay_us_to_range_km(SITE_FIRST_SAMPLE_DELAY_US[site]))


def lfm_sample_count(length_us: float, sr_mhz: float) -> int:
    return int(round(float(length_us) * float(sr_mhz)))


def dt_from_time_array(tm: np.ndarray, i: int) -> np.datetime64:
    base_dt = np.datetime64(
        f"{int(tm[0, i] + 2000):04d}-{int(tm[1, i]):02d}-{int(tm[2, i]):02d}T"
        f"{int(tm[3, i]):02d}:{int(tm[4, i]):02d}"
    )
    whole_sec = int(np.floor(tm[5, i]))
    frac_ns = int(np.round(1e9 * (float(tm[5, i]) - whole_sec)))
    return base_dt + np.timedelta64(whole_sec, "s") + np.timedelta64(frac_ns, "ns")


def matlab_time_ns(tm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = np.empty(tm.shape[1], dtype=np.uint64)
    utc = np.empty(tm.shape[1], dtype=np.uint64)
    for i in range(tm.shape[1]):
        dt_local = dt_from_time_array(tm, i)
        local[i] = np.uint64((dt_local - EPOCH).astype("timedelta64[ns]").astype(np.int64))
        utc[i] = np.uint64(((dt_local - SOURCE_TIMEZONE_OFFSET) - EPOCH).astype("timedelta64[ns]").astype(np.int64))
    return local, utc


def read_index(root: Path) -> list[dict]:
    rows = []
    with h5py.File(root / "head_echo_index.h5", "r") as h:
        for idx in range(len(h["event_id"])):
            rows.append(
                {
                    "event_id": decode(h["event_id"][idx]),
                    "site": decode(h["site"][idx]).lower(),
                    "event_h5": decode(h["event_h5"][idx]),
                    "source_file": decode(h["source_file"][idx]) if "source_file" in h else "",
                }
            )
    return rows


def resolve_event_path(root: Path, event_h5: str, site: str) -> Path:
    path = Path(event_h5)
    if path.exists():
        return path
    candidate = root / site / path.name
    if candidate.exists():
        return candidate
    candidate = root.parent / event_h5
    if candidate.exists():
        return candidate
    raise FileNotFoundError(event_h5)


def read_event(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        return {
            "times_local_ns": np.asarray(h["times_ns"][:], dtype=np.uint64),
            "range_gate": np.asarray(h["range_gate"][:], dtype=np.int64),
            "range_km": np.asarray(h["range_km"][:], dtype=np.float64),
            "snr_peak_db": np.asarray(h["snr_peak_db"][:], dtype=np.float64),
            "r0_km": float(h["r0"][()]),
            "r1_km": float(h["r1"][()]),
            "az_deg": float(h["az"][()]),
            "el_deg": float(h["el"][()]),
            "sr_mhz": float(h["sr_mhz"][()]),
            "bw_mhz": float(h["bw_mhz"][()]),
            "ipp_us": float(h["ipp_us"][()]),
            "pulse_length_us": float(h["pulse_length_us"][()]),
            "source_file": decode(h["source_file"][()]) if "source_file" in h else "",
        }


def fit_track(time_ns: np.ndarray, range_km: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    t_s = (time_ns.astype(np.float64) - float(time_ns[0])) / 1e9
    t_fit = t_s - float(np.mean(t_s))
    degree = max(1, min(int(degree), len(t_s) - 1))
    coeff = np.polyfit(t_fit, range_km.astype(np.float64), degree)
    fitted = np.polyval(coeff, t_fit)
    rate_km_s = np.polyval(np.polyder(coeff), t_fit)
    rms_m = float(np.sqrt(np.mean((range_km - fitted) ** 2.0)) * 1e3)
    return rate_km_s, rms_m


def sanya_slant_ranges_to_heights_km(ranges_km: np.ndarray) -> np.ndarray:
    heights = np.full(np.asarray(ranges_km).shape, np.nan, dtype=np.float64)
    for idx, range_km in enumerate(np.asarray(ranges_km, dtype=np.float64)):
        if not np.isfinite(range_km):
            continue
        llh = jcoord.az_el_r2geodetic(
            sc.lat0[0], sc.lon0[0], sc.alt0[0] * 1e3, SANYA_AZ_DEG, SANYA_EL_DEG, float(range_km) * 1e3
        )
        heights[idx] = float(llh[2] / 1e3)
    return heights


def selection_for_event(site: str, event: dict, args: argparse.Namespace) -> dict:
    local_ns = np.asarray(event["times_local_ns"], dtype=np.uint64)
    utc_ns = local_ns.astype(np.int64) - UTC8_NS
    raw_range_km = np.asarray(event["range_km"], dtype=np.float64)
    range_km = raw_range_km + SANYA_RANGE_CORRECTION_KM if site == "sanya" else raw_range_km.copy()
    snr_db = np.asarray(event["snr_peak_db"], dtype=np.float64)
    finite = np.isfinite(range_km) & np.isfinite(snr_db)
    if np.count_nonzero(finite) < args.min_points:
        return {"keep_event": False, "point_keep": np.zeros(len(range_km), dtype=bool), "reason": "too_few_finite_points"}

    order = np.argsort(utc_ns[finite])
    src_idx = np.flatnonzero(finite)[order]
    rate_km_s, rms_m = fit_track(utc_ns[src_idx], range_km[src_idx], args.poly_degree)
    point_keep = np.zeros(len(range_km), dtype=bool)
    max_snr_db = float(np.nanmax(snr_db[src_idx]))
    delay_us = 2.0 * range_km[src_idx] * 1e3 / C_MPS * 1e6
    height_km = np.full(len(range_km), np.nan, dtype=np.float64)

    if site == "sanya":
        keep_event = rms_m <= args.monostatic_max_rms_m
        heights = sanya_slant_ranges_to_heights_km(range_km[src_idx])
        height_km[src_idx] = heights
        outside = (heights < SANYA_LOW_HEIGHT_KM) | (heights > SANYA_HIGH_HEIGHT_KM)
        point_keep[src_idx] = np.isfinite(heights) & (~outside | (np.abs(rate_km_s) > SANYA_OUTSIDE_HEIGHT_MIN_ABS_VELOCITY_KM_S))
        reason = "sanya_rms_height_velocity"
    else:
        keep_event = max_snr_db >= args.bistatic_min_max_snr_db and rms_m <= args.bistatic_max_rms_m
        point_keep[src_idx] = delay_us > BISTATIC_MIN_DELAY_US
        reason = "remote_snr_rms_delay"

    return {
        "keep_event": bool(keep_event and np.any(point_keep)),
        "point_keep": point_keep,
        "rate_km_s_ordered": rate_km_s,
        "ordered_indices": src_idx,
        "range_poly_fit_rms_m": rms_m,
        "max_snr_db": max_snr_db,
        "height_km": height_km,
        "corrected_range_km": range_km,
        "reason": reason,
    }


def tri_static_force_set(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    event_ids: set[str] = set()
    with h5py.File(path, "r") as h:
        for key in ("sanya_event_id", "danzhou_event_id", "wenchang_event_id"):
            if key in h:
                event_ids.update(decode(value) for value in h[key][:])
        for key in ("sanya_event_h5", "danzhou_event_h5", "wenchang_event_h5"):
            if key in h:
                for value in h[key][:]:
                    name = Path(decode(value)).stem
                    if name:
                        event_ids.add(name)
    return event_ids


def collect_selected_events(args: argparse.Namespace) -> list[dict]:
    root = Path(args.head_echo_root)
    tri_force = tri_static_force_set(args.tristatic_index)
    selected = []
    for row in read_index(root):
        site = row["site"]
        if site not in SITE_ORDER:
            continue
        if args.site != "all" and site != args.site:
            continue
        path = resolve_event_path(root, row["event_h5"], site)
        event = read_event(path)
        selection = selection_for_event(site, event, args)
        forced = row["event_id"] in tri_force or Path(row["event_h5"]).stem in tri_force
        if not selection["keep_event"] and not forced:
            continue
        if forced and not np.any(selection["point_keep"]):
            selection["point_keep"] = np.isfinite(event["range_km"])
        selected.append({**row, "event_path": str(path), "event": event, "selection": selection, "force_tristatic": forced})
        if args.max_events > 0 and len(selected) >= args.max_events:
            break
    return selected


def load_mat_file(path: str, site: str) -> dict:
    with h5py.File(path, "r") as h:
        p = h["para"][:]
        tm = h["time"][:]
        n_range_gates = int(h["data_raw"].shape[0])

    pulse_length_us = float(p[10, 0])
    sr_mhz = float(p[14, 0])
    bw_mhz = float(p[15, 0])
    local_ns, utc_ns = matlab_time_ns(tm)
    dr_km = C_MPS / (sr_mhz * 1e6) / 2.0 / 1e3
    raw_r0_km = float(p[12, 0])
    ranges_km = raw_r0_km + dr_km * np.arange(n_range_gates, dtype=np.float64)
    return {
        "source_file": path,
        "range_km_axis": ranges_km,
        "local_time_ns": local_ns,
        "utc_time_ns": utc_ns,
        "para": p.astype(np.float32),
        "time": tm.astype(np.float32),
        "az_deg": float(p[6, 0]),
        "el_deg": float(p[7, 0]),
        "pulse_length_us": pulse_length_us,
        "ipp_us": float(p[11, 0]),
        "raw_r0_km": raw_r0_km,
        "r1_km": float(p[13, 0]),
        "sr_mhz": sr_mhz,
        "bw_mhz": bw_mhz,
    }


def nearest_indices(sorted_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_values, target_values)
    out = np.empty(len(target_values), dtype=np.int64)
    for ii, pos in enumerate(idx):
        candidates = []
        if pos < len(sorted_values):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            out[ii] = -1
        else:
            target = int(target_values[ii])
            out[ii] = min(candidates, key=lambda jj: abs(int(sorted_values[jj]) - target))
    return out


def output_path(output_root: str, site: str, event_id: str) -> str:
    return os.path.join(output_root, site, f"{event_id}.h5")


def write_cut(row: dict, mat: dict, raw_dataset, args: argparse.Namespace) -> dict:
    event = row["event"]
    selection = row["selection"]
    site = row["site"]
    event_id = row["event_id"]
    point_keep = np.asarray(selection["point_keep"], dtype=bool)
    keep_idx = np.flatnonzero(point_keep)
    if keep_idx.size == 0:
        keep_idx = np.arange(len(event["range_km"]), dtype=int)

    event_local_ns = np.asarray(event["times_local_ns"], dtype=np.uint64)
    detection_pulses = nearest_indices(mat["local_time_ns"], event_local_ns)
    if np.any(detection_pulses < 0):
        raise RuntimeError(f"Could not match event pulses for {event_id}")

    kept_detection_pulses = detection_pulses[keep_idx]
    t0 = int(np.min(event_local_ns[keep_idx])) - int(round(args.time_padding_s * 1e9))
    t1 = int(np.max(event_local_ns[keep_idx])) + int(round(args.time_padding_s * 1e9))
    p0 = max(0, int(np.searchsorted(mat["local_time_ns"], np.uint64(max(0, t0)), side="left")))
    p1 = min(len(mat["local_time_ns"]), int(np.searchsorted(mat["local_time_ns"], np.uint64(max(0, t1)), side="right")))
    if p1 <= p0:
        p0 = int(np.min(kept_detection_pulses))
        p1 = min(len(mat["local_time_ns"]), p0 + 1)

    raw_range = np.asarray(event["range_km"], dtype=np.float64)
    science_r0 = float(np.nanmin(raw_range[keep_idx])) - args.range_padding_km
    science_r1 = float(np.nanmax(raw_range[keep_idx])) + args.range_padding_km
    science_g0 = max(0, int(np.searchsorted(mat["range_km_axis"], science_r0, side="left")))
    science_g1 = min(len(mat["range_km_axis"]), int(np.searchsorted(mat["range_km_axis"], science_r1, side="right")))
    if science_g1 <= science_g0:
        science_g0 = int(np.nanmin(event["range_gate"][keep_idx]))
        science_g1 = min(len(mat["range_km_axis"]), science_g0 + 1)
    lfm_pulse_samples = lfm_sample_count(event["pulse_length_us"], event["sr_mhz"])
    filter_pre = lfm_pulse_samples // 2
    filter_post = lfm_pulse_samples - 1 - filter_pre
    g0 = max(0, science_g0 - filter_pre)
    g1 = min(len(mat["range_km_axis"]), science_g1 + filter_post)

    pulse_slice = slice(p0, p1)
    gate_slice = slice(g0, g1)
    local_cut = mat["local_time_ns"][pulse_slice]
    utc_cut = mat["utc_time_ns"][pulse_slice]
    rel_detection_pulses = detection_pulses - p0
    rel_detection_gates = np.asarray(event["range_gate"], dtype=np.int64) - g0
    selected_mask_in_cut = np.zeros(len(event["range_gate"]), dtype=bool)
    selected_mask_in_cut[keep_idx] = True

    path = output_path(args.output_root, site, event_id)
    if os.path.exists(path) and not args.overwrite:
        return {"event_id": event_id, "site": site, "cut_h5": path, "status": "exists"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    source_matlab_file = os.path.basename(event["source_file"])
    source_event_file = os.path.basename(row["event_path"])

    string_dtype = h5py.string_dtype(encoding="utf-8")
    raw_block = raw_dataset[gate_slice, pulse_slice]
    raw_voltage = np.asarray(raw_block["real"] + 1j * raw_block["imag"], dtype=np.complex64)

    with h5py.File(path, "w") as h:
        h.attrs["format_name"] = "sanya_head_echo_raw_voltage_cut"
        h.attrs["format_version"] = "1.0"
        h.attrs["event_id"] = event_id
        h.attrs["site"] = site
        h.attrs["station_az_deg"] = float(event["az_deg"])
        h.attrs["station_el_deg"] = float(event["el_deg"])
        h.attrs["station_pointing"] = "azimuth/elevation in degrees from the source MATLAB para array"
        h.attrs["source_matlab_file"] = source_matlab_file
        h.attrs["source_event_file"] = source_event_file
        h.attrs["source_time_zone"] = f"{SOURCE_TIMEZONE_NAME} (UTC+{SOURCE_TIMEZONE_OFFSET_HOURS})"
        h.attrs["unix_sample_unit"] = "nanoseconds since 1970-01-01T00:00:00 UTC at transmit-pulse start"
        h.attrs["raw_voltage_layout"] = "range gate x pulse"
        h.attrs["range_padding_km"] = float(args.range_padding_km)
        h.attrs["time_padding_s"] = float(args.time_padding_s)
        h.attrs["raw_complex_dtype_policy"] = "Original MATLAB real/imag voltage fields are stored as complex64."
        h.attrs["lfm_pulse_length_us"] = float(event["pulse_length_us"])
        h.attrs["lfm_pulse_samples"] = lfm_pulse_samples
        h.attrs["lfm_sample_rate_mhz"] = float(event["sr_mhz"])
        h.attrs["lfm_bandwidth_mhz"] = float(event["bw_mhz"])
        h.attrs["lfm_filter_pre_samples"] = filter_pre
        h.attrs["lfm_filter_post_samples"] = filter_post
        h.attrs["science_range_gate_start_index"] = int(science_g0)
        h.attrs["science_range_gate_stop_index_exclusive"] = int(science_g1)
        h.attrs["science_range_window_note"] = (
            "The requested echo range window is detections plus range_padding_km. "
            "The stored raw_voltage range window is expanded by the 199 us LFM pulse support "
            "so that matched filtering can be regenerated from raw voltage alone."
        )
        h.attrs["selection_reason"] = selection["reason"]
        h.attrs["force_tristatic_include"] = bool(row["force_tristatic"])
        h.attrs["range_poly_fit_rms_m"] = float(selection.get("range_poly_fit_rms_m", np.nan))
        h.attrs["event_max_snr_db"] = float(selection.get("max_snr_db", np.nan))
        h.attrs["sanya_range_correction_km"] = SANYA_RANGE_CORRECTION_KM
        h.attrs["bistatic_min_delay_us"] = BISTATIC_MIN_DELAY_US

        h.create_dataset("raw_voltage", data=raw_voltage, compression="gzip", shuffle=True)
        h.create_dataset("unix_sample", data=utc_cut.astype(np.uint64))
        h.create_dataset("time_ns_utc", data=utc_cut.astype(np.uint64))
        h.create_dataset("beijing_local_time_ns", data=local_cut.astype(np.uint64))
        h.create_dataset("range_km", data=mat["range_km_axis"][gate_slice].astype(np.float64))
        h.create_dataset("range_gate_index", data=np.arange(g0, g1, dtype=np.int32))
        h.create_dataset("science_range_gate_index", data=np.arange(science_g0, science_g1, dtype=np.int32))
        h.create_dataset("matlab_para", data=mat["para"].astype(np.float32))
        h.create_dataset("station_az_deg", data=np.asarray(float(event["az_deg"]), dtype=np.float64))
        h.create_dataset("station_el_deg", data=np.asarray(float(event["el_deg"]), dtype=np.float64))
        h.create_dataset("lfm_pulse_length_us", data=np.asarray(float(event["pulse_length_us"]), dtype=np.float64))
        h.create_dataset("lfm_sample_rate_mhz", data=np.asarray(float(event["sr_mhz"]), dtype=np.float64))
        h.create_dataset("lfm_bandwidth_mhz", data=np.asarray(float(event["bw_mhz"]), dtype=np.float64))

        det = h.create_group("detections")
        det.create_dataset("time_ns_utc", data=(event_local_ns.astype(np.int64) - UTC8_NS).astype(np.uint64))
        det.create_dataset("beijing_local_time_ns", data=event_local_ns.astype(np.uint64))
        det.create_dataset("pulse_index", data=rel_detection_pulses.astype(np.int32))
        det.create_dataset("range_gate_index", data=np.asarray(event["range_gate"], dtype=np.int32))
        det.create_dataset("range_gate_index_in_cut", data=rel_detection_gates.astype(np.int32))
        det.create_dataset("range_km", data=np.asarray(event["range_km"], dtype=np.float64))
        det.create_dataset("corrected_range_km", data=np.asarray(selection["corrected_range_km"], dtype=np.float64))
        det.create_dataset("snr_peak_db", data=np.asarray(event["snr_peak_db"], dtype=np.float32))
        det.create_dataset("selected_for_cut", data=selected_mask_in_cut)
        if "height_km" in selection:
            det.create_dataset("height_km", data=np.asarray(selection["height_km"], dtype=np.float64))

        h.create_dataset("source_matlab_file", data=np.asarray(source_matlab_file, dtype=string_dtype))
        h.create_dataset("source_event_file", data=np.asarray(source_event_file, dtype=string_dtype))

    return {
        "event_id": event_id,
        "site": site,
        "cut_h5": path,
        "status": "written",
        "source_matlab_file": source_matlab_file,
        "source_event_file": source_event_file,
        "n_pulses": int(p1 - p0),
        "n_range_gates": int(g1 - g0),
        "n_science_range_gates": int(science_g1 - science_g0),
        "n_detections": int(len(event["range_gate"])),
        "n_selected_detections": int(np.count_nonzero(selected_mask_in_cut)),
        "station_az_deg": float(event["az_deg"]),
        "station_el_deg": float(event["el_deg"]),
        "force_tristatic_include": bool(row["force_tristatic"]),
        "range_poly_fit_rms_m": float(selection.get("range_poly_fit_rms_m", np.nan)),
        "event_max_snr_db": float(selection.get("max_snr_db", np.nan)),
    }


def write_manifest(path: str, rows: list[dict], args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["format_name"] = "sanya_head_echo_raw_voltage_cut_index"
        h.attrs["format_version"] = "1.0"
        h.attrs["range_padding_km"] = float(args.range_padding_km)
        h.attrs["time_padding_s"] = float(args.time_padding_s)
        h.attrs["selection_summary"] = (
            "Sanya: polynomial range-fit RMS <= 100 m, plus points outside 80-120 km fixed-beam height "
            "require |v_r| > 10 km/s. Danzhou/Wenchang: event max SNR >= 15 dB, polynomial range-fit "
            "RMS <= 100 m, and selected detections have delay > 800 us. Tri-static index events are "
            "force-included if necessary."
        )
        if rows:
            for key in rows[0].keys():
                values = [row.get(key, "") for row in rows]
                if isinstance(values[0], str):
                    h.create_dataset(key, data=np.asarray(values, dtype=string_dtype))
                elif isinstance(values[0], bool):
                    h.create_dataset(key, data=np.asarray(values, dtype=bool))
                elif isinstance(values[0], int):
                    h.create_dataset(key, data=np.asarray(values, dtype=np.int64))
                else:
                    h.create_dataset(key, data=np.asarray(values, dtype=np.float64))


def write_manifest_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    events = collect_selected_events(args)
    if not events:
        raise RuntimeError("No events survived selection.")

    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in events:
        by_source[(row["site"], row["event"]["source_file"])].append(row)

    manifest_rows = []
    for idx, ((site, source_file), rows) in enumerate(sorted(by_source.items()), start=1):
        print(f"[{idx}/{len(by_source)}] {site} {source_file} ({len(rows)} events)", flush=True)
        mat = load_mat_file(source_file, site)
        with h5py.File(source_file, "r") as raw_h:
            raw_dataset = raw_h["data_raw"]
            for row in rows:
                manifest_rows.append(write_cut(row, mat, raw_dataset, args))

    manifest_rows.sort(key=lambda row: (row["site"], row["event_id"]))
    write_manifest(args.manifest, manifest_rows, args)
    write_manifest_csv(args.manifest_csv, manifest_rows)
    counts = defaultdict(int)
    forced = defaultdict(int)
    for row in manifest_rows:
        counts[row["site"]] += 1
        if row.get("force_tristatic_include"):
            forced[row["site"]] += 1
    print(f"wrote/kept {len(manifest_rows)} raw-voltage cut files")
    for site in SITE_ORDER:
        print(f"{site}: {counts[site]} events, {forced[site]} tri-static force-included")
    print(args.manifest)


if __name__ == "__main__":
    main()
