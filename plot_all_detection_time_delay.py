#!/usr/bin/env python3
"""Plot all monostatic and bistatic head-echo detections versus UTC time and delay."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import h5py
import jcoord
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import noise_model
import plot_pygdsm_station_sky_noise as sky_model
import sanya_opts as sc


C_MPS = 299792458.0
UTC8_NS = int(8 * 3600 * 1e9)
SANYA_RANGE_CORRECTION_KM = -16.0186
SITE_ORDER = ("sanya", "danzhou", "wenchang")
SITE_LABEL = {"sanya": "Sanya monostatic", "danzhou": "Danzhou bistatic", "wenchang": "Wenchang bistatic"}
SITE_MARKER = {"sanya": "o", "danzhou": "^", "wenchang": "s"}
SITE_COLOR = {"sanya": "#1f77b4", "danzhou": "#d95f02", "wenchang": "#2ca02c"}
NOISE_SITE_LABEL = {"sanya": "Sanya", "danzhou": "Danzhou", "wenchang": "Wenchang"}
NOISE_SITE_COLOR = {"sanya": "#1f77b4", "danzhou": "#2ca02c", "wenchang": "#d62728"}
SANYA_AZ_DEG = 15.0
SANYA_EL_DEG = 75.0
SANYA_LOW_HEIGHT_KM = 80.0
SANYA_HIGH_HEIGHT_KM = 120.0
SANYA_OUTSIDE_HEIGHT_MIN_ABS_VELOCITY_KM_S = 10.0
BISTATIC_MIN_DELAY_US = 800.0
DEFAULT_NOISE_H5 = "results/sanya_4mhz_system_noise_power_100pulse.h5"
DEFAULT_TRISTATIC_H5 = "results/tristatic_results.h5"
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/all_detection_time_delay_velocity.png"
PAPER_OUTPUT_PDF = "/Users/jvi019/src/sanya_tristatic_paper/figures/all_detection_time_delay_velocity.pdf"
NOISE_FLOOR_CADENCE_MIN = 2.5
NOISE_FREQUENCY_MHZ = sc.RADAR_FREQUENCY_MHZ
NOISE_BEAM_RADIUS_DEG = 5.0
NOISE_BEAM_GRID_STEP_DEG = 0.1
SANYA_SOLAR_TIME_UTC_OFFSET_HOURS = float(sc.lon0[0]) / 15.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-echo-root", default="results/head_echoes")
    p.add_argument("--output", default="results/all_detection_time_delay_velocity.png")
    p.add_argument("--pdf", default="results/all_detection_time_delay_velocity.pdf")
    p.add_argument("--csv", default="results/all_detection_time_delay_velocity.csv")
    p.add_argument("--noise-h5", default=DEFAULT_NOISE_H5, help="Reduced 100-pulse system-noise HDF5 product.")
    p.add_argument("--tristatic-h5", default=DEFAULT_TRISTATIC_H5, help="Tri-static candidate association HDF5 product.")
    p.add_argument("--no-noise-panel", action="store_true", help="Do not add the system-noise monitor panel.")
    p.add_argument("--paper-output", default=PAPER_OUTPUT_PNG, help="Optional article PNG copy path.")
    p.add_argument("--paper-pdf", default=PAPER_OUTPUT_PDF, help="Optional article PDF copy path.")
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--monostatic-max-rms-m", type=float, default=100.0)
    p.add_argument("--bistatic-min-max-snr-db", type=float, default=15.0)
    p.add_argument("--bistatic-max-rms-m", type=float, default=100.0)
    p.add_argument("--no-utc8-correction", action="store_true")
    return p.parse_args()


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def read_index(root: Path) -> list[dict]:
    index_path = root / "head_echo_index.h5"
    with h5py.File(index_path, "r") as h:
        rows = []
        for idx in range(len(h["event_id"])):
            rows.append(
                {
                    "event_id": decode(h["event_id"][idx]),
                    "site": decode(h["site"][idx]).lower(),
                    "event_h5": decode(h["event_h5"][idx]),
                    "n_echoes": int(h["n_echoes"][idx]),
                    "median_range_km": float(h["median_range_km"][idx]),
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


def fit_track(time_ns: np.ndarray, range_km: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    t_s = (time_ns.astype(np.float64) - float(time_ns[0])) / 1e9
    t_centered = t_s - float(np.mean(t_s))
    degree = max(1, min(int(degree), len(t_s) - 1))
    coeff = np.polyfit(t_centered, range_km.astype(np.float64), degree)
    fitted = np.polyval(coeff, t_centered)
    rate_km_s = np.polyval(np.polyder(coeff), t_centered)
    rms_m = float(np.sqrt(np.mean((range_km - fitted) ** 2.0)) * 1e3)
    return rate_km_s, rms_m


def sanya_slant_ranges_to_heights_km(ranges_km: np.ndarray) -> np.ndarray:
    heights = np.full(np.asarray(ranges_km).shape, np.nan, dtype=np.float64)
    for idx, range_km in enumerate(np.asarray(ranges_km, dtype=np.float64)):
        if not np.isfinite(range_km):
            continue
        llh = jcoord.az_el_r2geodetic(
            sc.lat0[0],
            sc.lon0[0],
            sc.alt0[0] * 1e3,
            SANYA_AZ_DEG,
            SANYA_EL_DEG,
            float(range_km) * 1e3,
        )
        heights[idx] = float(llh[2] / 1e3)
    return heights


def event_rows(root: Path, index_row: dict, args: argparse.Namespace) -> tuple[list[dict], dict | None]:
    site = index_row["site"]
    path = resolve_event_path(root, index_row["event_h5"], site)
    with h5py.File(path, "r") as h:
        time_ns = np.asarray(h["times_ns"][:], dtype=np.int64)
        range_km = np.asarray(h["range_km"][:], dtype=np.float64)
        snr_db = np.asarray(h["snr_peak_db"][:], dtype=np.float64)

    if site == "sanya":
        range_km = range_km + SANYA_RANGE_CORRECTION_KM
    if not args.no_utc8_correction:
        time_ns = time_ns - UTC8_NS

    finite = np.isfinite(range_km) & np.isfinite(snr_db)
    if np.count_nonzero(finite) < args.min_points:
        return [], None
    time_ns = time_ns[finite]
    range_km = range_km[finite]
    snr_db = snr_db[finite]
    order = np.argsort(time_ns)
    time_ns = time_ns[order]
    range_km = range_km[order]
    snr_db = snr_db[order]

    rate_km_s, rms_m = fit_track(time_ns, range_km, args.poly_degree)
    delay_us = 2.0 * range_km * 1e3 / C_MPS * 1e6
    max_snr_db = float(np.nanmax(snr_db))
    height_km = np.full(len(range_km), np.nan, dtype=np.float64)
    if site == "sanya":
        keep_event = rms_m <= args.monostatic_max_rms_m
        height_km = sanya_slant_ranges_to_heights_km(range_km)
        outside_height_window = (height_km < SANYA_LOW_HEIGHT_KM) | (height_km > SANYA_HIGH_HEIGHT_KM)
        point_keep = np.isfinite(height_km) & (~outside_height_window | (np.abs(rate_km_s) > SANYA_OUTSIDE_HEIGHT_MIN_ABS_VELOCITY_KM_S))
    else:
        keep_event = max_snr_db >= args.bistatic_min_max_snr_db and rms_m <= args.bistatic_max_rms_m
        point_keep = delay_us > BISTATIC_MIN_DELAY_US
    summary = {
        "event_id": index_row["event_id"],
        "site": site,
        "n_points": int(len(range_km)),
        "n_selected_points": int(np.count_nonzero(point_keep)) if keep_event else 0,
        "n_rejected_points": int(np.count_nonzero(~point_keep)) if keep_event else int(len(range_km)),
        "max_snr_db": max_snr_db,
        "range_poly_fit_rms_m": rms_m,
        "selected": bool(keep_event and np.any(point_keep)),
    }
    if not keep_event or not np.any(point_keep):
        return [], summary

    rows = []
    for ii in range(len(range_km)):
        if not point_keep[ii]:
            continue
        rows.append(
            {
                "event_id": index_row["event_id"],
                "site": site,
                "time_ns": int(time_ns[ii]),
                "utc_iso": np.datetime_as_string(np.datetime64(int(time_ns[ii]), "ns"), unit="ms"),
                "range_km": float(range_km[ii]),
                "height_km": float(height_km[ii]) if np.isfinite(height_km[ii]) else "",
                "delay_us": float(delay_us[ii]),
                "radial_velocity_km_s": float(rate_km_s[ii]),
                "snr_peak_db": float(snr_db[ii]),
                "n_event_points": int(len(range_km)),
                "event_max_snr_db": max_snr_db,
                "range_poly_fit_rms_m": rms_m,
            }
        )
    return rows, summary


def load_selected_points(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    root = Path(args.head_echo_root)
    rows = []
    summaries = []
    for index_row in read_index(root):
        if index_row["site"] not in SITE_ORDER:
            continue
        event_points, summary = event_rows(root, index_row, args)
        if summary is not None:
            summaries.append(summary)
        rows.extend(event_points)
    return rows, summaries


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = dict(row)
            for key in (
                "time_ns",
                "range_km",
                "delay_us",
                "radial_velocity_km_s",
                "snr_peak_db",
                "n_event_points",
                "event_max_snr_db",
                "range_poly_fit_rms_m",
            ):
                if key not in parsed or parsed[key] == "":
                    continue
                parsed[key] = int(float(parsed[key])) if key in {"time_ns", "n_event_points"} else float(parsed[key])
            if parsed.get("height_km", "") != "":
                parsed["height_km"] = float(parsed["height_km"])
            rows.append(parsed)
    return rows


def ns_to_datetimes(time_ns: np.ndarray) -> np.ndarray:
    return np.asarray([np.datetime64(int(t), "ns").astype("datetime64[ms]").astype(object) for t in time_ns])


def solar_time_formatter(x: float, _pos: int | None = None) -> str:
    dt = mdates.num2date(x)
    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    solar_hour = (utc_hour + SANYA_SOLAR_TIME_UTC_OFFSET_HOURS) % 24.0
    hour = int(np.floor(solar_hour))
    minute = int(np.round((solar_hour - hour) * 60.0))
    if minute >= 60:
        hour = (hour + 1) % 24
        minute = 0
    return f"{hour:02d}:{minute:02d}"


def hourly_event_count_series(rows: list[dict], site: str) -> tuple[np.ndarray, np.ndarray]:
    subset = [row for row in rows if row["site"] == site]
    if not subset:
        return np.array([], dtype=object), np.array([], dtype=int)
    time_ns = np.asarray([row["time_ns"] for row in subset], dtype=np.int64)
    event_ids = np.asarray([str(row["event_id"]) for row in subset], dtype=object)
    start_hour = np.datetime64(int(np.min(time_ns)), "ns").astype("datetime64[h]")
    stop_hour = np.datetime64(int(np.max(time_ns)), "ns").astype("datetime64[h]") + np.timedelta64(1, "h")
    bin_edges = np.arange(start_hour, stop_hour + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    bin_edges_ns = bin_edges.astype("datetime64[ns]").astype(np.int64)
    counts = np.zeros(len(bin_edges_ns) - 1, dtype=int)
    bin_index = np.searchsorted(bin_edges_ns, time_ns, side="right") - 1
    valid = (bin_index >= 0) & (bin_index < len(counts))
    for bi in np.unique(bin_index[valid]):
        counts[bi] = len(set(event_ids[valid][bin_index[valid] == bi]))
    centers = bin_edges[:-1] + np.timedelta64(30, "m")
    return centers.astype("datetime64[ms]").astype(object), counts


def hourly_timestamp_count_series(time_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(time_ns) == 0:
        return np.array([], dtype=object), np.array([], dtype=int)
    start_hour = np.datetime64(int(np.min(time_ns)), "ns").astype("datetime64[h]")
    stop_hour = np.datetime64(int(np.max(time_ns)), "ns").astype("datetime64[h]") + np.timedelta64(1, "h")
    bin_edges = np.arange(start_hour, stop_hour + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    bin_edges_ns = bin_edges.astype("datetime64[ns]").astype(np.int64)
    counts, _ = np.histogram(time_ns, bins=bin_edges_ns)
    centers = bin_edges[:-1] + np.timedelta64(30, "m")
    return centers.astype("datetime64[ms]").astype(object), counts


def event_id_timestamp_ns(event_id: str) -> int | None:
    try:
        return int(event_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def tristatic_candidate_count_series(path: str | os.PathLike, apply_utc8_correction: bool) -> tuple[np.ndarray, np.ndarray]:
    tristatic_path = Path(path)
    if not tristatic_path.exists():
        return np.array([], dtype=object), np.array([], dtype=int)
    with h5py.File(tristatic_path, "r") as h:
        if "summary_event_id" not in h:
            return np.array([], dtype=object), np.array([], dtype=int)
        event_ids = [decode(value) for value in h["summary_event_id"][:]]
    time_ns = np.asarray([t for event_id in event_ids if (t := event_id_timestamp_ns(event_id)) is not None], dtype=np.int64)
    if apply_utc8_correction:
        time_ns = time_ns - UTC8_NS
    return hourly_timestamp_count_series(time_ns)


def bin_quantile(
    time_ns: np.ndarray,
    values: np.ndarray,
    bin_edges_ns: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = (bin_edges_ns[:-1] + np.diff(bin_edges_ns) // 2).astype(np.int64)
    binned = np.full(len(centers), np.nan, dtype=np.float64)
    finite = np.isfinite(time_ns) & np.isfinite(values)
    idx = np.searchsorted(bin_edges_ns, time_ns[finite], side="right") - 1
    valid = (idx >= 0) & (idx < len(centers))
    idx = idx[valid]
    vals = values[finite][valid]
    for bi in np.unique(idx):
        binned[bi] = np.nanquantile(vals[idx == bi], quantile)
    ok = np.isfinite(binned)
    return centers[ok], binned[ok]


def load_noise_floor_temperature(path: str | os.PathLike) -> dict[str, dict[str, np.ndarray]]:
    noise_path = Path(path)
    if not noise_path.exists():
        return {}
    with h5py.File(noise_path, "r") as h:
        site_names = [decode(value) for value in h["site_names"][:]]
        site_names_lower = [name.lower() for name in site_names]
        station_id = np.asarray(h["bins/station_id"][:], dtype=np.int64)
        time_ns = np.asarray(h["bins/time_utc_mid_ns"][:], dtype=np.int64)
        power = np.asarray(h["bins/noise_power_mean_raw_voltage"][:], dtype=np.float64)

    start_ns = int(np.nanmin(time_ns))
    stop_ns = int(np.nanmax(time_ns))
    step_ns = int(round(NOISE_FLOOR_CADENCE_MIN * 60.0 * 1e9))
    bin_edges_ns = np.arange(start_ns, stop_ns + step_ns, step_ns, dtype=np.int64)
    if bin_edges_ns[-1] < stop_ns:
        bin_edges_ns = np.append(bin_edges_ns, stop_ns)
    else:
        bin_edges_ns[-1] = stop_ns

    times = sky_model.make_times(start_ns, stop_ns, NOISE_FLOOR_CADENCE_MIN)
    sky_x_ns = np.asarray(times.unix * 1e9, dtype=np.float64)
    gsm = sky_model.pygdsm.GlobalSkyModel(freq_unit="MHz", include_cmb=False)

    monitors: dict[str, dict[str, np.ndarray]] = {}
    for site in SITE_ORDER:
        if site not in site_names_lower:
            continue
        site_id = site_names_lower.index(site)
        site_name = site_names[site_id]
        if site_name not in noise_model.POWER_PER_K or site_name not in noise_model.FITTED_T_REC_K:
            continue
        floor_quantile = 0.50 if site == "sanya" else 0.10
        keep = (station_id == site_id) & np.isfinite(power) & (power > 0.0)
        if np.count_nonzero(keep) == 0:
            continue
        floor_ns, floor_power = bin_quantile(time_ns[keep], power[keep], bin_edges_ns, floor_quantile)
        floor_tsys_k = floor_power / noise_model.POWER_PER_K[site_name]
        t_sky, _n_samples = sky_model.station_sky_temperature(
            gsm,
            site_name,
            times,
            NOISE_FREQUENCY_MHZ,
            False,
            NOISE_BEAM_RADIUS_DEG,
            NOISE_BEAM_GRID_STEP_DEG,
            1.0,
        )
        fitted_tsys_k = t_sky + noise_model.FITTED_T_REC_K[site_name]
        monitors[site] = {
            "floor_time": ns_to_datetimes(floor_ns),
            "floor_tsys_k": floor_tsys_k,
            "model_time": mdates.num2date(sky_model.time_to_mpl(times)),
            "model_tsys_k": fitted_tsys_k,
        }
    return monitors


def make_plot(path: str, pdf_path: str, rows: list[dict], summaries: list[dict], args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    noise_monitor = {} if args.no_noise_panel else load_noise_floor_temperature(args.noise_h5)
    if noise_monitor:
        fig, (ax, ax_count, ax_noise) = plt.subplots(
            3,
            1,
            figsize=(8.4, 8.0),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
        )
    else:
        fig, (ax, ax_count) = plt.subplots(
            2,
            1,
            figsize=(8.4, 5.8),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        ax_noise = None
    ax_count.set_axisbelow(True)

    for site in SITE_ORDER:
        subset = [row for row in rows if row["site"] == site]
        if not subset:
            continue
        times = ns_to_datetimes(np.asarray([row["time_ns"] for row in subset], dtype=np.int64))
        delays = np.asarray([row["delay_us"] for row in subset], dtype=float)
        ax.scatter(
            times,
            delays,
            color=SITE_COLOR[site],
            s=7.0 if site == "sanya" else 13.0,
            marker=SITE_MARKER[site],
            alpha=0.58 if site == "sanya" else 0.74,
            linewidths=0,
            label=f"{SITE_LABEL[site]} ({len(subset)})",
        )
    ax.set_title("Head-echo detections through the Sanya tri-static experiment")
    if ax_noise is None:
        ax.set_xlabel("")
    ax.set_ylabel("Delay (us)")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_solar = ax.twiny()
    ax_solar.set_xlim(ax.get_xlim())
    ax_solar.xaxis.set_major_locator(ax.xaxis.get_major_locator())
    ax_solar.xaxis.set_major_formatter(mticker.FuncFormatter(solar_time_formatter))
    ax_solar.set_xlabel("Local solar time at Sanya (sunrise = 06:00)")
    ax_solar.tick_params(axis="x", labelsize=9)

    for site in SITE_ORDER:
        centers, counts = hourly_event_count_series(rows, site)
        if len(centers) == 0:
            continue
        positive_counts = np.where(counts > 0, counts, np.nan)
        ax_count.step(
            centers,
            positive_counts,
            where="mid",
            color=SITE_COLOR[site],
            linewidth=1.8,
            label=SITE_LABEL[site],
        )
    tri_centers, tri_counts = tristatic_candidate_count_series(args.tristatic_h5, apply_utc8_correction=not args.no_utc8_correction)
    if len(tri_centers):
        positive_tri_counts = np.where(tri_counts > 0, tri_counts, np.nan)
        ax_count.step(
            tri_centers,
            positive_tri_counts,
            where="mid",
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="Tri-static candidates",
        )
    ax_count.set_ylabel("Events\nper hour")
    ax_count.set_yscale("log")
    ax_count.grid(True, which="both", alpha=0.22)
    ax_count.legend(loc="upper left", ncol=1, frameon=True)
    ax_count.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if ax_noise is not None:
        all_noise_k = []
        for site in SITE_ORDER:
            monitor = noise_monitor.get(site)
            if monitor is None:
                continue
            color = NOISE_SITE_COLOR[site]
            y_k = monitor["floor_tsys_k"]
            all_noise_k.append(y_k[np.isfinite(y_k)])
            model_k = monitor["model_tsys_k"]
            all_noise_k.append(model_k[np.isfinite(model_k)])
            ax_noise.scatter(
                monitor["floor_time"],
                y_k,
                color=color,
                s=12.0,
                alpha=0.72,
                linewidths=0,
                label=NOISE_SITE_LABEL[site],
                zorder=3,
            )
            ax_noise.plot(
                monitor["model_time"],
                monitor["model_tsys_k"],
                color=color,
                linewidth=1.7,
                alpha=0.95,
                zorder=4,
            )
        if all_noise_k:
            combined = np.concatenate(all_noise_k)
            ymax = float(np.nanmax(combined))
            ax_noise.set_ylim(0.0, ymax * 1.04)
        ax_noise.set_xlabel("UTC time")
        ax_noise.set_ylabel("Noise\ntemperature (K)")
        ax_noise.grid(True, alpha=0.22)
        ax_noise.legend(loc="upper right", ncol=3, frameon=True, fontsize=8.5)
        ax_noise.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax_count.set_xlabel("UTC time")

    ax_solar.set_xlim(ax.get_xlim())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=260)
    fig.savefig(pdf_path)
    if args.paper_output:
        os.makedirs(os.path.dirname(args.paper_output) or ".", exist_ok=True)
        shutil.copyfile(path, args.paper_output)
    if args.paper_pdf:
        os.makedirs(os.path.dirname(args.paper_pdf) or ".", exist_ok=True)
        shutil.copyfile(pdf_path, args.paper_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    index_path = Path(args.head_echo_root) / "head_echo_index.h5"
    if index_path.exists():
        rows, summaries = load_selected_points(args)
        write_csv(args.csv, rows)
    elif Path(args.csv).exists():
        print(f"Using existing selected-point CSV because {index_path} is not available.")
        rows = read_csv_rows(args.csv)
        summaries = []
    else:
        rows, summaries = load_selected_points(args)
    if not rows:
        raise RuntimeError("No detections survived the selection.")
    make_plot(args.output, args.pdf, rows, summaries, args)
    print(f"wrote {args.output}")
    print(f"wrote {args.pdf}")
    print(f"wrote {args.csv}")
    if args.paper_output:
        print(f"copied {args.paper_output}")
    if args.paper_pdf:
        print(f"copied {args.paper_pdf}")
    for site in SITE_ORDER:
        if summaries:
            n_events = sum(1 for s in summaries if s["site"] == site and s["selected"])
        else:
            n_events = len({r["event_id"] for r in rows if r["site"] == site})
        n_points = sum(1 for r in rows if r["site"] == site)
        n_rejected = sum(s["n_rejected_points"] for s in summaries if s["site"] == site and s["selected"])
        print(f"{site}: {n_events} events, {n_points} detections, {n_rejected} selected-event points rejected")


if __name__ == "__main__":
    main()
