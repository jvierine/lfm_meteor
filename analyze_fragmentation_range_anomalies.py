#!/usr/bin/env python3
"""Search for fragmentation-like high-SNR range anomalies in head-echo events.

This diagnostic is intentionally conservative.  It applies the same broad
meteor-event selection used for the all-detections time-delay overview, rejects
events with broad or multi-peaked matched-filter profiles, and then searches
for high-SNR detections with large residuals relative to a robust range-time
polynomial fit.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

import sanya_opts as sc


C_MPS = 299792458.0
UTC8_NS = int(8 * 3600 * 1e9)
SANYA_RANGE_CORRECTION_KM = -16.0186
SANYA_AZ_DEG = 15.0
SANYA_EL_DEG = 75.0
SANYA_LOW_HEIGHT_KM = 80.0
SANYA_HIGH_HEIGHT_KM = 120.0
SANYA_OUTSIDE_HEIGHT_MIN_ABS_VELOCITY_KM_S = 10.0
BISTATIC_MIN_DELAY_US = 800.0
SITE_ORDER = ("sanya", "danzhou", "wenchang")
SITE_LABEL = {"sanya": "Sanya", "danzhou": "Danzhou", "wenchang": "Wenchang"}
SITE_COLOR = {"sanya": "#1f77b4", "danzhou": "#d95f02", "wenchang": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-echo-root", default="results/head_echoes")
    p.add_argument("--output-dir", default="results/fragmentation_range_anomalies")
    p.add_argument("--article-figure-dir", default="")
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--monostatic-max-rms-m", type=float, default=100.0)
    p.add_argument("--bistatic-min-max-snr-db", type=float, default=15.0)
    p.add_argument("--bistatic-max-rms-m", type=float, default=100.0)
    p.add_argument("--high-snr-db", type=float, default=15.0)
    p.add_argument("--clean-peak-sidelobe-db", type=float, default=8.0)
    p.add_argument("--clean-mainlobe-power-frac", type=float, default=0.02)
    p.add_argument("--min-clean-event-fraction", type=float, default=0.50)
    p.add_argument("--min-event-median-sidelobe-db", type=float, default=8.0)
    p.add_argument("--residual-floor-m", type=float, default=150.0)
    p.add_argument("--very-large-residual-m", type=float, default=300.0)
    p.add_argument("--relative-rcs-excursion-db", type=float, default=4.0)
    p.add_argument("--max-examples", type=int, default=4)
    p.add_argument("--max-issue-examples", type=int, default=4)
    p.add_argument("--no-utc8-correction", action="store_true")
    return p.parse_args()


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def read_index(root: Path) -> list[dict]:
    rows = []
    with h5py.File(root / "head_echo_index.h5", "r") as h:
        for idx in range(len(h["event_id"])):
            rows.append(
                {
                    "event_id": decode(h["event_id"][idx]),
                    "site": decode(h["site"][idx]).lower(),
                    "event_h5": decode(h["event_h5"][idx]),
                    "n_echoes": int(h["n_echoes"][idx]),
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
            "times_ns": np.asarray(h["times_ns"][:], dtype=np.int64),
            "range_km": np.asarray(h["range_km"][:], dtype=np.float64),
            "range_gate": np.asarray(h["range_gate"][:], dtype=np.int64),
            "snr_peak_db": np.asarray(h["snr_peak_db"][:], dtype=np.float64),
            "echoes": np.asarray(h["echoes"][:], dtype=np.complex64),
            "ranges_km_axis": np.asarray(h["ranges_km_axis"][:], dtype=np.float64),
            "sr_mhz": float(h["sr_mhz"][()]) if "sr_mhz" in h else 4.0,
        }


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.nanmedian(values))
    sigma = 1.4826 * float(np.nanmedian(np.abs(values - med)))
    return sigma


def fit_range(time_ns: np.ndarray, range_km: np.ndarray, degree: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    t_s = (time_ns.astype(np.float64) - float(time_ns[0])) / 1e9
    t_fit = t_s - float(np.mean(t_s))
    if mask is None:
        mask = np.isfinite(range_km)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(range_km)
    degree = max(1, min(int(degree), int(np.count_nonzero(mask)) - 1))
    coeff = np.polyfit(t_fit[mask], range_km[mask].astype(np.float64), degree)
    fitted = np.polyval(coeff, t_fit)
    rms_m = float(np.sqrt(np.mean((range_km[mask] - fitted[mask]) ** 2.0)) * 1e3)
    rate_km_s = np.polyval(np.polyder(coeff), t_fit)
    return fitted, rms_m, rate_km_s


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


def selection_mask(site: str, event: dict, args: argparse.Namespace) -> tuple[bool, np.ndarray, np.ndarray, float, float]:
    time_ns = event["times_ns"].copy()
    if not args.no_utc8_correction:
        time_ns = time_ns - UTC8_NS
    range_km = event["range_km"].astype(np.float64).copy()
    if site == "sanya":
        range_km = range_km + SANYA_RANGE_CORRECTION_KM
    snr_db = event["snr_peak_db"].astype(np.float64)
    finite = np.isfinite(time_ns) & np.isfinite(range_km) & np.isfinite(snr_db)
    if np.count_nonzero(finite) < args.min_points:
        return False, np.zeros(len(range_km), dtype=bool), range_km, np.nan, np.nan

    order = np.argsort(time_ns[finite])
    idx = np.flatnonzero(finite)[order]
    _, rms_m, rate_km_s = fit_range(time_ns[idx], range_km[idx], args.poly_degree)
    point_keep = np.zeros(len(range_km), dtype=bool)
    max_snr_db = float(np.nanmax(snr_db[idx]))
    if site == "sanya":
        keep_event = rms_m <= args.monostatic_max_rms_m
        height_km = sanya_slant_ranges_to_heights_km(range_km[idx])
        outside = (height_km < SANYA_LOW_HEIGHT_KM) | (height_km > SANYA_HIGH_HEIGHT_KM)
        point_keep[idx] = np.isfinite(height_km) & (~outside | (np.abs(rate_km_s) > SANYA_OUTSIDE_HEIGHT_MIN_ABS_VELOCITY_KM_S))
    else:
        delay_us = 2.0 * range_km[idx] * 1e3 / C_MPS * 1e6
        keep_event = max_snr_db >= args.bistatic_min_max_snr_db and rms_m <= args.bistatic_max_rms_m
        point_keep[idx] = delay_us > BISTATIC_MIN_DELAY_US
    return bool(keep_event and np.any(point_keep)), point_keep, range_km, rms_m, max_snr_db


def profile_metrics(echoes: np.ndarray, range_gate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak_sidelobe_db = np.full(echoes.shape[0], np.nan, dtype=np.float64)
    main_frac = np.full(echoes.shape[0], np.nan, dtype=np.float64)
    width_6db = np.full(echoes.shape[0], np.nan, dtype=np.float64)
    for ii, profile in enumerate(echoes):
        amp = np.abs(profile).astype(np.float64)
        if amp.size == 0 or not np.isfinite(amp).any():
            continue
        gate = int(range_gate[ii]) if np.isfinite(range_gate[ii]) else int(np.nanargmax(amp))
        gate = max(0, min(gate, amp.size - 1))
        peak = float(amp[gate])
        if peak <= 0:
            continue
        near0 = max(0, gate - 4)
        near1 = min(amp.size, gate + 5)
        excl0 = max(0, gate - 10)
        excl1 = min(amp.size, gate + 11)
        sidelobe = amp.copy()
        sidelobe[excl0:excl1] = 0.0
        max_side = float(np.nanmax(sidelobe))
        peak_sidelobe_db[ii] = 20.0 * np.log10(peak / max(max_side, 1e-30))
        total_power = float(np.nansum(amp**2.0))
        main_frac[ii] = float(np.nansum(amp[near0:near1] ** 2.0) / total_power) if total_power > 0 else np.nan
        above = np.flatnonzero(amp >= peak / np.sqrt(2.0))
        if above.size:
            width_6db[ii] = float(above.max() - above.min() + 1)
    return peak_sidelobe_db, main_frac, width_6db


def analyze_event(index_row: dict, args: argparse.Namespace) -> dict | None:
    root = Path(args.head_echo_root)
    site = index_row["site"]
    path = resolve_event_path(root, index_row["event_h5"], site)
    event = read_event(path)
    keep_event, point_keep, range_km, initial_rms_m, max_snr_db = selection_mask(site, event, args)
    if not keep_event:
        return None
    time_ns = event["times_ns"].copy()
    if not args.no_utc8_correction:
        time_ns = time_ns - UTC8_NS
    snr_db = event["snr_peak_db"].astype(np.float64)
    finite = point_keep & np.isfinite(time_ns) & np.isfinite(range_km) & np.isfinite(snr_db)
    if np.count_nonzero(finite) < args.min_points:
        return None

    order = np.argsort(time_ns[finite])
    idx = np.flatnonzero(finite)[order]
    time_ns = time_ns[idx]
    range_km = range_km[idx]
    snr_db = snr_db[idx]
    range_gate = event["range_gate"][idx]
    echoes = event["echoes"][idx]

    psr_db, main_frac, width_6db = profile_metrics(echoes, range_gate)
    clean_pulse = (
        np.isfinite(psr_db)
        & np.isfinite(main_frac)
        & (psr_db >= args.clean_peak_sidelobe_db)
        & (main_frac >= args.clean_mainlobe_power_frac)
        & (width_6db <= 12)
    )
    high_snr = snr_db >= args.high_snr_db
    clean_high_snr = clean_pulse & high_snr
    clean_fraction = float(np.count_nonzero(clean_pulse) / len(clean_pulse))
    clean_high_snr_fraction = float(np.count_nonzero(clean_high_snr) / max(1, np.count_nonzero(high_snr)))
    median_psr_db = float(np.nanmedian(psr_db))
    rfi_like = (clean_high_snr_fraction < args.min_clean_event_fraction) or (median_psr_db < args.min_event_median_sidelobe_db)

    fit_mask = clean_pulse
    if np.count_nonzero(fit_mask) < args.min_points:
        fit_mask = np.ones(len(range_km), dtype=bool)
    fitted1, _, _ = fit_range(time_ns, range_km, args.poly_degree, fit_mask)
    resid1_m = (range_km - fitted1) * 1e3
    sigma1 = robust_sigma(resid1_m[fit_mask])
    inlier_mask = fit_mask & (np.abs(resid1_m) <= max(args.residual_floor_m, 4.0 * sigma1 if np.isfinite(sigma1) else args.residual_floor_m))
    if np.count_nonzero(inlier_mask) >= args.min_points:
        fitted, fit_rms_m, _ = fit_range(time_ns, range_km, args.poly_degree, inlier_mask)
    else:
        fitted, fit_rms_m, _ = fit_range(time_ns, range_km, args.poly_degree, fit_mask)
    resid_m = (range_km - fitted) * 1e3
    sigma_m = robust_sigma(resid_m[fit_mask])
    threshold_m = max(args.residual_floor_m, 4.0 * sigma_m if np.isfinite(sigma_m) else args.residual_floor_m)

    # Diagnostic relative RCS proxy: SNR corrected only for range spreading and
    # normalized within each event.  It is not an absolute RCS estimate.
    rel_rcs_db = snr_db + 40.0 * np.log10(np.maximum(range_km, 1e-6))
    rel_rcs_db = rel_rcs_db - float(np.nanmedian(rel_rcs_db))
    anomaly = clean_high_snr & (np.abs(resid_m) >= threshold_m)
    anomaly_rcs = np.abs(rel_rcs_db) >= args.relative_rcs_excursion_db
    event_candidate = (not rfi_like) and (
        (np.count_nonzero(anomaly) >= 2)
        or (np.any(anomaly & anomaly_rcs))
        or np.any(clean_high_snr & (np.abs(resid_m) >= args.very_large_residual_m) & anomaly_rcs)
    )

    max_anom_resid_m = float(np.nanmax(np.abs(resid_m[anomaly]))) if np.any(anomaly) else 0.0
    max_anom_snr_db = float(np.nanmax(snr_db[anomaly])) if np.any(anomaly) else np.nan
    max_anom_rcs_exc_db = float(np.nanmax(np.abs(rel_rcs_db[anomaly]))) if np.any(anomaly) else 0.0
    score = max_anom_resid_m * (1.0 + max(0, np.count_nonzero(anomaly) - 1) / 5.0) + 25.0 * max_anom_rcs_exc_db
    issue_score = (
        max(0.0, max_snr_db - args.high_snr_db)
        + 0.02 * float(np.nanmax(np.abs(resid_m)))
        + 2.5 * max(0.0, args.min_event_median_sidelobe_db - median_psr_db)
        + 0.7 * float(np.nanpercentile(rel_rcs_db, 95) - np.nanpercentile(rel_rcs_db, 5))
    )

    return {
        "event_id": index_row["event_id"],
        "site": site,
        "path": str(path),
        "time_ns": time_ns,
        "range_km": range_km,
        "fitted_range_km": fitted,
        "resid_m": resid_m,
        "snr_db": snr_db,
        "rel_rcs_db": rel_rcs_db,
        "range_gate": range_gate,
        "echoes": echoes,
        "psr_db": psr_db,
        "main_frac": main_frac,
        "clean_pulse": clean_pulse,
        "high_snr": high_snr,
        "anomaly": anomaly,
        "rfi_like": rfi_like,
        "event_candidate": event_candidate,
        "n_points": int(len(range_km)),
        "n_clean": int(np.count_nonzero(clean_pulse)),
        "n_high_snr": int(np.count_nonzero(high_snr)),
        "n_anomaly": int(np.count_nonzero(anomaly)),
        "fit_rms_m": float(fit_rms_m),
        "robust_sigma_m": float(sigma_m),
        "residual_threshold_m": float(threshold_m),
        "clean_fraction": clean_fraction,
        "clean_high_snr_fraction": clean_high_snr_fraction,
        "median_peak_sidelobe_db": median_psr_db,
        "median_mainlobe_power_frac": float(np.nanmedian(main_frac)),
        "max_abs_residual_m": float(np.nanmax(np.abs(resid_m))),
        "max_anomaly_residual_m": max_anom_resid_m,
        "max_anomaly_snr_db": max_anom_snr_db,
        "max_anomaly_rel_rcs_db": max_anom_rcs_exc_db,
        "rel_rcs_p95_p05_db": float(np.nanpercentile(rel_rcs_db, 95) - np.nanpercentile(rel_rcs_db, 5)),
        "max_snr_db": float(max_snr_db),
        "initial_selection_rms_m": float(initial_rms_m),
        "score": float(score),
        "issue_score": float(issue_score),
    }


def scalar_row(event: dict) -> dict:
    keys = [
        "event_id",
        "site",
        "n_points",
        "n_clean",
        "n_high_snr",
        "n_anomaly",
        "event_candidate",
        "rfi_like",
        "fit_rms_m",
        "robust_sigma_m",
        "residual_threshold_m",
        "clean_fraction",
        "clean_high_snr_fraction",
        "median_peak_sidelobe_db",
        "median_mainlobe_power_frac",
        "max_abs_residual_m",
        "max_anomaly_residual_m",
        "max_anomaly_snr_db",
        "max_anomaly_rel_rcs_db",
        "rel_rcs_p95_p05_db",
        "max_snr_db",
        "initial_selection_rms_m",
        "score",
        "issue_score",
        "path",
    ]
    return {key: event[key] for key in keys}


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_tex(path: str, events: list[dict]) -> dict:
    clean = [e for e in events if not e["rfi_like"]]
    candidates = [e for e in clean if e["event_candidate"]]
    by_site = Counter(e["site"] for e in clean)
    cand_site = Counter(e["site"] for e in candidates)
    rfi = [e for e in events if e["rfi_like"]]
    stats = {
        "n_selected": len(events),
        "n_clean": len(clean),
        "n_rfi_like": len(rfi),
        "n_candidates": len(candidates),
        "fraction_candidates_clean": 100.0 * len(candidates) / max(1, len(clean)),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by analyze_fragmentation_range_anomalies.py\n")
        fh.write("\\begin{tabular}{lrrrr}\\toprule\n")
        fh.write("Station & Clean events & Candidate events & Fraction (\\%) & RFI-like rejected \\\\\\midrule\n")
        for site in SITE_ORDER:
            n_clean = by_site[site]
            n_cand = cand_site[site]
            n_rfi = sum(1 for e in rfi if e["site"] == site)
            frac = 100.0 * n_cand / max(1, n_clean)
            fh.write(f"{SITE_LABEL[site]} & {n_clean:d} & {n_cand:d} & {frac:.1f} & {n_rfi:d} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(
            f"All & {len(clean):d} & {len(candidates):d} & {stats['fraction_candidates_clean']:.1f} & {len(rfi):d} \\\\\n"
        )
        fh.write("\\bottomrule\\end{tabular}\n")
    return stats


def plot_example(event: dict, output_dir: str) -> tuple[str, str]:
    t_s = (event["time_ns"].astype(np.float64) - float(event["time_ns"][0])) / 1e9
    anomaly = event["anomaly"]
    clean = event["clean_pulse"]
    site = event["site"]
    color = SITE_COLOR[site]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t_s, event["fitted_range_km"], color="0.15", lw=1.2, label="robust polynomial fit")
    ax.scatter(t_s[clean], event["range_km"][clean], c=event["snr_db"][clean], cmap="viridis", s=22, edgecolor="none")
    if np.any(anomaly):
        ax.scatter(t_s[anomaly], event["range_km"][anomaly], facecolor="none", edgecolor="#d62728", s=70, lw=1.6, label="high-SNR range anomaly")
    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("Range (km)")
    ax.set_title(f"{SITE_LABEL[site]} {event['event_id']}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.axhline(0, color="0.35", lw=0.8)
    ax.axhline(event["residual_threshold_m"], color="#d62728", lw=0.9, ls="--")
    ax.axhline(-event["residual_threshold_m"], color="#d62728", lw=0.9, ls="--")
    ax.scatter(t_s[clean], event["resid_m"][clean], color=color, s=24, alpha=0.75)
    if np.any(anomaly):
        ax.scatter(t_s[anomaly], event["resid_m"][anomaly], facecolor="none", edgecolor="#d62728", s=70, lw=1.6)
    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("Range residual (m)")
    ax.set_title("Range residuals")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(t_s, event["snr_db"], "-o", color="#496d89", ms=3.2, lw=1.0, label="SNR")
    ax2 = ax.twinx()
    ax2.plot(t_s, event["rel_rcs_db"], "-s", color="#b04a2f", ms=3.0, lw=0.9, label="relative RCS proxy")
    if np.any(anomaly):
        ax.scatter(t_s[anomaly], event["snr_db"][anomaly], facecolor="none", edgecolor="#d62728", s=70, lw=1.6)
    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("SNR (dB)")
    ax2.set_ylabel("Relative RCS proxy (dB)")
    ax.set_title("Amplitude fluctuations")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    if np.any(anomaly):
        ii = int(np.flatnonzero(anomaly)[np.argmax(np.abs(event["resid_m"][anomaly]))])
    else:
        ii = int(np.nanargmax(np.abs(event["resid_m"])))
    amp = np.abs(event["echoes"][ii]).astype(np.float64)
    gate = int(event["range_gate"][ii])
    lo = max(0, gate - 90)
    hi = min(len(amp), gate + 91)
    rel_gate = np.arange(lo, hi) - gate
    amp_db = 20.0 * np.log10(np.maximum(amp[lo:hi] / np.nanmax(amp[lo:hi]), 1e-12))
    ax.plot(rel_gate, amp_db, color="0.15", lw=1.0)
    ax.axvline(0, color="#d62728", lw=1.0)
    ax.axhline(-event["psr_db"][ii], color="0.45", lw=0.8, ls=":")
    ax.set_ylim(-45, 2)
    ax.set_xlabel("Range gate relative to detection")
    ax.set_ylabel("Matched-filter amplitude (dB)")
    ax.set_title(f"Peak/sidelobe = {event['psr_db'][ii]:.1f} dB")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Fragmentation-like diagnostic: residual {event['max_anomaly_residual_m']:.0f} m, "
        f"relative RCS excursion {event['max_anomaly_rel_rcs_db']:.1f} dB",
        fontsize=11,
    )
    os.makedirs(output_dir, exist_ok=True)
    safe = event["event_id"].replace("/", "_")
    png = os.path.join(output_dir, f"fragmentation_candidate_{safe}.png")
    pdf = os.path.join(output_dir, f"fragmentation_candidate_{safe}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_population(events: list[dict], output_dir: str) -> tuple[str, str]:
    clean = np.asarray([not e["rfi_like"] for e in events], dtype=bool)
    cand = np.asarray([e["event_candidate"] and not e["rfi_like"] for e in events], dtype=bool)
    max_resid = np.asarray([e["max_abs_residual_m"] for e in events], dtype=float)
    rcs_span = np.asarray([e["rel_rcs_p95_p05_db"] for e in events], dtype=float)
    psr = np.asarray([e["median_peak_sidelobe_db"] for e in events], dtype=float)
    sites = np.asarray([e["site"] for e in events])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    ax = axes[0]
    for site in SITE_ORDER:
        m = clean & (sites == site)
        ax.scatter(max_resid[m], rcs_span[m], s=14, alpha=0.45, color=SITE_COLOR[site], label=SITE_LABEL[site])
    ax.scatter(max_resid[cand], rcs_span[cand], facecolor="none", edgecolor="#d62728", s=48, lw=1.1, label="candidate")
    ax.set_xlabel("Maximum range residual (m)")
    ax.set_ylabel("Relative RCS proxy p95-p05 (dB)")
    ax.set_title("Clean events")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    bins = np.linspace(0, max(40, np.nanpercentile(psr[np.isfinite(psr)], 99)), 40)
    ax.hist(psr[~clean], bins=bins, color="0.65", alpha=0.7, label="RFI-like rejected")
    ax.hist(psr[clean], bins=bins, histtype="step", color="0.15", lw=1.4, label="retained")
    ax.set_xlabel("Median peak/sidelobe separation (dB)")
    ax.set_ylabel("Events")
    ax.set_title("Matched-filter cleanliness")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, "fragmentation_population_summary.png")
    pdf = os.path.join(output_dir, "fragmentation_population_summary.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_data_issue_example(event: dict, output_dir: str) -> tuple[str, str]:
    t_s = (event["time_ns"].astype(np.float64) - float(event["time_ns"][0])) / 1e9
    amp = np.abs(event["echoes"]).astype(np.float64)
    power_db = 20.0 * np.log10(np.maximum(amp, 1e-12))
    power_db = power_db - float(np.nanmedian(power_db))
    n_gate = power_db.shape[1]
    gate_axis = np.arange(n_gate, dtype=np.float64)

    median_gate = float(np.nanmedian(event["range_gate"]))
    gate_spread = float(np.nanmax(event["range_gate"]) - np.nanmin(event["range_gate"]))
    half_width = max(50.0, 0.5 * gate_spread + 60.0)
    lo = max(0, int(np.floor(median_gate - half_width)))
    hi = min(n_gate, int(np.ceil(median_gate + half_width)))
    if hi <= lo + 3:
        lo, hi = 0, n_gate

    unclean_high_snr = event["high_snr"] & ~event["clean_pulse"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), constrained_layout=True)

    ax = axes[0, 0]
    mesh = ax.pcolormesh(
        t_s,
        gate_axis[lo:hi],
        power_db[:, lo:hi].T,
        shading="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=max(20.0, min(65.0, float(np.nanpercentile(power_db[:, lo:hi], 99.7)))),
    )
    ax.scatter(t_s, event["range_gate"], c="white", s=9, edgecolor="none", alpha=0.9, label="detected gate")
    if np.any(unclean_high_snr):
        ax.scatter(
            t_s[unclean_high_snr],
            event["range_gate"][unclean_high_snr],
            facecolor="none",
            edgecolor="#55c6ff",
            s=55,
            lw=1.3,
            label="high-SNR rejected pulse",
        )
    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("Range gate")
    ax.set_title("Matched-filter RTI")
    ax.legend(loc="best", fontsize=7)
    cb = fig.colorbar(mesh, ax=ax, pad=0.015)
    cb.set_label("Relative amplitude (dB)")

    ax = axes[0, 1]
    ax.axhline(0, color="0.35", lw=0.8)
    ax.plot(t_s, event["resid_m"], color="0.25", lw=0.9)
    ax.scatter(t_s[event["clean_pulse"]], event["resid_m"][event["clean_pulse"]], color="#1f77b4", s=18, label="clean pulse")
    if np.any(unclean_high_snr):
        ax.scatter(
            t_s[unclean_high_snr],
            event["resid_m"][unclean_high_snr],
            facecolor="none",
            edgecolor="#d62728",
            s=58,
            lw=1.3,
            label="rejected high-SNR pulse",
        )
    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("Range residual (m)")
    ax.set_title("Fit residuals")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)

    ax = axes[1, 0]
    ax.plot(t_s, event["snr_db"], "-o", color="#496d89", ms=3.0, lw=0.9, label="SNR")
    ax2 = ax.twinx()
    ax2.plot(t_s, event["rel_rcs_db"], "-s", color="#b04a2f", ms=2.8, lw=0.8, label="relative RCS proxy")
    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("SNR (dB)")
    ax2.set_ylabel("Relative RCS proxy (dB)")
    ax.set_title("Amplitude behavior")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    if np.any(unclean_high_snr):
        bad_indices = np.flatnonzero(unclean_high_snr)
        ii = int(bad_indices[np.argmin(event["psr_db"][bad_indices])])
    else:
        ii = int(np.nanargmin(event["psr_db"]))
    profile = amp[ii]
    gate = int(event["range_gate"][ii])
    plo = max(0, gate - 100)
    phi = min(len(profile), gate + 101)
    rel_gate = np.arange(plo, phi) - gate
    profile_db = 20.0 * np.log10(np.maximum(profile[plo:phi] / np.nanmax(profile[plo:phi]), 1e-12))
    ax.plot(rel_gate, profile_db, color="0.15", lw=1.0)
    ax.axvline(0, color="#d62728", lw=1.0)
    ax.axhline(-event["psr_db"][ii], color="0.45", lw=0.8, ls=":")
    ax.set_ylim(-45, 2)
    ax.set_xlabel("Range gate relative to detection")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title(f"Rejected pulse peak/sidelobe = {event['psr_db'][ii]:.1f} dB")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Detected data issue: {SITE_LABEL[event['site']]} {event['event_id']} "
        f"(clean high-SNR fraction {event['clean_high_snr_fraction']:.2f}, median peak/sidelobe {event['median_peak_sidelobe_db']:.1f} dB)",
        fontsize=10.5,
    )
    os.makedirs(output_dir, exist_ok=True)
    safe = event["event_id"].replace("/", "_")
    png = os.path.join(output_dir, f"fragmentation_data_issue_{safe}.png")
    pdf = os.path.join(output_dir, f"fragmentation_data_issue_{safe}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    events = []
    for row in read_index(Path(args.head_echo_root)):
        if row["site"] not in SITE_ORDER:
            continue
        out = analyze_event(row, args)
        if out is not None:
            events.append(out)
    if not events:
        raise RuntimeError("No events survived the diagnostic selection.")

    csv_path = os.path.join(args.output_dir, "fragmentation_range_anomaly_events.csv")
    write_csv(csv_path, [scalar_row(e) for e in events])
    summary_tex = os.path.join(args.output_dir, "fragmentation_range_anomaly_summary.tex")
    stats = write_summary_tex(summary_tex, events)
    pop_png, pop_pdf = plot_population(events, args.output_dir)

    examples = sorted(
        [e for e in events if e["event_candidate"] and not e["rfi_like"]],
        key=lambda e: e["score"],
        reverse=True,
    )[: args.max_examples]
    example_paths = []
    for event in examples:
        example_paths.append(plot_example(event, args.output_dir))

    issue_examples = sorted(
        [e for e in events if e["rfi_like"] and e["n_high_snr"] >= args.min_points],
        key=lambda e: e["issue_score"],
        reverse=True,
    )[: args.max_issue_examples]
    issue_paths = []
    for event in issue_examples:
        issue_paths.append(plot_data_issue_example(event, args.output_dir))

    if args.article_figure_dir:
        os.makedirs(args.article_figure_dir, exist_ok=True)
        import shutil

        for path in [pop_png, pop_pdf] + [p for pair in example_paths + issue_paths for p in pair]:
            shutil.copy2(path, os.path.join(args.article_figure_dir, os.path.basename(path)))

    print(f"events_analyzed={len(events)}")
    print(f"clean_events={stats['n_clean']}")
    print(f"rfi_like_rejected={stats['n_rfi_like']}")
    print(f"fragmentation_like_candidates={stats['n_candidates']}")
    print(f"candidate_fraction_clean_percent={stats['fraction_candidates_clean']:.2f}")
    print(csv_path)
    print(summary_tex)
    print(pop_png)
    for png, _ in example_paths:
        print(png)
    for png, _ in issue_paths:
        print(png)


if __name__ == "__main__":
    main()
