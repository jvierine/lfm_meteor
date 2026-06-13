#!/usr/bin/env python3
"""Test the LFM Doppler range-correction sign with satellite passes.

Selection is deliberately simple: high-SNR detections whose raw diagnostic
range offset is 16 +/- 1 km.  There is no boresight-angle cut and no SNR
weighting.  For each event--satellite--alias pass, fit an unweighted polynomial
to observed range, use its derivative for the Doppler range correction, and
choose the sign that gives the smallest within-pass range-offset variance.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


RADAR_FREQUENCY_HZ = 440e6
BANDWIDTH_HZ = 4e6
LFM_DURATION_S = 199e-6
CHIRP_RATE_HZ_PER_S = BANDWIDTH_HZ / LFM_DURATION_S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/satellite_correlation/v20260613c_snr15_full")
    p.add_argument("--output-dir", default="results/satellite_correlation/v20260613c_snr15_full/lfm_doppler_sign_highsnr")
    p.add_argument("--min-snr-db", type=float, default=30.0)
    p.add_argument("--target-offset-km", type=float, default=16.0)
    p.add_argument("--offset-half-width-km", type=float, default=1.0)
    p.add_argument("--min-pulses", type=int, default=8)
    p.add_argument("--poly-degree", type=int, default=2)
    return p.parse_args()


def parse_utc(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def row_float(row: dict, key: str) -> float:
    return float(row[key])


def group_key(row: dict) -> tuple[str, str, int]:
    return row["event_id"], row["sat_id"], int(row["alias_n"])


def group_rows(rows: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return groups


def rms_about_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean((values - np.mean(values)) ** 2.0)))


def polynomial_range_rate(
    time_s: np.ndarray,
    ranges_km: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    degree = max(1, min(int(degree), len(time_s) - 1))
    t_fit = time_s - np.mean(time_s)
    coeff = np.polyfit(t_fit, ranges_km, degree)
    fitted_range_km = np.polyval(coeff, t_fit)
    rate_km_s = np.polyval(np.polyder(coeff), t_fit)
    return rate_km_s, fitted_range_km, rms_about_mean(ranges_km - fitted_range_km) * 1e3


def summarize_pulses(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([row_float(row, key) for row in rows], dtype=float)
    return {
        "mean_offset_km": float(np.mean(values)),
        "rms_about_mean_m": rms_about_mean(values) * 1e3,
    }


def summarize_pass_means(pass_rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in pass_rows], dtype=float)
    return {
        "mean_of_pass_offsets_km": float(np.mean(values)),
        "pass_mean_rms_m": rms_about_mean(values) * 1e3,
    }


def summarize_within_pass_variance(pass_rows: list[dict], key: str) -> dict[str, float]:
    rms_values = np.asarray([row[key.replace("_km", "_internal_rms_m")] for row in pass_rows], dtype=float)
    return {
        "internal_variance_rms_m": float(np.sqrt(np.mean(rms_values**2))),
        "mean_internal_rms_m": float(np.mean(rms_values)),
    }


def build_passes(raw_rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    selected = []
    for row in raw_rows:
        if row_float(row, "snr_db") < args.min_snr_db:
            continue
        if abs(row_float(row, "range_offset_km") - args.target_offset_km) > args.offset_half_width_km:
            continue
        selected.append(dict(row))

    corrected_rows = []
    pass_summaries = []
    lfm_factor_s = RADAR_FREQUENCY_HZ / CHIRP_RATE_HZ_PER_S
    for key, rows in sorted(group_rows(selected).items()):
        if len(rows) < args.min_pulses:
            continue
        rows = sorted(rows, key=lambda row: row["time_utc"])
        t_unix = np.asarray([parse_utc(row["time_utc"]).timestamp() for row in rows], dtype=float)
        t_rel = t_unix - t_unix[0]
        obs = np.asarray([row_float(row, "observed_range_km") for row in rows], dtype=float)
        pred = np.asarray([row_float(row, "predicted_aliased_range_km") for row in rows], dtype=float)
        snr_db = np.asarray([row_float(row, "snr_db") for row in rows], dtype=float)
        beam = np.asarray([row_float(row, "beam_angle_deg") for row in rows], dtype=float)

        obs_rate_km_s, obs_fit_km, obs_fit_rms_m = polynomial_range_rate(t_rel, obs, args.poly_degree)
        pred_rate_km_s, pred_fit_km, pred_fit_rms_m = polynomial_range_rate(t_rel, pred, args.poly_degree)
        shift_obs_km = -lfm_factor_s * obs_rate_km_s
        shift_pred_km = -lfm_factor_s * pred_rate_km_s

        pass_corrected = []
        for row, obs_rate, pred_rate, obs_fit, pred_fit, obs_shift, pred_shift in zip(
            rows, obs_rate_km_s, pred_rate_km_s, obs_fit_km, pred_fit_km, shift_obs_km, shift_pred_km
        ):
            out = dict(row)
            obs_km = row_float(row, "observed_range_km")
            pred_km = row_float(row, "predicted_aliased_range_km")
            out["observed_poly_range_km"] = float(obs_fit)
            out["predicted_poly_range_km"] = float(pred_fit)
            out["observed_poly_range_rate_km_s"] = float(obs_rate)
            out["predicted_poly_range_rate_km_s"] = float(pred_rate)
            out["lfm_shift_from_observed_rate_km"] = float(obs_shift)
            out["lfm_shift_from_predicted_rate_km"] = float(pred_shift)
            out["offset_no_lfm_km"] = obs_km - pred_km
            out["offset_plus_observed_rate_km"] = obs_km - (pred_km + obs_shift)
            out["offset_minus_observed_rate_km"] = obs_km - (pred_km - obs_shift)
            out["offset_plus_predicted_rate_km"] = obs_km - (pred_km + pred_shift)
            out["offset_minus_predicted_rate_km"] = obs_km - (pred_km - pred_shift)
            pass_corrected.append(out)

        summary = {
            "event_id": key[0],
            "sat_id": key[1],
            "alias_n": key[2],
            "n_pulses": len(rows),
            "start_utc": rows[0]["time_utc"],
            "stop_utc": rows[-1]["time_utc"],
            "mean_snr_db": float(np.mean(snr_db)),
            "max_snr_db": float(np.max(snr_db)),
            "mean_beam_angle_deg": float(np.mean(beam)),
            "min_beam_angle_deg": float(np.min(beam)),
            "max_beam_angle_deg": float(np.max(beam)),
            "beam_angle_span_deg": float(np.max(beam) - np.min(beam)),
            "poly_degree": int(args.poly_degree),
            "observed_poly_fit_rms_m": obs_fit_rms_m,
            "predicted_poly_fit_rms_m": pred_fit_rms_m,
            "observed_range_rate_mean_km_s": float(np.mean(obs_rate_km_s)),
            "predicted_range_rate_mean_km_s": float(np.mean(pred_rate_km_s)),
            "lfm_shift_from_observed_rate_mean_m": float(np.mean(shift_obs_km)) * 1e3,
            "lfm_shift_from_predicted_rate_mean_m": float(np.mean(shift_pred_km)) * 1e3,
            "raw_offset_span_m": float((np.max(obs - pred) - np.min(obs - pred)) * 1e3),
        }
        for offset_key in VARIANTS:
            values = np.asarray([row_float(row, offset_key) for row in pass_corrected], dtype=float)
            summary[offset_key] = float(np.mean(values))
            summary[offset_key.replace("_km", "_internal_rms_m")] = rms_about_mean(values) * 1e3

        corrected_rows.extend(pass_corrected)
        pass_summaries.append(summary)
    return corrected_rows, pass_summaries


VARIANTS = (
    "offset_no_lfm_km",
    "offset_plus_observed_rate_km",
    "offset_minus_observed_rate_km",
    "offset_plus_predicted_rate_km",
    "offset_minus_predicted_rate_km",
)


def make_plot(path: str, pass_summaries: list[dict]) -> None:
    styles = [
        ("offset_no_lfm_km", "No LFM", "#5f6368"),
        ("offset_plus_observed_rate_km", "+ observed-rate shift", "#d95f02"),
        ("offset_minus_observed_rate_km", "- observed-rate shift", "#1b9e77"),
        ("offset_plus_predicted_rate_km", "+ TLE-rate shift", "#7570b3"),
        ("offset_minus_predicted_rate_km", "- TLE-rate shift", "#e7298a"),
    ]
    shift_m = np.asarray([row["lfm_shift_from_observed_rate_mean_m"] for row in pass_summaries], dtype=float)
    beam_span = np.asarray([row["beam_angle_span_deg"] for row in pass_summaries], dtype=float)
    sizes = 32 + 18 * np.sqrt(np.maximum(beam_span, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    ax = axes[0]
    for key, label, color in styles:
        y = np.asarray([row[key.replace("_km", "_internal_rms_m")] for row in pass_summaries], dtype=float)
        ax.scatter(shift_m, y, s=sizes, color=color, alpha=0.78, label=label)
    ax.set_xlabel("Mean LFM range shift from observed polynomial range-rate (m)")
    ax.set_ylabel("Within-pass corrected-offset RMS (m)")
    ax.set_title("Unweighted variance per pass")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    labels = []
    rms = []
    colors = []
    for key, label, color in styles:
        stats = summarize_within_pass_variance(pass_summaries, key)
        labels.append(label)
        rms.append(stats["internal_variance_rms_m"])
        colors.append(color)
    ax.bar(np.arange(len(labels)), rms, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Unweighted within-pass variance RMS (m)")
    ax.set_title("Lower is the correct Doppler sign")
    fig.savefig(path, dpi=220)


def main() -> None:
    args = parse_args()
    raw_path = os.path.join(args.results_dir, "sanya_satellite_detection_matches_raw.csv")
    raw_rows = read_csv(raw_path)
    corrected_rows, pass_summaries = build_passes(raw_rows, args)
    if not corrected_rows:
        raise RuntimeError("No satellite pulses survived the selection.")

    within_pass = {key: summarize_within_pass_variance(pass_summaries, key) for key in VARIANTS}
    best_observed = min(
        ("offset_plus_observed_rate_km", "offset_minus_observed_rate_km"),
        key=lambda key: within_pass[key]["internal_variance_rms_m"],
    )
    output = {
        "input_raw_csv": raw_path,
        "selection": {
            "min_snr_db": args.min_snr_db,
            "beam_angle_filter": None,
            "target_offset_km": args.target_offset_km,
            "offset_half_width_km": args.offset_half_width_km,
            "min_pulses": args.min_pulses,
            "poly_degree": args.poly_degree,
            "weighting": "none",
        },
        "lfm": {
            "radar_frequency_hz": RADAR_FREQUENCY_HZ,
            "bandwidth_hz": BANDWIDTH_HZ,
            "duration_s": LFM_DURATION_S,
            "chirp_rate_hz_per_s": CHIRP_RATE_HZ_PER_S,
            "range_shift_from_rate_km": "-(f0/gamma) * polynomial_range_rate_km_s",
        },
        "n_selected_pulses": len(corrected_rows),
        "n_selected_passes": len(pass_summaries),
        "pulse_summary": {key: summarize_pulses(corrected_rows, key) for key in VARIANTS},
        "pass_mean_summary": {key: summarize_pass_means(pass_summaries, key) for key in VARIANTS},
        "within_pass_variance_summary": within_pass,
        "best_observed_rate_sign": best_observed,
        "best_observed_rate_sign_label": {
            "offset_plus_observed_rate_km": "observed - (predicted + shift)",
            "offset_minus_observed_rate_km": "observed - (predicted - shift)",
        }[best_observed],
        "top_passes_by_pulses": sorted(pass_summaries, key=lambda row: row["n_pulses"], reverse=True)[:20],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    raw_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_raw.csv")
    pass_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_passes.csv")
    summary_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_summary.json")
    plot_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_consistency.png")
    write_csv(raw_out, corrected_rows, list(corrected_rows[0].keys()))
    write_csv(pass_out, pass_summaries, list(pass_summaries[0].keys()))
    make_plot(plot_out, pass_summaries)
    with open(summary_out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
