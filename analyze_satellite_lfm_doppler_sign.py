#!/usr/bin/env python3
"""Use high-SNR Sanya satellite passes to test the LFM Doppler range sign.

The test is intentionally narrow: use only bright detections whose raw
observed-minus-predicted aliased range offset is already near the satellite
calibration family, then estimate range-rates from polynomial range fits.
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
    p.add_argument(
        "--results-dir",
        default="results/satellite_correlation/v20260613c_snr15_full",
        help="Directory containing sanya_satellite_detection_matches_raw.csv.",
    )
    p.add_argument(
        "--output-dir",
        default="results/satellite_correlation/v20260613c_snr15_full/lfm_doppler_sign_highsnr",
    )
    p.add_argument("--min-snr-db", type=float, default=30.0)
    p.add_argument("--max-beam-angle-deg", type=float, default=2.0)
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


def snr_weight(snr_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, snr_db / 10.0)


def group_key(row: dict) -> tuple[str, str, int]:
    return row["event_id"], row["sat_id"], int(row["alias_n"])


def group_rows(rows: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return groups


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values) / np.sum(weights))


def weighted_rms(values: np.ndarray, weights: np.ndarray, center: float) -> float:
    return float(np.sqrt(np.sum(weights * (values - center) ** 2.0) / np.sum(weights)))


def weighted_poly_rate(
    time_s: np.ndarray,
    ranges_km: np.ndarray,
    weights: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    degree = max(1, min(int(degree), len(time_s) - 1))
    t_center = weighted_mean(time_s, weights)
    t_fit = time_s - t_center
    coeff = np.polyfit(t_fit, ranges_km, degree, w=np.sqrt(weights))
    rate_coeff = np.polyder(coeff)
    fitted_range_km = np.polyval(coeff, t_fit)
    rate_km_s = np.polyval(rate_coeff, t_fit)
    rms_m = weighted_rms(ranges_km, weights, fitted_range_km) * 1e3
    return rate_km_s, fitted_range_km, rms_m


def summarize_variant(rows: list[dict], offset_key: str) -> dict[str, float]:
    values_km = np.asarray([row_float(row, offset_key) for row in rows], dtype=float)
    weights = np.asarray([row_float(row, "snr_weight") for row in rows], dtype=float)
    mean_km = weighted_mean(values_km, weights)
    return {
        "weighted_mean_offset_km": mean_km,
        "weighted_rms_about_mean_m": weighted_rms(values_km, weights, mean_km) * 1e3,
        "unweighted_mean_offset_km": float(np.mean(values_km)),
        "unweighted_rms_about_mean_m": float(np.sqrt(np.mean((values_km - np.mean(values_km)) ** 2.0)) * 1e3),
    }


def summarize_groups(group_summaries: list[dict], offset_key: str) -> dict[str, float]:
    means_km = np.asarray([row[offset_key] for row in group_summaries], dtype=float)
    weights = np.asarray([row["total_snr_weight"] for row in group_summaries], dtype=float)
    center_km = weighted_mean(means_km, weights)
    return {
        "weighted_mean_of_pass_offsets_km": center_km,
        "snr_weighted_pass_mean_rms_m": weighted_rms(means_km, weights, center_km) * 1e3,
        "unweighted_pass_mean_rms_m": float(np.sqrt(np.mean((means_km - np.mean(means_km)) ** 2.0)) * 1e3),
    }


def build_high_snr_passes(raw_rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    pulse_rows = []
    for row in raw_rows:
        snr_db = row_float(row, "snr_db")
        beam_deg = row_float(row, "beam_angle_deg")
        raw_offset_km = row_float(row, "range_offset_km")
        if snr_db < args.min_snr_db:
            continue
        if beam_deg > args.max_beam_angle_deg:
            continue
        if abs(raw_offset_km - args.target_offset_km) > args.offset_half_width_km:
            continue
        pulse_rows.append(dict(row))

    corrected_rows = []
    group_summaries = []
    lfm_factor_s = RADAR_FREQUENCY_HZ / CHIRP_RATE_HZ_PER_S
    for key, rows in sorted(group_rows(pulse_rows).items()):
        if len(rows) < args.min_pulses:
            continue
        rows = sorted(rows, key=lambda row: row["time_utc"])
        t_unix = np.asarray([parse_utc(row["time_utc"]).timestamp() for row in rows], dtype=float)
        t_rel = t_unix - t_unix[0]
        obs = np.asarray([row_float(row, "observed_range_km") for row in rows], dtype=float)
        pred = np.asarray([row_float(row, "predicted_aliased_range_km") for row in rows], dtype=float)
        snr_db = np.asarray([row_float(row, "snr_db") for row in rows], dtype=float)
        beam = np.asarray([row_float(row, "beam_angle_deg") for row in rows], dtype=float)
        weights = snr_weight(snr_db)

        obs_rate_km_s, obs_fit_km, obs_fit_rms_m = weighted_poly_rate(
            t_rel, obs, weights, args.poly_degree
        )
        pred_rate_km_s, pred_fit_km, pred_fit_rms_m = weighted_poly_rate(
            t_rel, pred, weights, args.poly_degree
        )
        raw_offsets = obs - pred

        # Sanya monostatic one-way range shift from a polynomial range-rate.
        # The two possible sign conventions are tested symmetrically below.
        shift_from_obs_rate_km = -lfm_factor_s * obs_rate_km_s
        shift_from_pred_rate_km = -lfm_factor_s * pred_rate_km_s

        pass_corrected = []
        for row, w, obs_rate, pred_rate, obs_fit, pred_fit, obs_shift, pred_shift in zip(
            rows,
            weights,
            obs_rate_km_s,
            pred_rate_km_s,
            obs_fit_km,
            pred_fit_km,
            shift_from_obs_rate_km,
            shift_from_pred_rate_km,
        ):
            out = dict(row)
            obs_km = row_float(row, "observed_range_km")
            pred_km = row_float(row, "predicted_aliased_range_km")
            out["snr_weight"] = float(w)
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

        pass_weights = np.asarray([row_float(row, "snr_weight") for row in pass_corrected], dtype=float)
        summary = {
            "event_id": key[0],
            "sat_id": key[1],
            "alias_n": key[2],
            "n_pulses": len(rows),
            "start_utc": rows[0]["time_utc"],
            "stop_utc": rows[-1]["time_utc"],
            "total_snr_weight": float(np.sum(pass_weights)),
            "mean_snr_db": float(np.mean(snr_db)),
            "max_snr_db": float(np.max(snr_db)),
            "snr_weighted_beam_angle_deg": weighted_mean(beam, weights),
            "min_beam_angle_deg": float(np.min(beam)),
            "poly_degree": args.poly_degree,
            "observed_poly_fit_rms_m": obs_fit_rms_m,
            "predicted_poly_fit_rms_m": pred_fit_rms_m,
            "observed_range_rate_mean_km_s": weighted_mean(obs_rate_km_s, weights),
            "predicted_range_rate_mean_km_s": weighted_mean(pred_rate_km_s, weights),
            "lfm_shift_from_observed_rate_mean_m": weighted_mean(shift_from_obs_rate_km, weights) * 1e3,
            "lfm_shift_from_predicted_rate_mean_m": weighted_mean(shift_from_pred_rate_km, weights) * 1e3,
            "raw_offset_span_m": float((np.max(raw_offsets) - np.min(raw_offsets)) * 1e3),
        }
        for offset_key in (
            "offset_no_lfm_km",
            "offset_plus_observed_rate_km",
            "offset_minus_observed_rate_km",
            "offset_plus_predicted_rate_km",
            "offset_minus_predicted_rate_km",
        ):
            values = np.asarray([row_float(row, offset_key) for row in pass_corrected], dtype=float)
            summary[offset_key] = weighted_mean(values, pass_weights)
            summary[offset_key.replace("_km", "_internal_rms_m")] = weighted_rms(
                values, pass_weights, summary[offset_key]
            ) * 1e3

        corrected_rows.extend(pass_corrected)
        group_summaries.append(summary)

    return corrected_rows, group_summaries


def make_plot(path: str, group_summaries: list[dict]) -> None:
    variants = [
        ("offset_no_lfm_km", "No LFM", "#5f6368"),
        ("offset_plus_observed_rate_km", "+ observed-rate shift", "#d95f02"),
        ("offset_minus_observed_rate_km", "- observed-rate shift", "#1b9e77"),
        ("offset_plus_predicted_rate_km", "+ TLE-rate shift", "#7570b3"),
        ("offset_minus_predicted_rate_km", "- TLE-rate shift", "#e7298a"),
    ]
    weights = np.asarray([row["total_snr_weight"] for row in group_summaries], dtype=float)
    shift_m = np.asarray([row["lfm_shift_from_observed_rate_mean_m"] for row in group_summaries], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    ax = axes[0]
    for key, label, color in variants:
        offsets = np.asarray([row[key] for row in group_summaries], dtype=float)
        center = weighted_mean(offsets, weights)
        ax.scatter(
            shift_m,
            (offsets - center) * 1e3,
            s=20 + 0.00035 * weights,
            color=color,
            alpha=0.78,
            label=label,
        )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.55)
    ax.set_xlabel("Pass mean LFM shift from polynomial range-rate (m)")
    ax.set_ylabel("Pass mean offset minus weighted mean (m)")
    ax.set_title("High-SNR satellite passes")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    rms = []
    labels = []
    colors = []
    for key, label, color in variants:
        stats = summarize_groups(group_summaries, key)
        rms.append(stats["snr_weighted_pass_mean_rms_m"])
        labels.append(label)
        colors.append(color)
    ax.bar(np.arange(len(labels)), rms, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("SNR-weighted RMS of pass mean offsets (m)")
    ax.set_title("Lower is more consistent")
    fig.savefig(path, dpi=220)


def main() -> None:
    args = parse_args()
    raw_path = os.path.join(args.results_dir, "sanya_satellite_detection_matches_raw.csv")
    raw_rows = read_csv(raw_path)
    corrected_rows, group_summaries = build_high_snr_passes(raw_rows, args)
    if not corrected_rows:
        raise RuntimeError("No high-SNR satellite pulses survived the selection.")

    variants = [
        "offset_no_lfm_km",
        "offset_plus_observed_rate_km",
        "offset_minus_observed_rate_km",
        "offset_plus_predicted_rate_km",
        "offset_minus_predicted_rate_km",
    ]
    pulse_summary = {key: summarize_variant(corrected_rows, key) for key in variants}
    pass_summary = {key: summarize_groups(group_summaries, key) for key in variants}
    best_observed_rate = min(
        ("offset_plus_observed_rate_km", "offset_minus_observed_rate_km"),
        key=lambda key: pass_summary[key]["snr_weighted_pass_mean_rms_m"],
    )

    output = {
        "input_raw_csv": raw_path,
        "selection": {
            "min_snr_db": args.min_snr_db,
            "max_beam_angle_deg": args.max_beam_angle_deg,
            "target_offset_km": args.target_offset_km,
            "offset_half_width_km": args.offset_half_width_km,
            "min_pulses": args.min_pulses,
            "poly_degree": args.poly_degree,
            "snr_weight": "10**(snr_db/10)",
        },
        "lfm": {
            "radar_frequency_hz": RADAR_FREQUENCY_HZ,
            "bandwidth_hz": BANDWIDTH_HZ,
            "duration_s": LFM_DURATION_S,
            "chirp_rate_hz_per_s": CHIRP_RATE_HZ_PER_S,
            "range_shift_from_rate_km": "-(f0/gamma) * polynomial_range_rate_km_s",
        },
        "n_selected_pulses": len(corrected_rows),
        "n_selected_passes": len(group_summaries),
        "pulse_summary": pulse_summary,
        "pass_mean_summary": pass_summary,
        "best_observed_rate_sign": best_observed_rate,
        "best_observed_rate_sign_label": {
            "offset_plus_observed_rate_km": "observed - (predicted + shift)",
            "offset_minus_observed_rate_km": "observed - (predicted - shift)",
        }[best_observed_rate],
        "top_passes_by_snr_weight": sorted(
            group_summaries,
            key=lambda row: row["total_snr_weight"],
            reverse=True,
        )[:20],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    raw_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_raw.csv")
    group_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_passes.csv")
    summary_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_summary.json")
    plot_out = os.path.join(args.output_dir, "satellite_lfm_doppler_sign_highsnr_consistency.png")
    write_csv(raw_out, corrected_rows, list(corrected_rows[0].keys()))
    write_csv(group_out, group_summaries, list(group_summaries[0].keys()))
    make_plot(plot_out, group_summaries)
    with open(summary_out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
