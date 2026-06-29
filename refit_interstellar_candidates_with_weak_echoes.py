"""Refit interstellar candidates using prior-guided weak-echo measurements."""

from __future__ import annotations

import csv
import os

import h5py
import jcoord
import numpy as np

import fit_all_ballistic_snr_weighted as fitmod
import recover_interstellar_candidate_weak_echoes as weak


INPUT_CSV = "results/interstellar_candidate_weak_echo_recovery.csv"
OUTPUT_CSV = "results/interstellar_candidate_weak_echo_refit_summary.csv"
MIN_SNR_DB = 8.0
MAX_ABS_PATH_RESID_M = 150.0
MATCH_TOLERANCE_NS = 3_500_000


def original_fit(event_id: str) -> dict:
    with h5py.File(weak.FIT_H5, "r") as h:
        g = h["points"][event_id]
        return {
            "params": g["params"][:].astype(np.float64),
            "time_ns": g["time_ns"][:].astype(np.int64),
            "x_itrs_m": g["x_itrs_m"][:].astype(np.float64),
            "measured_total_paths_m": g["measured_total_paths_m"][:].astype(np.float64),
            "snr_db": g["snr_db"][:].astype(np.float64),
            "rms_total_path_residual_m": float(np.sqrt(np.mean(g["residuals_m"][:] ** 2.0))),
            "sigma_floor_m": float(h.attrs["sigma_floor_m"]),
            "sigma_0_m": float(h.attrs["sigma_0_m"]),
        }


def match_triples(rows: np.ndarray) -> list[dict]:
    by = {}
    for site in weak.SITE_ORDER:
        site_rows = rows[rows["site"] == site]
        site_rows = np.sort(site_rows, order="time_ns")
        by[site] = site_rows
    triples = []
    for sr in by["sanya"]:
        triple = {"sanya": sr}
        ok = True
        for site in ("danzhou", "wenchang"):
            times = by[site]["time_ns"].astype(np.int64)
            j = int(np.argmin(np.abs(times - int(sr["time_ns"]))))
            if abs(int(times[j]) - int(sr["time_ns"])) > MATCH_TOLERANCE_NS:
                ok = False
                break
            triple[site] = by[site][j]
        if ok:
            triples.append(triple)
    return triples


def selected_measurements(event_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw = [row for row in reader if row["event_id"] == event_id]

    parsed = []
    for row in raw:
        parsed.append(
            {
                **row,
                "time_ns": int(row["time_ns"]),
                "recovered_total_path_m": float(row["recovered_total_path_m"]),
                "prior_window_snr_db": float(row["prior_window_snr_db"]),
                "path_residual_m": float(row["path_residual_m"]),
                "original_detected": row["original_detected"] == "True",
            }
        )
    rows_by_site = {site: sorted([r for r in parsed if r["site"] == site], key=lambda r: r["time_ns"]) for site in weak.SITE_ORDER}
    triples = []
    for sr in rows_by_site["sanya"]:
        triple = {"sanya": sr}
        ok = True
        for site in ("danzhou", "wenchang"):
            candidates = rows_by_site[site]
            j = min(range(len(candidates)), key=lambda ii: abs(candidates[ii]["time_ns"] - sr["time_ns"]))
            if abs(candidates[j]["time_ns"] - sr["time_ns"]) > MATCH_TOLERANCE_NS:
                ok = False
                break
            triple[site] = candidates[j]
        if not ok:
            continue
        if min(triple[site]["prior_window_snr_db"] for site in weak.SITE_ORDER) < MIN_SNR_DB:
            continue
        if max(abs(triple[site]["path_residual_m"]) for site in weak.SITE_ORDER) > MAX_ABS_PATH_RESID_M:
            continue
        triples.append(triple)

    times = []
    measured = []
    snr = []
    original_mask = []
    for triple in triples:
        times.append(int(round(np.mean([triple[site]["time_ns"] for site in weak.SITE_ORDER]))))
        measured.append([triple[site]["recovered_total_path_m"] for site in weak.SITE_ORDER])
        snr.append([triple[site]["prior_window_snr_db"] for site in weak.SITE_ORDER])
        original_mask.append(all(triple[site]["original_detected"] for site in weak.SITE_ORDER))
    order = np.argsort(times)
    return (
        np.asarray(times, dtype=np.int64)[order],
        np.asarray(measured, dtype=np.float64)[order],
        np.asarray(snr, dtype=np.float64)[order],
        np.asarray(original_mask, dtype=bool)[order],
    )


def shifted_initial_params(fit: dict, first_time_ns: int, rho) -> np.ndarray:
    t_rel = np.asarray([(float(first_time_ns) - float(fit["time_ns"][0])) / 1e9], dtype=np.float64)
    x, v = weak.propagate_params_any(fit["params"], t_rel, rho)
    return np.concatenate([x[0], v[0], [fit["params"][6]]])


def fit_event(event_id: str) -> dict:
    fit = original_fit(event_id)
    rho = weak.rho_from_fit({"time_ns": fit["time_ns"], "x_itrs_m": fit["x_itrs_m"]})
    times, measured, snr, original_mask = selected_measurements(event_id)
    p0 = shifted_initial_params(fit, int(times[0]), rho)
    sigma = fitmod.sigma_from_snr_db(snr, fit["sigma_floor_m"], fit["sigma_0_m"])
    result = fitmod.fit_ballistic(measured, times, rho, p0, sigma_m=sigma, loss="linear")
    llh = np.asarray([jcoord.ecef2geodetic(*x) for x in result["x_itrs_m"]])
    return {
        "event_id": event_id,
        "n_original_fit_points": int(len(fit["time_ns"])),
        "n_recovered_refit_points": int(len(times)),
        "n_refit_points_from_original_detections": int(np.sum(original_mask)),
        "n_refit_points_from_weak_recovery": int(len(times) - np.sum(original_mask)),
        "t_start_ms_vs_original": float((times[0] - fit["time_ns"][0]) / 1e6),
        "t_end_ms_vs_original": float((times[-1] - fit["time_ns"][0]) / 1e6),
        "original_rms_m": float(fit["rms_total_path_residual_m"]),
        "recovered_refit_rms_m": float(np.sqrt(np.mean(result["residuals_m"] ** 2.0))),
        "start_speed_km_s": float(result["speed_km_s"][0]),
        "end_speed_km_s": float(result["speed_km_s"][-1]),
        "start_alt_km": float(llh[0, 2] / 1e3),
        "end_alt_km": float(llh[-1, 2] / 1e3),
        "log10_cd_a_over_m": float(result["params"][6]),
        "covariance_available": bool(result["covariance_available"]),
    }


def main() -> None:
    rows = [fit_event(event_id) for event_id in weak.CANDIDATES]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
