"""Prior-guided weak-echo recovery for the interstellar candidate events.

The script uses the fitted tri-static trajectory as a prior, matched-filters
the compact raw-voltage cuts for every pulse in the cut, and searches a small
range-gate window around the predicted path for weaker echoes before, inside,
and after the original detections.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import fit_all_ballistic_snr_weighted as fitmod
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import sanya_opts as sc
import test_rank02_range_interpolation as interp


FIT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
RAW_CUT_ROOT = "results/interstellar_candidate_raw_cuts"
OUTPUT_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures/interstellar_candidate_weak_echo_recovery"
CSV_PATH = "results/interstellar_candidate_weak_echo_recovery.csv"
SEARCH_HALF_WIDTH_GATES = 10
UPSAMPLE_FACTOR = 4
C_MPS = 299792458.0
RADAR_WAVELENGTH_M = gfit.RADAR_WAVELENGTH_M

SITE_ORDER = ("sanya", "danzhou", "wenchang")
SITE_LABELS = {"sanya": "Sanya", "danzhou": "Danzhou", "wenchang": "Wenchang"}
SITE_INDEX = {"sanya": 0, "danzhou": 1, "wenchang": 2}

CANDIDATES = {
    "tri_0108_1713818804364349365": {
        "local_rti_id": "tri_0108_1713847604364349365",
        "cuts": {
            "sanya": "sanya/sanya_1713847604364349365.h5",
            "danzhou": "danzhou/danzhou_1713847604374351501.h5",
            "wenchang": "wenchang/wenchang_1713847604379348755.h5",
        },
    },
    "tri_0200_1713826337469350815": {
        "local_rti_id": "tri_0200_1713855137469350815",
        "cuts": {
            "sanya": "sanya/sanya_1713855137469350815.h5",
            "danzhou": "danzhou/danzhou_1713855137469350815.h5",
            "wenchang": "wenchang/wenchang_1713855137474349976.h5",
        },
    },
}


@dataclass
class SiteCut:
    path: str
    raw_voltage: np.ndarray
    times_ns: np.ndarray
    range_gate_index: np.ndarray
    range_km: np.ndarray
    science_range_gate_index: np.ndarray
    sr_mhz: float
    bw_mhz: float
    pulse_length_us: float
    detections: dict[str, np.ndarray]


def read_cut(path: str) -> SiteCut:
    with h5py.File(path, "r") as h:
        detections = {key: h["detections"][key][:] for key in h["detections"].keys()}
        return SiteCut(
            path=path,
            raw_voltage=h["raw_voltage"][:].astype(np.complex64),
            times_ns=h["time_ns_utc"][:].astype(np.int64),
            range_gate_index=h["range_gate_index"][:].astype(np.int64),
            range_km=h["range_km"][:].astype(np.float64),
            science_range_gate_index=h["science_range_gate_index"][:].astype(np.int64),
            sr_mhz=float(h["lfm_sample_rate_mhz"][()]),
            bw_mhz=float(h["lfm_bandwidth_mhz"][()]),
            pulse_length_us=float(h["lfm_pulse_length_us"][()]),
            detections=detections,
        )


def load_fit(event_id: str) -> dict[str, np.ndarray | float | int]:
    with h5py.File(FIT_H5, "r") as h:
        group = h["points"][event_id]
        return {
            "event_id": event_id,
            "params": group["params"][:].astype(np.float64),
            "time_ns": group["time_ns"][:].astype(np.int64),
            "x_itrs_m": group["x_itrs_m"][:].astype(np.float64),
            "rms_total_path_residual_m": float(h["rms_total_path_residual_m"][np.where(h["event_id"][:] == event_id.encode())[0][0]])
            if event_id.encode() in h["event_id"][:]
            else float(np.sqrt(np.mean(group["residuals_m"][:] ** 2))),
        }


def rho_from_fit(fit: dict):
    try:
        rho, _meta = fitmod.density_interpolator(fit["time_ns"], fit["x_itrs_m"])
        return rho
    except Exception:
        def fallback(alt_m):
            alt_km = np.clip(np.asarray(alt_m, dtype=np.float64) / 1e3, 70.0, 160.0)
            return 5.0e-7 * np.exp(-(alt_km - 100.0) / 7.0)

        return fallback


def propagate_params_any(params: np.ndarray, t_rel_s: np.ndarray, rho_of_alt_m) -> tuple[np.ndarray, np.ndarray]:
    b_drag = float(np.clip(10.0 ** params[6], fitmod.MIN_B, fitmod.MAX_B))
    targets = np.asarray(t_rel_s, dtype=np.float64)
    out_pos = np.zeros((len(targets), 3), dtype=np.float64)
    out_vel = np.zeros((len(targets), 3), dtype=np.float64)

    for direction in (-1, 1):
        idx = np.flatnonzero(targets * direction >= -1e-12)
        if idx.size == 0:
            continue
        order = idx[np.argsort(targets[idx] * direction)]
        state = np.concatenate([params[:3], params[3:6]]).astype(np.float64)
        t_prev = 0.0
        for ii in order:
            target = float(targets[ii])
            while abs(target - t_prev) > 1e-12:
                step = np.clip(target - t_prev, -0.002, 0.002)
                state = fitmod.rk4_step(state, float(step), b_drag, rho_of_alt_m)
                t_prev += float(step)
            out_pos[ii] = state[:3]
            out_vel[ii] = state[3:6]
    return out_pos, out_vel


def predict_for_times(fit: dict, times_ns: np.ndarray, rho_of_alt_m):
    t_rel_s = (times_ns.astype(np.float64) - float(fit["time_ns"][0])) / 1e9
    x_gcrs, v_gcrs = propagate_params_any(fit["params"], t_rel_s, rho_of_alt_m)
    x_itrs, v_itrs = fitmod.gcrs_state_samples_to_itrs(x_gcrs, v_gcrs, times_ns)
    total_paths_m, path_rates_mps = gfit.link_total_paths_and_rates_m(
        x_itrs, v_itrs, gfit.LINK_TX_POSITIONS_M, gfit.LINK_RX_POSITIONS_M
    )
    total_paths_m = total_paths_m + gfit.lfm_total_path_bias_m(path_rates_mps)
    return t_rel_s, total_paths_m, path_rates_mps


def total_path_to_gate(total_path_m: np.ndarray, sr_mhz: float) -> np.ndarray:
    delay_us = np.asarray(total_path_m, dtype=np.float64) / C_MPS * 1e6
    return (delay_us - sc.SANYA_CORRECTED_TXRX_DELAY_US) * float(sr_mhz)


def matched_profile(raw_col: np.ndarray, fd_hz: float, sr_mhz: float, bw_mhz: float) -> np.ndarray:
    code, t_s = interp.lfm(sr_mhz=sr_mhz, bandwidth_hz=bw_mhz * 1e6)
    doppler_code = code * np.exp(1j * 2.0 * np.pi * fd_hz * t_s).astype(np.complex64)
    return sig.fftconvolve(raw_col, np.conj(doppler_code), mode="same")


def local_peak(power: np.ndarray, center: float, half_width: int) -> tuple[float, float]:
    center_i = int(round(center))
    lo = max(0, center_i - half_width)
    hi = min(len(power), center_i + half_width + 1)
    if hi <= lo:
        return float("nan"), float("nan")
    idx = lo + int(np.argmax(power[lo:hi]))
    delta = 0.0
    if 0 < idx < len(power) - 1:
        ym1, y0, yp1 = float(power[idx - 1]), float(power[idx]), float(power[idx + 1])
        denom = ym1 - 2.0 * y0 + yp1
        if denom < 0.0:
            delta = float(np.clip(0.5 * (ym1 - yp1) / denom, -0.5, 0.5))
    return float(idx) + delta, float(power[idx])


def detection_by_pulse(cut: SiteCut) -> dict[int, dict[str, float]]:
    out = {}
    det = cut.detections
    for j, pidx in enumerate(det["pulse_index"].astype(int)):
        out[int(pidx)] = {
            "detected": True,
            "selected": bool(det["selected_for_cut"][j]),
            "detected_gate": float(det["range_gate_index"][j]),
            "detected_snr_db": float(det["snr_peak_db"][j]),
        }
    return out


def analyze_site(event_id: str, site: str, cut: SiteCut, fit: dict, rho_of_alt_m) -> list[dict]:
    t_rel_s, total_paths_m, path_rates_mps = predict_for_times(fit, cut.times_ns, rho_of_alt_m)
    link = SITE_INDEX[site]
    pred_gate = total_path_to_gate(total_paths_m[:, link], cut.sr_mhz)
    pred_local_gate = pred_gate - float(cut.range_gate_index[0])
    fd_hz = -path_rates_mps[:, link] / RADAR_WAVELENGTH_M
    det_map = detection_by_pulse(cut)
    rows = []
    science_mask = np.isin(cut.range_gate_index, cut.science_range_gate_index)
    if not np.any(science_mask):
        science_mask = np.ones_like(cut.range_gate_index, dtype=bool)

    for pulse_idx in range(cut.raw_voltage.shape[1]):
        profile = matched_profile(cut.raw_voltage[:, pulse_idx], float(fd_hz[pulse_idx]), cut.sr_mhz, cut.bw_mhz)
        power = np.abs(profile) ** 2.0
        gate_local, peak_power = local_peak(power, pred_local_gate[pulse_idx], SEARCH_HALF_WIDTH_GATES)
        global_gate = float(cut.range_gate_index[0]) + gate_local
        noise = float(np.nanmedian(power[science_mask]))
        snr_prior_db = 10.0 * np.log10(max(peak_power, 1e-30) / max(noise, 1e-30))
        pred_power = float(np.interp(pred_local_gate[pulse_idx], np.arange(len(power)), power))
        snr_at_pred_db = 10.0 * np.log10(max(pred_power, 1e-30) / max(noise, 1e-30))
        measured_total_path_m = C_MPS * (sc.SANYA_CORRECTED_TXRX_DELAY_US + global_gate / cut.sr_mhz) * 1e-6
        det = det_map.get(pulse_idx, {"detected": False, "selected": False, "detected_gate": np.nan, "detected_snr_db": np.nan})
        rows.append(
            {
                "event_id": event_id,
                "site": site,
                "pulse_index": pulse_idx,
                "time_ns": int(cut.times_ns[pulse_idx]),
                "t_rel_s": float(t_rel_s[pulse_idx]),
                "predicted_total_path_m": float(total_paths_m[pulse_idx, link]),
                "predicted_gate": float(pred_gate[pulse_idx]),
                "recovered_gate": float(global_gate),
                "gate_residual": float(global_gate - pred_gate[pulse_idx]),
                "recovered_total_path_m": float(measured_total_path_m),
                "path_residual_m": float(measured_total_path_m - total_paths_m[pulse_idx, link]),
                "prior_window_snr_db": float(snr_prior_db),
                "predicted_gate_snr_db": float(snr_at_pred_db),
                "doppler_hz": float(fd_hz[pulse_idx]),
                "original_detected": bool(det["detected"]),
                "original_selected": bool(det["selected"]),
                "original_detected_gate": float(det["detected_gate"]),
                "original_detected_snr_db": float(det["detected_snr_db"]),
            }
        )
    return rows


def plot_event(event_id: str, rows: list[dict]) -> tuple[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), sharex=True, constrained_layout=True)
    colors = {"sanya": "#1f77b4", "danzhou": "#d95f02", "wenchang": "#2ca02c"}
    for ax, site in zip(axes, SITE_ORDER):
        site_rows = [r for r in rows if r["site"] == site]
        t_ms = np.asarray([r["t_rel_s"] for r in site_rows]) * 1e3
        snr = np.asarray([r["prior_window_snr_db"] for r in site_rows])
        resid_m = np.asarray([r["path_residual_m"] for r in site_rows])
        detected = np.asarray([r["original_detected"] for r in site_rows], dtype=bool)
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.scatter(t_ms[~detected], resid_m[~detected], c=snr[~detected], cmap="viridis", vmin=0, vmax=35, s=42, marker="s", label="prior-guided weak search")
        sca = ax.scatter(t_ms[detected], resid_m[detected], c=snr[detected], cmap="viridis", vmin=0, vmax=35, s=48, edgecolor="black", linewidth=0.6, label="original detection")
        ax.set_ylabel(f"{SITE_LABELS[site]}\npath resid. (m)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time from first fitted sample (ms)")
    cb = fig.colorbar(sca, ax=axes, pad=0.015, shrink=0.95)
    cb.set_label("Prior-window matched-filter SNR (dB)")
    fig.suptitle(f"Prior-guided weak echo recovery: {event_id}")
    png = os.path.join(OUTPUT_DIR, f"{event_id}_weak_echo_recovery.png")
    pdf = os.path.join(OUTPUT_DIR, f"{event_id}_weak_echo_recovery.pdf")
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows = []
    for event_id, meta in CANDIDATES.items():
        fit = load_fit(event_id)
        rho = rho_from_fit(fit)
        event_rows = []
        for site, relpath in meta["cuts"].items():
            cut = read_cut(os.path.join(RAW_CUT_ROOT, relpath))
            rows = analyze_site(event_id, site, cut, fit, rho)
            event_rows.extend(rows)
        all_rows.extend(event_rows)
        png, pdf = plot_event(event_id, event_rows)
        print(f"wrote {png}")
        print(f"wrote {pdf}")
        for site in SITE_ORDER:
            site_rows = [r for r in event_rows if r["site"] == site]
            n_new = sum((not r["original_detected"]) and r["prior_window_snr_db"] >= 8.0 and abs(r["path_residual_m"]) <= 750.0 for r in site_rows)
            n_det = sum(r["original_detected"] for r in site_rows)
            print(f"{event_id} {site}: pulses={len(site_rows)} original={n_det} weak_candidates_snr8={n_new}")
    write_csv(CSV_PATH, all_rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
