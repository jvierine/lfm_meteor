import glob
import math
import os
from dataclasses import dataclass

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so

import sanya_opts as sc


C = 299792458.0
SAN_PATTERN = os.path.join("results", "tristatic_head_echoes", "sanya", "sanya_*.h5")
DAN_PATTERN = os.path.join("results", "tristatic_head_echoes", "danzhou", "danzhou_*.h5")
WEN_PATTERN = os.path.join("results", "tristatic_head_echoes", "wenchang", "wenchang_*.h5")
OUTPUT_PNG = os.path.join("results", "delay_fit_beam_center_100km.png")
TARGET_ALT_M = 100e3
SANYA_DELAY_US = 466.32
INITIAL_DELAYS_US = np.array([713.07, 843.83], dtype=np.float64)


@dataclass
class Event:
    path: str
    site: str
    times_ns: np.ndarray
    range_gate: np.ndarray
    r0_km: float
    sr_mhz: float
    az_deg: float
    el_deg: float
    t0_ns: int
    t1_ns: int


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return value


def load_event(path):
    with h5py.File(path, "r") as h:
        times_ns = h["times_ns"][()].astype(np.int64)
        echoes = h["echoes"][()]
        if "range_gate" in h:
            range_gate = h["range_gate"][()].astype(np.int32)
        else:
            range_gate = np.argmax(np.abs(echoes), axis=1).astype(np.int32)
        return Event(
            path=path,
            site=str(decode_scalar(h["site"][()])).lower(),
            times_ns=times_ns,
            range_gate=range_gate,
            r0_km=float(h["r0"][()]),
            sr_mhz=float(h["sr_mhz"][()]) if "sr_mhz" in h else 4.0,
            az_deg=float(h["az"][()]),
            el_deg=float(h["el"][()]),
            t0_ns=int(times_ns.min()),
            t1_ns=int(times_ns.max()),
        )


def load_events(pattern):
    return [load_event(path) for path in sorted(glob.glob(pattern))]


def overlap_ns(a, b):
    return max(0, min(a.t1_ns, b.t1_ns) - max(a.t0_ns, b.t0_ns))


def best_overlap(event, candidates):
    best = None
    best_overlap_ns = 0
    for candidate in candidates:
        shared = overlap_ns(event, candidate)
        if shared > best_overlap_ns:
            best_overlap_ns = shared
            best = candidate
    return best


def nearest_index(times_ns, t_ns):
    idx = int(np.searchsorted(times_ns, t_ns))
    options = []
    if idx < len(times_ns):
        options.append(idx)
    if idx > 0:
        options.append(idx - 1)
    if not options:
        return None
    return min(options, key=lambda i: abs(int(times_ns[i]) - int(t_ns)))


def pair_tristatic_events(san_events, dan_events, wen_events):
    triplets = []
    for san_event in san_events:
        dan_event = best_overlap(san_event, dan_events)
        wen_event = best_overlap(san_event, wen_events)
        if dan_event is None or wen_event is None:
            continue
        if overlap_ns(san_event, dan_event) == 0 or overlap_ns(san_event, wen_event) == 0:
            continue
        triplets.append((san_event, dan_event, wen_event))
    return triplets


def match_pulses(san_event, dan_event, wen_event, tolerance_ms=7.5):
    tolerance_ns = int(tolerance_ms * 1e6)
    matches = []
    for san_idx, san_t in enumerate(san_event.times_ns):
        dan_idx = nearest_index(dan_event.times_ns, san_t)
        wen_idx = nearest_index(wen_event.times_ns, san_t)
        if dan_idx is None or wen_idx is None:
            continue
        dan_t = int(dan_event.times_ns[dan_idx])
        wen_t = int(wen_event.times_ns[wen_idx])
        if abs(dan_t - int(san_t)) > tolerance_ns:
            continue
        if abs(wen_t - int(san_t)) > tolerance_ns:
            continue
        matches.append((san_idx, dan_idx, wen_idx))
    return matches


def delay_us_to_range_km(delay_us):
    return 0.5 * np.asarray(delay_us, dtype=np.float64) * 1e-6 * C / 1e3


def gate_to_delay_us(range_gate, sr_mhz):
    return np.asarray(range_gate, dtype=np.float64) / float(sr_mhz)


def range_gates_to_km(range_gate, r0_km, sr_mhz):
    dr_km = C / (2.0 * sr_mhz * 1e6) / 1e3
    return r0_km + dr_km * np.asarray(range_gate, dtype=np.float64)


def beam_center_target():
    def alt_error(range_km):
        llh = jcoord.az_el_r2geodetic(
            sc.lat0[0], sc.lon0[0], sc.alt0[0] * 1e3, 15.0, 75.0, range_km * 1e3
        )
        return llh[2] - TARGET_ALT_M

    result = so.root_scalar(alt_error, bracket=[50.0, 200.0], method="brentq")
    llh = jcoord.az_el_r2geodetic(
        sc.lat0[0], sc.lon0[0], sc.alt0[0] * 1e3, 15.0, 75.0, result.root * 1e3
    )
    ecef = np.asarray(jcoord.geodetic2ecef(llh[0], llh[1], llh[2]), dtype=np.float64)
    return result.root, llh, ecef


def initial_guess(az_deg, el_deg, range_km):
    llh = jcoord.az_el_r2geodetic(
        sc.lat0[0], sc.lon0[0], sc.alt0[0] * 1e3, az_deg, el_deg, range_km * 1e3
    )
    return np.asarray(jcoord.geodetic2ecef(llh[0], llh[1], llh[2]), dtype=np.float64)


def solve_position(san_range_km, dan_range_km, wen_range_km, x0):
    p_san = np.asarray(sc.p_san, dtype=np.float64)
    p_dan = np.asarray(sc.p_dan, dtype=np.float64)
    p_wen = np.asarray(sc.p_wen, dtype=np.float64)
    san_range_m = san_range_km * 1e3
    dan_equiv_m = dan_range_km * 1e3
    wen_equiv_m = wen_range_km * 1e3

    def residual(x):
        r_san = np.linalg.norm(x - p_san)
        r_dan = np.linalg.norm(x - p_dan)
        r_wen = np.linalg.norm(x - p_wen)
        return np.array(
            [
                r_san - san_range_m,
                0.5 * (r_san + r_dan) - dan_equiv_m,
                0.5 * (r_san + r_wen) - wen_equiv_m,
            ],
            dtype=np.float64,
        )

    return so.least_squares(residual, x0=x0, method="lm").x


def representative_measurements(triplets):
    reps = []
    for idx, (san_event, dan_event, wen_event) in enumerate(triplets):
        matches = match_pulses(san_event, dan_event, wen_event)
        if not matches:
            continue
        san_idx, dan_idx, wen_idx = matches[len(matches) // 2]
        san_ranges_km = range_gates_to_km(san_event.range_gate, san_event.r0_km, san_event.sr_mhz)
        reps.append(
            {
                "event_id": f"tri_{idx:04d}_{san_event.t0_ns}",
                "san_az_deg": san_event.az_deg,
                "san_el_deg": san_event.el_deg,
                "san_range_km": float(san_ranges_km[san_idx]),
                "dan_gate": int(dan_event.range_gate[dan_idx]),
                "dan_sr_mhz": float(dan_event.sr_mhz),
                "wen_gate": int(wen_event.range_gate[wen_idx]),
                "wen_sr_mhz": float(wen_event.sr_mhz),
            }
        )
    return reps


def solve_representatives(reps, dan_delay0_us, wen_delay0_us):
    points = []
    for rep in reps:
        dan_range_km = float(delay_us_to_range_km(dan_delay0_us + gate_to_delay_us(rep["dan_gate"], rep["dan_sr_mhz"])))
        wen_range_km = float(delay_us_to_range_km(wen_delay0_us + gate_to_delay_us(rep["wen_gate"], rep["wen_sr_mhz"])))
        x0 = initial_guess(rep["san_az_deg"], rep["san_el_deg"], rep["san_range_km"])
        xhat = solve_position(rep["san_range_km"], dan_range_km, wen_range_km, x0)
        points.append(xhat)
    return np.asarray(points, dtype=np.float64)


def objective(delays_us, reps, target_ecef):
    dan_delay0_us, wen_delay0_us = delays_us
    points = solve_representatives(reps, dan_delay0_us, wen_delay0_us)
    mean_point = points.mean(axis=0)
    return np.linalg.norm(mean_point - target_ecef) / 1e3


def plot_points(points_ecef, target_llh):
    llh = np.asarray([jcoord.ecef2geodetic(p[0], p[1], p[2]) for p in points_ecef], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 6))
    sca = ax.scatter(llh[:, 1], llh[:, 0], c=llh[:, 2] / 1e3, s=18, alpha=0.8)
    ax.scatter([sc.lon0[0], sc.lon0[1], sc.lon0[2]], [sc.lat0[0], sc.lat0[1], sc.lat0[2]], c="red", marker="^", s=70)
    ax.scatter([target_llh[1]], [target_llh[0]], c="black", marker="x", s=100)
    ax.text(sc.lon0[0] + 0.03, sc.lat0[0] + 0.03, "Sanya")
    ax.text(sc.lon0[1] + 0.03, sc.lat0[1] + 0.03, "Danzhou")
    ax.text(sc.lon0[2] + 0.03, sc.lat0[2] + 0.03, "Wenchang")
    ax.text(target_llh[1] + 0.03, target_llh[0] + 0.03, "Beam center @ 100 km")
    cb = fig.colorbar(sca, ax=ax)
    cb.set_label("Altitude (km)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Representative Tri-static Solutions with Delay Fit to Beam Center")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)
    return llh


def main():
    san_events = load_events(SAN_PATTERN)
    dan_events = load_events(DAN_PATTERN)
    wen_events = load_events(WEN_PATTERN)
    triplets = pair_tristatic_events(san_events, dan_events, wen_events)
    reps = representative_measurements(triplets)
    beam_range_km, beam_llh, beam_ecef = beam_center_target()

    result = so.minimize(
        objective,
        x0=INITIAL_DELAYS_US,
        args=(reps, beam_ecef),
        method="Nelder-Mead",
        options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-3},
    )

    best_dan_us, best_wen_us = result.x
    points_ecef = solve_representatives(reps, best_dan_us, best_wen_us)
    llh = plot_points(points_ecef, beam_llh)
    mean_llh = np.asarray(jcoord.ecef2geodetic(points_ecef[:, 0].mean(), points_ecef[:, 1].mean(), points_ecef[:, 2].mean()))

    print(f"Representative trajectories: {len(reps)}")
    print(f"Beam center slant range for 100 km altitude: {beam_range_km:.3f} km")
    print(f"Best Danzhou first-sample delay: {best_dan_us:.3f} us")
    print(f"Best Wenchang first-sample delay: {best_wen_us:.3f} us")
    print(f"Mean solved lat/lon/alt: {mean_llh[0]:.5f}, {mean_llh[1]:.5f}, {mean_llh[2]/1e3:.3f} km")
    print(f"Target beam-center lat/lon/alt: {beam_llh[0]:.5f}, {beam_llh[1]:.5f}, {beam_llh[2]/1e3:.3f} km")
    print(f"Mean-point to beam-center distance: {objective(result.x, reps, beam_ecef):.3f} km")
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
