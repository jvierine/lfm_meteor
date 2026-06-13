import glob
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
DAN_CENTER_US = 438.426
WEN_CENTER_US = 430.906
MAX_LAT_DEG = 18.7
DELTA_US = 200.0
STEP_US = 20.0
OUTPUT_H5 = os.path.join("results", "delay_grid_search_beam_axis.h5")
OUTPUT_PNG = os.path.join("results", "delay_grid_search_beam_axis.png")


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
    times_ns_are_utc: bool
    source_timezone_offset_hours: float


@dataclass
class Trajectory:
    event_id: str
    san_az_deg: float
    san_el_deg: float
    san_ranges_km: np.ndarray
    dan_gates: np.ndarray
    dan_sr_mhz: float
    wen_gates: np.ndarray
    wen_sr_mhz: float


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return value


def load_event(path):
    with h5py.File(path, "r") as h:
        times_ns = h["times_ns"][()].astype(np.int64)
        times_ns_are_utc = str(decode_scalar(h.attrs.get("times_ns_time_scale", ""))).upper() == "UTC"
        source_timezone_offset_hours = float(h.attrs.get("source_timezone_offset_hours", 8.0))
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
            times_ns_are_utc=times_ns_are_utc,
            source_timezone_offset_hours=source_timezone_offset_hours,
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


def range_gates_to_km(range_gate, r0_km, sr_mhz):
    dr_km = C / (2.0 * sr_mhz * 1e6) / 1e3
    return r0_km + dr_km * np.asarray(range_gate, dtype=np.float64)


def gate_to_delay_us(range_gate, sr_mhz):
    return np.asarray(range_gate, dtype=np.float64) / float(sr_mhz)


def delay_us_to_range_km(delay_us):
    return 0.5 * np.asarray(delay_us, dtype=np.float64) * 1e-6 * C / 1e3


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
    dan_target_m = 2.0 * dan_equiv_m - san_range_m
    wen_target_m = 2.0 * wen_equiv_m - san_range_m

    fast = solve_three_spheres(
        p_san,
        p_dan,
        p_wen,
        san_range_m,
        dan_target_m,
        wen_target_m,
        x0,
    )
    if fast is not None:
        return fast

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


def solve_three_spheres(p1, p2, p3, r1, r2, r3, x0):
    ex = p2 - p1
    d = np.linalg.norm(ex)
    if not np.isfinite(d) or d <= 0.0:
        return None
    ex = ex / d
    p3p1 = p3 - p1
    i = float(np.dot(ex, p3p1))
    ey0 = p3p1 - i * ex
    j = np.linalg.norm(ey0)
    if not np.isfinite(j) or j <= 0.0:
        return None
    ey = ey0 / j
    ez = np.cross(ex, ey)

    x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    y = (r1 * r1 - r3 * r3 + i * i + j * j - 2.0 * i * x) / (2.0 * j)
    z2 = r1 * r1 - x * x - y * y
    if not np.isfinite(z2):
        return None
    if z2 < -1.0:
        return None
    z = np.sqrt(max(z2, 0.0))
    a = p1 + x * ex + y * ey + z * ez
    b = p1 + x * ex + y * ey - z * ez
    return a if np.linalg.norm(a - x0) <= np.linalg.norm(b - x0) else b


def build_trajectories():
    san_events = load_events(SAN_PATTERN)
    dan_events = load_events(DAN_PATTERN)
    wen_events = load_events(WEN_PATTERN)
    triplets = pair_tristatic_events(san_events, dan_events, wen_events)
    trajectories = []
    for idx, (san_event, dan_event, wen_event) in enumerate(triplets):
        matches = match_pulses(san_event, dan_event, wen_event)
        if len(matches) < 3:
            continue
        san_ranges_all = (
            range_gates_to_km(san_event.range_gate, san_event.r0_km, san_event.sr_mhz)
            + sc.SANYA_RANGE_CORRECTION_KM
        )
        san_ranges = []
        dan_gates = []
        wen_gates = []
        for san_idx, dan_idx, wen_idx in matches:
            san_ranges.append(float(san_ranges_all[san_idx]))
            dan_gates.append(int(dan_event.range_gate[dan_idx]))
            wen_gates.append(int(wen_event.range_gate[wen_idx]))
        trajectories.append(
            Trajectory(
                event_id=f"tri_{idx:04d}_{san_event.t0_ns}",
                san_az_deg=san_event.az_deg,
                san_el_deg=san_event.el_deg,
                san_ranges_km=np.asarray(san_ranges, dtype=np.float64),
                dan_gates=np.asarray(dan_gates, dtype=np.int32),
                dan_sr_mhz=float(dan_event.sr_mhz),
                wen_gates=np.asarray(wen_gates, dtype=np.int32),
                wen_sr_mhz=float(wen_event.sr_mhz),
            )
        )
    return trajectories


def beam_axis():
    origin = np.asarray(jcoord.geodetic2ecef(sc.lat0[0], sc.lon0[0], sc.alt0[0] * 1e3), dtype=np.float64)
    llh = jcoord.az_el_r2geodetic(sc.lat0[0], sc.lon0[0], sc.alt0[0] * 1e3, 15.0, 75.0, 150e3)
    point = np.asarray(jcoord.geodetic2ecef(llh[0], llh[1], llh[2]), dtype=np.float64)
    direction = point - origin
    direction = direction / np.linalg.norm(direction)
    return origin, direction


def solve_trajectory_points(traj, dan_delay0_us, wen_delay0_us):
    dan_ranges_km = delay_us_to_range_km(dan_delay0_us + gate_to_delay_us(traj.dan_gates, traj.dan_sr_mhz))
    wen_ranges_km = delay_us_to_range_km(wen_delay0_us + gate_to_delay_us(traj.wen_gates, traj.wen_sr_mhz))
    x0 = initial_guess(traj.san_az_deg, traj.san_el_deg, float(np.median(traj.san_ranges_km)))
    points = []
    for san_range_km, dan_range_km, wen_range_km in zip(traj.san_ranges_km, dan_ranges_km, wen_ranges_km):
        xhat = solve_position(float(san_range_km), float(dan_range_km), float(wen_range_km), x0)
        x0 = xhat
        points.append(xhat)
    return np.asarray(points, dtype=np.float64)


def score_grid(trajectories, axis_origin, axis_direction, dan_delay0_us, wen_delay0_us):
    total_km = 0.0
    all_alt_km = []
    for traj in trajectories:
        points = solve_trajectory_points(traj, dan_delay0_us, wen_delay0_us)
        llh = np.asarray([jcoord.ecef2geodetic(p[0], p[1], p[2]) for p in points], dtype=np.float64)
        alt_km = llh[:, 2] / 1e3
        keep = np.isfinite(llh[:, 0]) & np.isfinite(llh[:, 1]) & np.isfinite(alt_km) & (llh[:, 0] <= MAX_LAT_DEG)
        if not np.any(keep):
            continue
        rel = points[keep] - axis_origin
        distances_m = np.linalg.norm(np.cross(rel, axis_direction), axis=1)
        total_km += float(np.sum(distances_m) / 1e3)
        all_alt_km.extend(alt_km[keep].tolist())
    all_alt_km = np.asarray(all_alt_km, dtype=np.float64)
    return total_km, float(np.mean(all_alt_km)), float(np.median(all_alt_km))


def main():
    trajectories = build_trajectories()
    axis_origin, axis_direction = beam_axis()
    dan_grid = np.arange(DAN_CENTER_US - DELTA_US, DAN_CENTER_US + DELTA_US + 0.5 * STEP_US, STEP_US)
    wen_grid = np.arange(WEN_CENTER_US - DELTA_US, WEN_CENTER_US + DELTA_US + 0.5 * STEP_US, STEP_US)

    score_grid_km = np.zeros((len(dan_grid), len(wen_grid)), dtype=np.float64)
    mean_alt_grid_km = np.zeros_like(score_grid_km)
    median_alt_grid_km = np.zeros_like(score_grid_km)

    best = None
    for i, dan_delay0_us in enumerate(dan_grid):
        for j, wen_delay0_us in enumerate(wen_grid):
            score_km, mean_alt_km, median_alt_km = score_grid(
                trajectories, axis_origin, axis_direction, float(dan_delay0_us), float(wen_delay0_us)
            )
            score_grid_km[i, j] = score_km
            mean_alt_grid_km[i, j] = mean_alt_km
            median_alt_grid_km[i, j] = median_alt_km
            if best is None or score_km < best["score_km"]:
                best = {
                    "dan_delay0_us": float(dan_delay0_us),
                    "wen_delay0_us": float(wen_delay0_us),
                    "score_km": float(score_km),
                    "mean_alt_km": float(mean_alt_km),
                    "median_alt_km": float(median_alt_km),
                }
            print(
                f"D_dan={dan_delay0_us:7.3f} us D_wen={wen_delay0_us:7.3f} us "
                f"axis_score={score_km:10.3f} km mean_alt={mean_alt_km:7.3f} km median_alt={median_alt_km:7.3f} km"
            )

    with h5py.File(OUTPUT_H5, "w") as h:
        h["dan_delay_grid_us"] = dan_grid
        h["wen_delay_grid_us"] = wen_grid
        h["score_grid_km"] = score_grid_km
        h["mean_alt_grid_km"] = mean_alt_grid_km
        h["median_alt_grid_km"] = median_alt_grid_km
        h.attrs["sanya_range_correction_km"] = sc.SANYA_RANGE_CORRECTION_KM
        h.attrs["sanya_range_correction_sign"] = "san_range_km = raw_range_km + SANYA_RANGE_CORRECTION_KM"

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(wen_grid, dan_grid, score_grid_km, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Sum of distances to beam axis (km)")
    ax.scatter([best["wen_delay0_us"]], [best["dan_delay0_us"]], c="red", marker="x", s=100)
    ax.set_xlabel("Wenchang first-sample delay (us)")
    ax.set_ylabel("Danzhou first-sample delay (us)")
    ax.set_title("Tri-static Beam-Axis Grid Search")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)

    print("")
    print(f"Trajectories used: {len(trajectories)}")
    print(f"Best Danzhou first-sample delay: {best['dan_delay0_us']:.3f} us")
    print(f"Best Wenchang first-sample delay: {best['wen_delay0_us']:.3f} us")
    print(f"Best beam-axis score: {best['score_km']:.3f} km")
    print(f"Best mean altitude: {best['mean_alt_km']:.3f} km")
    print(f"Best median altitude: {best['median_alt_km']:.3f} km")
    print(OUTPUT_H5)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
