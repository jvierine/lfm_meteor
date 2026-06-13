import argparse
import glob
import os
from dataclasses import dataclass

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as n
import scipy.optimize as so
import stuffr

import sanya_opts as sc


C = 299792458.0
DEFAULT_RESULTS_DIR = os.path.expanduser("~/src/lfm_meteor/results")
DEFAULT_MATCH_TOLERANCE_MS = 7.5
DEFAULT_SAN_PATTERN = os.path.join(DEFAULT_RESULTS_DIR, "head_echoes", "sanya", "sanya_*.h5")
DEFAULT_DAN_PATTERN = os.path.join(DEFAULT_RESULTS_DIR, "head_echoes", "danzhou", "danzhou_*.h5")
DEFAULT_WEN_PATTERN = os.path.join(DEFAULT_RESULTS_DIR, "head_echoes", "wenchang", "wenchang_*.h5")
SITE_FIRST_SAMPLE_DELAY_US = {
    "sanya": 466.32,
    "danzhou": 438.426,
    "wenchang": 430.906,
}


@dataclass
class Event:
    path: str
    site: str
    times_ns: n.ndarray
    range_gate: n.ndarray
    range_km: n.ndarray
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


def range_gates_to_km(range_gate, r0_km, sr_mhz):
    dr_km = C / (2.0 * sr_mhz * 1e6) / 1e3
    return r0_km + dr_km * n.asarray(range_gate, dtype=n.float64)


def range_km_to_delay_us(range_km):
    return 2.0 * n.asarray(range_km, dtype=n.float64) * 1e3 / C * 1e6


def delay_us_to_range_km(delay_us):
    return 0.5 * n.asarray(delay_us, dtype=n.float64) * 1e-6 * C / 1e3


def gate_to_delay_us(range_gate, sr_mhz):
    return n.asarray(range_gate, dtype=n.float64) / float(sr_mhz)


def delay_us_to_first_sample_range_km(delay_us):
    return 0.5 * float(delay_us) * 1e-6 * C / 1e3


def load_event(path):
    with h5py.File(path, "r") as h:
        times_ns = h["times_ns"][()].astype(n.int64)
        echoes = h["echoes"][()]
        r0_km = float(h["r0"][()])
        sr_mhz = float(h["sr_mhz"][()]) if "sr_mhz" in h else 4.0
        az_deg = float(h["az"][()])
        el_deg = float(h["el"][()])
        site = str(decode_scalar(h["site"][()])).lower()

        if "range_gate" in h:
            range_gate = h["range_gate"][()].astype(n.int32)
        else:
            range_gate = n.argmax(n.abs(echoes), axis=1).astype(n.int32)

        range_km = range_gates_to_km(range_gate, r0_km, sr_mhz)

    return Event(
        path=path,
        site=site,
        times_ns=times_ns,
        range_gate=range_gate,
        range_km=range_km,
        r0_km=r0_km,
        sr_mhz=sr_mhz,
        az_deg=az_deg,
        el_deg=el_deg,
        t0_ns=int(times_ns.min()),
        t1_ns=int(times_ns.max()),
    )


def load_events(pattern):
    events = [load_event(path) for path in sorted(glob.glob(pattern))]
    return events


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
    idx = int(n.searchsorted(times_ns, t_ns))
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


def match_pulses(san_event, dan_event, wen_event, tolerance_ms):
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

        matches.append(
            {
                "san_idx": san_idx,
                "dan_idx": dan_idx,
                "wen_idx": wen_idx,
                "time_ns": int(round((int(san_t) + dan_t + wen_t) / 3.0)),
            }
        )
    return matches


def initial_guess(az_deg, el_deg, range_km):
    llh = jcoord.az_el_r2geodetic(
        sc.lat0[0], sc.lon0[0], sc.alt0[0] * 1e3, az_deg, el_deg, range_km * 1e3
    )
    return n.asarray(jcoord.geodetic2ecef(llh[0], llh[1], llh[2]), dtype=n.float64)


def remote_equivalent_ranges_km(remote_event):
    delay0_us = SITE_FIRST_SAMPLE_DELAY_US[remote_event.site]
    delay_us = delay0_us + gate_to_delay_us(remote_event.range_gate, remote_event.sr_mhz)
    return delay_us_to_range_km(delay_us), delay0_us


def solve_position(ranges_km, x0):
    p_san = n.asarray(sc.p_san, dtype=n.float64)
    p_dan = n.asarray(sc.p_dan, dtype=n.float64)
    p_wen = n.asarray(sc.p_wen, dtype=n.float64)
    ranges_m = n.asarray(ranges_km, dtype=n.float64) * 1e3
    san_range_m, dan_equiv_m, wen_equiv_m = ranges_m

    def residual(x):
        r_san = n.linalg.norm(x - p_san)
        r_dan = n.linalg.norm(x - p_dan)
        r_wen = n.linalg.norm(x - p_wen)
        return n.array(
            [
                r_san - san_range_m,
                0.5 * (r_san + r_dan) - dan_equiv_m,
                0.5 * (r_san + r_wen) - wen_equiv_m,
            ],
            dtype=n.float64,
        )

    result = so.least_squares(residual, x0=x0, method="lm")
    llh = jcoord.ecef2geodetic(result.x[0], result.x[1], result.x[2])
    return result.x, llh, result.cost, residual(result.x)


def fit_track(points_ecef, times_ns):
    rel_t = (times_ns - times_ns[0]).astype(n.float64) / 1e9
    coeffs = [n.polyfit(rel_t, points_ecef[:, axis], 1) for axis in range(3)]
    fitted = n.column_stack([n.polyval(coeff, rel_t) for coeff in coeffs])
    velocity_mps = n.array([coeff[0] for coeff in coeffs])
    return rel_t, fitted, velocity_mps


def write_results_h5(path, all_rows, summaries):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        if all_rows:
            h["delay_time_ns"] = n.asarray([row["time_ns"] for row in all_rows], dtype=n.int64)
            h["san_range_km"] = n.asarray([row["san_range_km"] for row in all_rows], dtype=n.float64)
            h["dan_range_km"] = n.asarray([row["dan_range_km"] for row in all_rows], dtype=n.float64)
            h["wen_range_km"] = n.asarray([row["wen_range_km"] for row in all_rows], dtype=n.float64)
            h["san_delay_us"] = n.asarray([row["san_delay_us"] for row in all_rows], dtype=n.float64)
            h["dan_delay_us"] = n.asarray([row["dan_delay_us"] for row in all_rows], dtype=n.float64)
            h["wen_delay_us"] = n.asarray([row["wen_delay_us"] for row in all_rows], dtype=n.float64)
            h["alt_km"] = n.asarray([row["alt_km"] for row in all_rows], dtype=n.float64)
            h["lat_deg"] = n.asarray([row["lat_deg"] for row in all_rows], dtype=n.float64)
            h["lon_deg"] = n.asarray([row["lon_deg"] for row in all_rows], dtype=n.float64)
            h["fit_cost"] = n.asarray([row["fit_cost"] for row in all_rows], dtype=n.float64)
            h["delay_event_id"] = n.asarray([row["event_id"] for row in all_rows], dtype=string_dtype)
            h["delay_utc"] = n.asarray([row["utc"] for row in all_rows], dtype=string_dtype)

        if summaries:
            h["summary_event_id"] = n.asarray([row["event_id"] for row in summaries], dtype=string_dtype)
            h["summary_start_utc"] = n.asarray([row["start_utc"] for row in summaries], dtype=string_dtype)
            h["summary_end_utc"] = n.asarray([row["end_utc"] for row in summaries], dtype=string_dtype)
            h["summary_n_points"] = n.asarray([row["n_points"] for row in summaries], dtype=n.int32)
            h["summary_duration_s"] = n.asarray([row["duration_s"] for row in summaries], dtype=n.float64)
            h["summary_speed_km_s"] = n.asarray([row["speed_km_s"] for row in summaries], dtype=n.float64)
            h["summary_start_alt_km"] = n.asarray([row["start_alt_km"] for row in summaries], dtype=n.float64)
            h["summary_end_alt_km"] = n.asarray([row["end_alt_km"] for row in summaries], dtype=n.float64)
            h["summary_start_lat_deg"] = n.asarray([row["start_lat_deg"] for row in summaries], dtype=n.float64)
            h["summary_start_lon_deg"] = n.asarray([row["start_lon_deg"] for row in summaries], dtype=n.float64)
            h["summary_end_lat_deg"] = n.asarray([row["end_lat_deg"] for row in summaries], dtype=n.float64)
            h["summary_end_lon_deg"] = n.asarray([row["end_lon_deg"] for row in summaries], dtype=n.float64)


def plot_event(path, event_id, rows, rel_t, alt_km, fit_alt_km):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(rel_t, [row["san_delay_us"] for row in rows], ".", label="Sanya")
    axes[0].plot(rel_t, [row["dan_delay_us"] for row in rows], ".", label="Danzhou")
    axes[0].plot(rel_t, [row["wen_delay_us"] for row in rows], ".", label="Wenchang")
    axes[0].set_ylabel("Group delay (us)")
    axes[0].legend()
    axes[0].set_title(event_id)

    axes[1].plot(rel_t, alt_km, ".", label="Solved altitude")
    axes[1].plot(rel_t, fit_alt_km, "-", label="Linear fit")
    axes[1].set_xlabel("Time since event start (s)")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overview(path, all_rows):
    if not all_rows:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sca = ax.scatter(
        [row["time_ns"] / 1e9 for row in all_rows],
        [row["alt_km"] for row in all_rows],
        c=[row["san_delay_us"] for row in all_rows],
        s=12,
    )
    ax.set_xlabel("Unix time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Tri-static delay solutions")
    cb = fig.colorbar(sca, ax=ax)
    cb.set_label("Sanya group delay (us)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def process_triplet(event_id, san_event, dan_event, wen_event, tolerance_ms):
    matches = match_pulses(san_event, dan_event, wen_event, tolerance_ms=tolerance_ms)
    if not matches:
        return [], None

    rows = []
    points = []
    times_ns = []
    x0 = initial_guess(san_event.az_deg, san_event.el_deg, float(n.median(san_event.range_km)))
    dan_equiv_ranges_km, dan_delay0_us = remote_equivalent_ranges_km(dan_event)
    wen_equiv_ranges_km, wen_delay0_us = remote_equivalent_ranges_km(wen_event)

    for match in matches:
        san_range_km = float(san_event.range_km[match["san_idx"]])
        dan_range_km = float(dan_equiv_ranges_km[match["dan_idx"]])
        wen_range_km = float(wen_equiv_ranges_km[match["wen_idx"]])
        xhat, llh, cost, _ = solve_position([san_range_km, dan_range_km, wen_range_km], x0)
        x0 = xhat
        san_delay_us = float(range_km_to_delay_us(san_range_km))
        dan_delay_us = float(dan_delay0_us + gate_to_delay_us(dan_event.range_gate[match["dan_idx"]], dan_event.sr_mhz))
        wen_delay_us = float(wen_delay0_us + gate_to_delay_us(wen_event.range_gate[match["wen_idx"]], wen_event.sr_mhz))

        row = {
            "event_id": event_id,
            "time_ns": match["time_ns"],
            "utc": stuffr.unix2datestr(match["time_ns"] / 1e9),
            "san_range_km": san_range_km,
            "dan_range_km": dan_range_km,
            "wen_range_km": wen_range_km,
            "san_delay_us": san_delay_us,
            "dan_delay_us": dan_delay_us,
            "wen_delay_us": wen_delay_us,
            "alt_km": float(llh[2] / 1e3),
            "lat_deg": float(llh[0]),
            "lon_deg": float(llh[1]),
            "fit_cost": float(cost),
            "dan_delay0_us": float(dan_delay0_us),
            "wen_delay0_us": float(wen_delay0_us),
        }
        rows.append(row)
        points.append(xhat)
        times_ns.append(match["time_ns"])

    points = n.asarray(points)
    times_ns = n.asarray(times_ns, dtype=n.int64)
    rel_t, fitted, velocity_mps = fit_track(points, times_ns)
    fit_llh = n.asarray([jcoord.ecef2geodetic(p[0], p[1], p[2]) for p in fitted])
    alt_km = n.asarray([row["alt_km"] for row in rows], dtype=n.float64)
    fit_alt_km = fit_llh[:, 2] / 1e3

    summary = {
        "event_id": event_id,
        "n_points": len(rows),
        "start_utc": rows[0]["utc"],
        "end_utc": rows[-1]["utc"],
        "duration_s": float(rel_t[-1]) if len(rel_t) > 1 else 0.0,
        "speed_km_s": float(n.linalg.norm(velocity_mps) / 1e3),
        "start_alt_km": float(alt_km[0]),
        "end_alt_km": float(alt_km[-1]),
        "start_lat_deg": float(rows[0]["lat_deg"]),
        "start_lon_deg": float(rows[0]["lon_deg"]),
        "end_lat_deg": float(rows[-1]["lat_deg"]),
        "end_lon_deg": float(rows[-1]["lon_deg"]),
    }
    return rows, (summary, rel_t, alt_km, fit_alt_km)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triangulate tri-static meteor echoes from per-site delay files."
    )
    parser.add_argument("--san-pattern", default=DEFAULT_SAN_PATTERN)
    parser.add_argument("--dan-pattern", default=DEFAULT_DAN_PATTERN)
    parser.add_argument("--wen-pattern", default=DEFAULT_WEN_PATTERN)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--match-tolerance-ms", type=float, default=DEFAULT_MATCH_TOLERANCE_MS)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    san_events = load_events(args.san_pattern)
    dan_events = load_events(args.dan_pattern)
    wen_events = load_events(args.wen_pattern)

    if not san_events or not dan_events or not wen_events:
        raise SystemExit(
            "Need local per-site event files first. Expected patterns: "
            f"{args.san_pattern}, {args.dan_pattern}, {args.wen_pattern}"
        )

    triplets = pair_tristatic_events(san_events, dan_events, wen_events)
    all_rows = []
    summaries = []

    for idx, (san_event, dan_event, wen_event) in enumerate(triplets):
        event_id = f"tri_{idx:04d}_{san_event.t0_ns}"
        rows, extra = process_triplet(
            event_id, san_event, dan_event, wen_event, args.match_tolerance_ms
        )
        if not rows or extra is None:
            continue

        summary, rel_t, alt_km, fit_alt_km = extra
        summaries.append(summary)
        all_rows.extend(rows)
        plot_event(
            os.path.join(args.results_dir, f"{event_id}.png"),
            event_id,
            rows,
            rel_t,
            alt_km,
            fit_alt_km,
        )

    results_h5 = os.path.join(args.results_dir, "tristatic_results.h5")
    write_results_h5(results_h5, all_rows, summaries)
    plot_overview(os.path.join(args.results_dir, "tristatic_overview.png"), all_rows)

    print(f"Wrote {len(all_rows)} tri-static delay rows and {len(summaries)} summaries to {results_h5}")


if __name__ == "__main__":
    main()
