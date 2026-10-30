#!/usr/bin/env python3
"""Propagate Sanya tri-static WJ orbit samples backward and classify them.

The input catalogue contains Whipple-Jacchia bootstrap samples of the GCRS
position and velocity at first radar detection.  This script propagates each
sample backward with REBOUND and computes heliocentric Keplerian elements at
the requested epoch.  The Jopek-Williams cometary/asteroidal criteria are then
evaluated on the propagated ecliptic elements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import rebound
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time


AU_M = 1.495978707e11
G_M3_KG_S2 = 6.67430e-11
MU_SUN_M3_S2 = 1.3271244004193938e20
JUPITER_A_AU = 5.2038
SIDEREAL_YEAR_J2000_D = 365.256363004
M_EARTH_KG = 5.9722e24
M_SUN_KG = 1.98847e30
GAUSS_GRAV_K = 2.0 * np.pi / (
    SIDEREAL_YEAR_J2000_D * np.sqrt(1.0 + M_EARTH_KG / M_SUN_KG)
)
J2000_OBLIQUITY_DEG = 23.4392911

BODY_GM_M3_S2 = {
    "sun": MU_SUN_M3_S2,
    "mercury": 2.2032e13,
    "venus": 3.24858592e14,
    "earth": 3.98600435436e14,
    "moon": 4.902800118e12,
    "mars": 4.2828375214e13,
    "jupiter": 1.266865349218008e17,
    "saturn": 3.793120749865224e16,
    "uranus": 5.793951322279009e15,
    "neptune": 6.835099502439672e15,
}
PROPAGATING_BODIES = tuple(BODY_GM_M3_S2.keys())

DEFAULT_INPUT_DIR = Path("results/tristatic_whipple_jacchia_monotonic_bootstrap_orbit100_20260702")
DEFAULT_OUTPUT_H5 = Path("results/tristatic_orbits_minus1000d_jopek_williams_20260703.h5")
DEFAULT_OUTPUT_TEX = Path(
    "/Users/jvi019/src/sanya_tristatic_paper/tables/tristatic_orbit_classification_minus1000d.tex"
)


def equatorial_to_ecliptic(vec: np.ndarray) -> np.ndarray:
    eps = np.deg2rad(J2000_OBLIQUITY_DEG)
    ce = np.cos(eps)
    se = np.sin(eps)
    out = np.array(vec, dtype=np.float64, copy=True)
    y = out[..., 1].copy()
    z = out[..., 2].copy()
    out[..., 1] = ce * y + se * z
    out[..., 2] = -se * y + ce * z
    return out


def kepler_from_heliocentric_state(r_m: np.ndarray, v_mps: np.ndarray) -> dict[str, np.ndarray]:
    r_m = np.asarray(r_m, dtype=np.float64)
    v_mps = np.asarray(v_mps, dtype=np.float64)
    n = r_m.shape[0]
    out = {
        "a_au": np.full(n, np.nan),
        "e": np.full(n, np.nan),
        "i_deg": np.full(n, np.nan),
        "q_au": np.full(n, np.nan),
        "Q_au": np.full(n, np.nan),
        "period_years": np.full(n, np.nan),
        "specific_energy_j_kg": np.full(n, np.nan),
        "tisserand_jupiter": np.full(n, np.nan),
    }
    good = np.all(np.isfinite(r_m), axis=1) & np.all(np.isfinite(v_mps), axis=1)
    if not np.any(good):
        return out

    r = equatorial_to_ecliptic(r_m[good])
    v = equatorial_to_ecliptic(v_mps[good])
    rnorm = np.linalg.norm(r, axis=1)
    v2 = np.sum(v * v, axis=1)
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h, axis=1)
    evec = np.cross(v, h) / MU_SUN_M3_S2 - r / np.maximum(rnorm[:, None], 1e-30)
    e = np.linalg.norm(evec, axis=1)
    energy = 0.5 * v2 - MU_SUN_M3_S2 / rnorm
    a_m = -MU_SUN_M3_S2 / (2.0 * energy)
    a_au = a_m / AU_M
    inc = np.rad2deg(np.arccos(np.clip(h[:, 2] / np.maximum(hnorm, 1e-30), -1.0, 1.0)))
    q_au = a_au * (1.0 - e)
    q_au = np.where(e >= 1.0, np.abs(a_au) * (e - 1.0), q_au)
    Q_au = np.where(e < 1.0, a_au * (1.0 + e), np.nan)
    period_years = np.where(a_au > 0.0, np.power(a_au, 1.5), np.inf)
    tj_term = a_au * (1.0 - e * e)
    tj = 1.0 / a_au + 2.0 * JUPITER_A_AU ** (-1.5) * np.sqrt(np.maximum(tj_term, 0.0)) * np.cos(
        np.deg2rad(inc)
    )

    idx = np.flatnonzero(good)
    out["a_au"][idx] = a_au
    out["e"][idx] = e
    out["i_deg"][idx] = inc
    out["q_au"][idx] = q_au
    out["Q_au"][idx] = Q_au
    out["period_years"][idx] = period_years
    out["specific_energy_j_kg"][idx] = energy
    out["tisserand_jupiter"][idx] = tj
    return out


def jopek_williams_criteria(a_au: np.ndarray, e: np.ndarray, inc_deg: np.ndarray, period_years: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        k_value = np.log(a_au * (1.0 + e) / (1.0 - e)) - 1.0
        q_aphelion = a_au * (1.0 + e)
        energy = -(GAUSS_GRAV_K**2) / (2.0 * a_au)
        tj_term = a_au * (1.0 - e * e)
        tj = 1.0 / a_au + 2.0 * JUPITER_A_AU ** (-1.5) * np.sqrt(np.maximum(tj_term, 0.0)) * np.cos(
            np.deg2rad(inc_deg)
        )

    high_i = inc_deg > 75.0
    hyperbolic = e >= 1.0
    return {
        "TJMinusI": np.logical_or.reduce((tj < 0.58, high_i, hyperbolic)),
        "KMinusI": np.logical_or.reduce((k_value > 0.0, high_i, hyperbolic)),
        "QMinusI": np.logical_or.reduce((q_aphelion > 4.6, high_i, hyperbolic)),
        "PMinusI": np.logical_or.reduce((period_years * e > 2.5, high_i, hyperbolic)),
        "EpsilonMinusI": np.logical_or.reduce((energy > -5.28e-5, high_i, hyperbolic)),
    }


CRITERIA_LABELS = {
    "TJMinusI": (r"$T_J-i$", "Tisserand parameter with respect to Jupiter"),
    "KMinusI": (r"$K-i$", "Semimajor-axis--eccentricity criterion"),
    "QMinusI": (r"$Q-i$", "Aphelion-distance criterion"),
    "PMinusI": (r"$P-i$", "Orbital-period criterion"),
    "EpsilonMinusI": (r"$\varepsilon-i$", "Specific-orbital-energy criterion"),
}


def body_states_at_epoch(epoch: Time) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    states = {}
    for body in PROPAGATING_BODIES:
        pos, vel = get_body_barycentric_posvel(body, epoch)
        states[body] = (pos.xyz.to_value(u.m), vel.xyz.to_value(u.m / u.s))
    return states


def propagate_event(path: Path, days_back: float) -> dict[str, object]:
    with h5py.File(path, "r") as h:
        g = h["joint_fit"]
        event_id = str(h.attrs.get("event_id", path.stem.replace("joint_delay_doppler_fft_", "")))
        time_ns = int(g["time_ns"][0])
        x_gcrs = np.asarray(g["bootstrap_x_gcrs_m"][:, 0, :], dtype=np.float64)
        v_gcrs = np.asarray(g["bootstrap_v_gcrs_mps"][:, 0, :], dtype=np.float64)
        e_initial = np.asarray(g["bootstrap_eccentricity"], dtype=np.float64)

    good = np.all(np.isfinite(x_gcrs), axis=1) & np.all(np.isfinite(v_gcrs), axis=1)
    epoch = Time(time_ns / 1e9, format="unix", scale="utc")
    states = body_states_at_epoch(epoch)
    earth_r, earth_v = states["earth"]
    met_r = x_gcrs[good] + earth_r[None, :]
    met_v = v_gcrs[good] + earth_v[None, :]

    sim = rebound.Simulation()
    sim.G = G_M3_KG_S2
    sim.integrator = "ias15"
    sim.ri_ias15.epsilon = 1e-10
    for body in PROPAGATING_BODIES:
        r_body, v_body = states[body]
        sim.add(
            m=BODY_GM_M3_S2[body] / G_M3_KG_S2,
            x=r_body[0],
            y=r_body[1],
            z=r_body[2],
            vx=v_body[0],
            vy=v_body[1],
            vz=v_body[2],
            hash=body,
        )
    for idx, (r_met, v_met) in enumerate(zip(met_r, met_v)):
        sim.add(m=0.0, x=r_met[0], y=r_met[1], z=r_met[2], vx=v_met[0], vy=v_met[1], vz=v_met[2], hash=f"m{idx}")
    sim.N_active = len(PROPAGATING_BODIES)
    sim.integrate(-float(days_back) * 86400.0, exact_finish_time=1)

    sun = sim.particles["sun"]
    sun_r = np.array([sun.x, sun.y, sun.z], dtype=np.float64)
    sun_v = np.array([sun.vx, sun.vy, sun.vz], dtype=np.float64)
    r_past = np.full_like(x_gcrs, np.nan)
    v_past = np.full_like(v_gcrs, np.nan)
    good_indices = np.flatnonzero(good)
    for local_idx, global_idx in enumerate(good_indices):
        part = sim.particles[f"m{local_idx}"]
        r_past[global_idx] = np.array([part.x, part.y, part.z], dtype=np.float64) - sun_r
        v_past[global_idx] = np.array([part.vx, part.vy, part.vz], dtype=np.float64) - sun_v

    elements = kepler_from_heliocentric_state(r_past, v_past)
    return {
        "event_id": event_id,
        "time_ns": time_ns,
        "valid": good,
        "initial_e": e_initial,
        **elements,
    }


def summarize_classification(events: list[dict[str, object]]) -> list[dict[str, object]]:
    a = np.concatenate([np.asarray(ev["a_au"]) for ev in events])
    e = np.concatenate([np.asarray(ev["e"]) for ev in events])
    inc = np.concatenate([np.asarray(ev["i_deg"]) for ev in events])
    period = np.concatenate([np.asarray(ev["period_years"]) for ev in events])
    finite = np.isfinite(a) & np.isfinite(e) & np.isfinite(inc)
    criteria = jopek_williams_criteria(a[finite], e[finite], inc[finite], period[finite])
    rows = []
    for key, cometary in criteria.items():
        cometary = np.asarray(cometary, dtype=bool)
        event_probs = []
        for ev in events:
            ev_a = np.asarray(ev["a_au"])
            ev_e = np.asarray(ev["e"])
            ev_i = np.asarray(ev["i_deg"])
            ev_p = np.asarray(ev["period_years"])
            ev_finite = np.isfinite(ev_a) & np.isfinite(ev_e) & np.isfinite(ev_i)
            if not np.any(ev_finite):
                continue
            ev_c = jopek_williams_criteria(ev_a[ev_finite], ev_e[ev_finite], ev_i[ev_finite], ev_p[ev_finite])[key]
            event_probs.append(float(np.mean(ev_c)))
        event_probs_arr = np.asarray(event_probs, dtype=np.float64)
        rows.append(
            {
                "key": key,
                "symbol": CRITERIA_LABELS[key][0],
                "name": CRITERIA_LABELS[key][1],
                "n_samples": int(cometary.size),
                "n_events": int(event_probs_arr.size),
                "cometary_sample_fraction": float(np.mean(cometary)),
                "asteroidal_sample_fraction": float(1.0 - np.mean(cometary)),
                "cometary_event_mean_fraction": float(np.mean(event_probs_arr)),
                "asteroidal_event_mean_fraction": float(1.0 - np.mean(event_probs_arr)),
                "cometary_event_majority_fraction": float(np.mean(event_probs_arr >= 0.5)),
                "asteroidal_event_majority_fraction": float(np.mean(event_probs_arr < 0.5)),
            }
        )
    rows.append(
        {
            "key": "CombinedMean",
            "symbol": "Combined",
            "name": "Mean of the five criteria",
            "n_samples": int(rows[0]["n_samples"]) if rows else 0,
            "n_events": int(rows[0]["n_events"]) if rows else 0,
            "cometary_sample_fraction": float(np.mean([row["cometary_sample_fraction"] for row in rows])),
            "asteroidal_sample_fraction": float(np.mean([row["asteroidal_sample_fraction"] for row in rows])),
            "cometary_event_mean_fraction": float(np.mean([row["cometary_event_mean_fraction"] for row in rows])),
            "asteroidal_event_mean_fraction": float(np.mean([row["asteroidal_event_mean_fraction"] for row in rows])),
            "cometary_event_majority_fraction": float(np.mean([row["cometary_event_majority_fraction"] for row in rows])),
            "asteroidal_event_majority_fraction": float(np.mean([row["asteroidal_event_majority_fraction"] for row in rows])),
        }
    )
    return rows


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.0f}\\%"


def write_outputs(events: list[dict[str, object]], rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    max_samples = max(len(np.asarray(ev["e"])) for ev in events)
    n_events = len(events)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output_h5, "w") as h:
        h.attrs["script"] = Path(__file__).name
        h.attrs["input_dir"] = str(args.input_dir)
        h.attrs["days_back"] = float(args.days_back)
        h.attrs["propagator"] = "REBOUND IAS15"
        h.attrs["included_bodies"] = ",".join(PROPAGATING_BODIES)
        h.attrs["orbit_epoch"] = f"first_detection_minus_{args.days_back:g}_days"
        h.attrs["classification"] = "Jopek-Williams cometary/asteroidal criteria"
        h.create_dataset("event_id", data=np.asarray([ev["event_id"] for ev in events], dtype=object), dtype=str_dtype)
        h.create_dataset("time_ns", data=np.asarray([ev["time_ns"] for ev in events], dtype=np.int64))
        for name in ("valid", "initial_e", "a_au", "e", "i_deg", "q_au", "Q_au", "period_years", "specific_energy_j_kg", "tisserand_jupiter"):
            fill = False if name == "valid" else np.nan
            dtype = np.bool_ if name == "valid" else np.float64
            arr = np.full((n_events, max_samples), fill, dtype=dtype)
            for idx, ev in enumerate(events):
                values = np.asarray(ev[name], dtype=dtype)
                arr[idx, : values.size] = values
            h.create_dataset(name, data=arr, compression="gzip", shuffle=True)
        summary = h.create_group("classification_summary")
        summary.create_dataset("key", data=np.asarray([r["key"] for r in rows], dtype=object), dtype=str_dtype)
        summary.create_dataset("symbol", data=np.asarray([r["symbol"] for r in rows], dtype=object), dtype=str_dtype)
        summary.create_dataset("name", data=np.asarray([r["name"] for r in rows], dtype=object), dtype=str_dtype)
        for field in (
            "n_samples",
            "n_events",
            "cometary_sample_fraction",
            "asteroidal_sample_fraction",
            "cometary_event_mean_fraction",
            "asteroidal_event_mean_fraction",
            "cometary_event_majority_fraction",
            "asteroidal_event_majority_fraction",
        ):
            summary.create_dataset(field, data=np.asarray([r[field] for r in rows]))

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by /Users/jvi019/src/lfm_meteor/propagate_tristatic_orbits_1000d_classification.py; do not edit by hand.\n",
        "\\begin{tabular}{llrr}\n",
        "\\toprule\n",
        "Criterion & Quantity & Cometary & Asteroidal \\\\\n",
        "\\midrule\n",
    ]
    for row in rows:
        lines.append(
            f"{row['symbol']} & {row['name']} & "
            f"{fmt_pct(row['cometary_sample_fraction'])} & {fmt_pct(row['asteroidal_sample_fraction'])} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    args.output_tex.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
    p.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    p.add_argument("--days-back", type=float, default=1000.0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("joint_delay_doppler_fft_tri_*.h5"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No tri-static WJ HDF5 files found in {args.input_dir}")

    solar_system_ephemeris.set("builtin")
    events = []
    for idx, path in enumerate(files, start=1):
        ev = propagate_event(path, args.days_back)
        events.append(ev)
        print(f"{idx:4d}/{len(files)} {ev['event_id']} propagated", flush=True)

    rows = summarize_classification(events)
    write_outputs(events, rows, args)
    print(f"wrote {args.output_h5}")
    print(f"wrote {args.output_tex}")
    for row in rows:
        print(
            f"{row['symbol']:18s} cometary_samples={100*row['cometary_sample_fraction']:.1f}% "
            f"asteroidal_samples={100*row['asteroidal_sample_fraction']:.1f}% "
            f"cometary_events={100*row['cometary_event_mean_fraction']:.1f}%"
        )


if __name__ == "__main__":
    main()
