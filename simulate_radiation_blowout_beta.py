#!/usr/bin/env python3
"""Validate radiation-pressure blowout thresholds with REBOUND."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rebound


GMSUN = 4.0 * np.pi**2  # AU^3 / yr^2 for a 1 Msun Sun.
VERSION = "v20260701a"


def beta_crit_true_anomaly(ecc, true_anomaly_rad):
    ecc = np.asarray(ecc, dtype=np.float64)
    true_anomaly_rad = np.asarray(true_anomaly_rad, dtype=np.float64)
    return (1.0 - ecc**2) / (2.0 * (1.0 + ecc * np.cos(true_anomaly_rad)))


def solve_kepler(mean_anomaly_rad, ecc, tol=1e-13, max_iter=64):
    mean_anomaly_rad = np.asarray(mean_anomaly_rad, dtype=np.float64)
    ecc = np.asarray(ecc, dtype=np.float64)
    mean_wrapped = (mean_anomaly_rad + np.pi) % (2.0 * np.pi) - np.pi
    E = mean_wrapped + ecc * np.sin(mean_wrapped) / np.maximum(1.0 - np.sin(mean_wrapped + ecc) + np.sin(mean_wrapped), 0.2)
    for _ in range(max_iter):
        f = E - ecc * np.sin(E) - mean_wrapped
        fp = 1.0 - ecc * np.cos(E)
        step = f / fp
        E -= step
        if np.nanmax(np.abs(step)) < tol:
            break
    return E


def beta_crit_mean_anomaly(ecc, mean_anomaly_rad):
    E = solve_kepler(mean_anomaly_rad, ecc)
    return 0.5 * (1.0 - ecc * np.cos(E))


def parent_state(a_au, ecc, true_anomaly_rad):
    p = a_au * (1.0 - ecc**2)
    r = p / (1.0 + ecc * np.cos(true_anomaly_rad))
    x = r * np.cos(true_anomaly_rad)
    y = r * np.sin(true_anomaly_rad)
    vscale = np.sqrt(GMSUN / p)
    vx = -vscale * np.sin(true_anomaly_rad)
    vy = vscale * (ecc + np.cos(true_anomaly_rad))
    return np.array([x, y, 0.0]), np.array([vx, vy, 0.0])


def two_body_elements_after_release(r_vec, v_vec, beta):
    mu = GMSUN * (1.0 - beta)
    r = float(np.linalg.norm(r_vec))
    v2 = float(np.dot(v_vec, v_vec))
    energy = 0.5 * v2 - mu / r
    h_vec = np.cross(r_vec, v_vec)
    h2 = float(np.dot(h_vec, h_vec))
    if energy < 0.0:
        a = -mu / (2.0 * energy)
    else:
        a = -mu / (2.0 * energy) if energy > 0.0 else np.inf
    ecc2 = 1.0 + 2.0 * energy * h2 / mu**2
    return energy, a, float(np.sqrt(max(ecc2, 0.0)))


def run_rebound_release(a_au, ecc, true_anomaly_rad, beta, t_end_yr=4.0):
    r_vec, v_vec = parent_state(a_au, ecc, true_anomaly_rad)
    sim = rebound.Simulation()
    sim.G = GMSUN
    sim.integrator = "whfast"
    sim.dt = 0.002
    sim.add(m=1.0 - beta)
    sim.add(m=0.0, x=r_vec[0], y=r_vec[1], z=0.0, vx=v_vec[0], vy=v_vec[1], vz=0.0)
    sim.move_to_com()
    initial_energy, initial_a, initial_e = two_body_elements_after_release(r_vec, v_vec, beta)
    n_steps = 48
    times = np.linspace(0.0, t_end_yr, n_steps)
    radius = np.empty(n_steps, dtype=np.float64)
    for idx, time in enumerate(times):
        sim.integrate(float(time), exact_finish_time=0)
        p = sim.particles[1]
        radius[idx] = np.sqrt(p.x**2 + p.y**2 + p.z**2)
    p = sim.particles[1]
    final_r = np.array([p.x, p.y, p.z])
    final_v = np.array([p.vx, p.vy, p.vz])
    final_energy, final_a, final_e = two_body_elements_after_release(final_r, final_v, beta)
    return {
        "times_yr": times,
        "radius_au": radius,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "initial_a_au": initial_a,
        "final_a_au": final_a,
        "initial_e": initial_e,
        "final_e": final_e,
        "escaped_energy": bool(final_energy >= 0.0),
        "max_radius_au": float(np.max(radius)),
    }


def generate_validation_grid():
    ecc_values = np.array([0.0, 0.5, 0.9, 0.99], dtype=np.float64)
    true_anomaly_values = np.deg2rad(np.arange(0.0, 360.0, 15.0))
    ratios = np.array([0.8, 1.2], dtype=np.float64)
    rows = []
    for ecc in ecc_values:
        for f in true_anomaly_values:
            bc = float(beta_crit_true_anomaly(ecc, f))
            for ratio in ratios:
                beta = ratio * bc
                r_vec, v_vec = parent_state(1.0, ecc, f)
                sim = rebound.Simulation()
                sim.G = GMSUN
                sim.add(m=1.0 - beta)
                sim.add(m=0.0, x=r_vec[0], y=r_vec[1], z=0.0, vx=v_vec[0], vy=v_vec[1], vz=0.0)
                p = sim.particles[1]
                rr = np.array([p.x, p.y, p.z])
                vv = np.array([p.vx, p.vy, p.vz])
                energy, _a, post_e = two_body_elements_after_release(rr, vv, beta)
                rows.append(
                    (
                        ecc,
                        f,
                        ratio,
                        bc,
                        beta,
                        energy,
                        energy,
                        post_e,
                        post_e,
                        np.nan,
                    )
                )
    dtype = [
        ("ecc", "f8"),
        ("true_anomaly_rad", "f8"),
        ("beta_over_beta_crit", "f8"),
        ("beta_crit", "f8"),
        ("beta", "f8"),
        ("initial_energy", "f8"),
        ("final_energy", "f8"),
        ("initial_e_beta", "f8"),
        ("final_e_beta", "f8"),
        ("max_radius_au", "f8"),
    ]
    return np.array(rows, dtype=dtype)


def generate_time_series():
    cases = [
        ("e0.9_peri_bound", 0.9, 0.0, 0.8),
        ("e0.9_peri_unbound", 0.9, 0.0, 1.2),
        ("e0.9_quadrature_bound", 0.9, 0.5 * np.pi, 0.8),
        ("e0.9_quadrature_unbound", 0.9, 0.5 * np.pi, 1.2),
    ]
    out = {}
    for name, ecc, f, ratio in cases:
        bc = float(beta_crit_true_anomaly(ecc, f))
        beta = ratio * bc
        result = run_rebound_release(1.0, ecc, f, beta, t_end_yr=2.0)
        out[name] = {
            "ecc": ecc,
            "true_anomaly_rad": f,
            "beta_crit": bc,
            "beta": beta,
            **result,
        }
    return out


def write_h5(path, anomaly_grid, mean_grid, validation, time_series):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["version"] = VERSION
        h.attrs["description"] = "Radiation-pressure beta blowout threshold validation with REBOUND"
        g = h.create_group("analytic")
        for key, value in anomaly_grid.items():
            g[key] = value
        gm = h.create_group("mean_anomaly")
        for key, value in mean_grid.items():
            gm[key] = value
        h.create_dataset("rebound_validation", data=validation)
        gt = h.create_group("time_series")
        for name, values in time_series.items():
            c = gt.create_group(name)
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    c[key] = value
                else:
                    c.attrs[key] = value


def make_plots(figure_base, anomaly_grid, mean_grid, validation, time_series):
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    ax = axes[0, 0]
    f_deg = np.rad2deg(anomaly_grid["true_anomaly_rad"])
    for idx, ecc in enumerate(anomaly_grid["ecc_values"]):
        ax.plot(f_deg, anomaly_grid["beta_crit_true"][idx], label=f"e = {ecc:g}", lw=1.7)
    ax.set_xlabel("True anomaly (deg)")
    ax.set_ylabel(r"Critical $\beta$")
    ax.set_title("True-anomaly blowout threshold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    ax = axes[0, 1]
    M_deg = np.rad2deg(mean_grid["mean_anomaly_rad"])
    for idx, ecc in enumerate(mean_grid["ecc_values"]):
        ax.plot(M_deg, mean_grid["beta_crit_mean"][idx], label=f"e = {ecc:g}", lw=1.7)
    ax.set_xlabel("Mean anomaly (deg)")
    ax.set_ylabel(r"Critical $\beta$")
    ax.set_title("Mean-anomaly blowout threshold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    bound = validation["final_energy"] < 0.0
    colors = np.where(bound, "#4c78a8", "#e45756")
    ax.scatter(
        validation["beta_over_beta_crit"],
        validation["final_e_beta"],
        c=colors,
        s=12,
        alpha=0.75,
        linewidths=0,
    )
    ax.axvline(1.0, color="0.1", lw=1.0, ls="--")
    ax.axhline(1.0, color="0.1", lw=1.0, ls=":")
    ax.set_xlabel(r"$\beta/\beta_{\rm crit}$")
    ax.set_ylabel("Post-release eccentricity")
    ax.set_title("REBOUND classification")
    ax.grid(True, alpha=0.3)
    ax.text(0.04, 0.95, "blue: bound\nred: unbound", transform=ax.transAxes, va="top")

    ax = axes[1, 1]
    for name, values in time_series.items():
        label = (
            f"e={values['ecc']:.1f}, f={np.rad2deg(values['true_anomaly_rad']):.0f} deg, "
            rf"$\beta/\beta_c$={values['beta']/values['beta_crit']:.1f}"
        )
        ax.plot(values["times_yr"], values["radius_au"], lw=1.5, label=label)
    ax.set_xlabel("Time after release (yr)")
    ax.set_ylabel("Heliocentric distance (AU)")
    ax.set_title("Example REBOUND trajectories")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle("Radiation-pressure blowout depends strongly on release anomaly")
    for ext in ("pdf", "png"):
        fig.savefig(figure_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("/Users/jvi019/src/sanya_tristatic_paper"),
    )
    parser.add_argument("--version", default=VERSION)
    args = parser.parse_args()

    print("building analytic grids", flush=True)
    ecc_values = np.array([0.0, 0.5, 0.9, 0.99], dtype=np.float64)
    true_anomaly_rad = np.linspace(0.0, 2.0 * np.pi, 721)
    beta_true = np.vstack([beta_crit_true_anomaly(e, true_anomaly_rad) for e in ecc_values])
    anomaly_grid = {
        "ecc_values": ecc_values,
        "true_anomaly_rad": true_anomaly_rad,
        "beta_crit_true": beta_true,
    }
    mean_anomaly_rad = np.linspace(0.0, 2.0 * np.pi, 721)
    beta_mean = np.vstack([beta_crit_mean_anomaly(e, mean_anomaly_rad) for e in ecc_values])
    mean_grid = {
        "ecc_values": ecc_values,
        "mean_anomaly_rad": mean_anomaly_rad,
        "beta_crit_mean": beta_mean,
    }
    print("building rebound validation grid", flush=True)
    validation = generate_validation_grid()
    print("integrating rebound example trajectories", flush=True)
    time_series = generate_time_series()

    h5_path = Path("results") / f"memo29_radiation_blowout_beta_{args.version}.h5"
    figure_base = args.paper_dir / "memos" / "figures" / f"memo29_radiation_blowout_beta_{args.version}"
    print("writing hdf5", flush=True)
    write_h5(h5_path, anomaly_grid, mean_grid, validation, time_series)
    print("writing figures", flush=True)
    make_plots(figure_base, anomaly_grid, mean_grid, validation, time_series)
    print(f"wrote {h5_path}")
    print(f"wrote {figure_base.with_suffix('.pdf')}")
    print(f"wrote {figure_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
