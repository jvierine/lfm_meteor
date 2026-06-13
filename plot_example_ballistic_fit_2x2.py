import argparse
import os

import astropy.units as u
import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time

import fit_gcrs_trajectories_lfm_ambiguity as gfit


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260611c.h5"
OUTPUT_DIR = "results"
AXIS_LABELS = ("GCRS x - mean (km)", "GCRS y - mean (km)", "GCRS z - mean (km)")


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def choose_event(h, requested=None):
    event_ids = decode_strings(h["event_id"][:])
    if requested is not None:
        matches = np.flatnonzero(event_ids == requested)
        if len(matches) == 0:
            raise ValueError(f"Event {requested} not found in {INPUT_H5}")
        return int(matches[0]), event_ids[int(matches[0])]

    n_points = h["n_points"][:]
    rms = h["rms_total_path_residual_m"][:]
    weighted_rms = h["weighted_rms"][:]
    score = np.abs(rms - np.nanmedian(rms)) + 5.0 * np.abs(weighted_rms - 1.0)
    score[n_points < 20] = np.inf
    idx = int(np.nanargmin(score))
    return idx, event_ids[idx]


def triangulate_points(measured_total_paths_m):
    x0 = gfit.initial_guess(15.0, 75.0, float(np.nanmedian(measured_total_paths_m[:, 0]) / 2e3))
    points = []
    for measured in measured_total_paths_m:
        point = gfit.solve_position_from_total_paths_m(measured, x0)
        points.append(point)
        x0 = point
    return np.asarray(points, dtype=np.float64)


def fitted_lfm_corrected_points(measured_total_paths_m, x_itrs_m, v_itrs_mps):
    _geom_paths_m, path_rates_mps = gfit.link_total_paths_and_rates_m(
        x_itrs_m,
        v_itrs_mps,
        gfit.LINK_TX_POSITIONS_M,
        gfit.LINK_RX_POSITIONS_M,
    )
    corrected_paths_m = measured_total_paths_m - gfit.lfm_total_path_bias_m(path_rates_mps)
    return triangulate_points(corrected_paths_m)


def ecef_to_gcrs(points_ecef_m, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    itrs = ITRS(
        CartesianRepresentation(
            points_ecef_m[:, 0] * u.m,
            points_ecef_m[:, 1] * u.m,
            points_ecef_m[:, 2] * u.m,
        ),
        obstime=obstime,
    )
    return itrs.transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m).T


def itrs_state_to_gcrs_positions(positions_itrs_m, velocities_itrs_mps, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    representation = CartesianRepresentation(
        positions_itrs_m[:, 0] * u.m,
        positions_itrs_m[:, 1] * u.m,
        positions_itrs_m[:, 2] * u.m,
        differentials=CartesianDifferential(
            velocities_itrs_mps[:, 0] * u.m / u.s,
            velocities_itrs_mps[:, 1] * u.m / u.s,
            velocities_itrs_mps[:, 2] * u.m / u.s,
        ),
    )
    itrs = ITRS(representation, obstime=obstime)
    return itrs.transform_to(GCRS(obstime=obstime)).cartesian.without_differentials().xyz.to_value(u.m).T


def plot_event(h, idx, event_id):
    g = h["points"][event_id]
    times_ns = g["time_ns"][:]
    t_rel_s = g["t_rel_s"][:]
    measured_total_paths_m = g["measured_total_paths_m"][:]
    residuals_m = g["residuals_m"][:]
    x_itrs_m = g["x_itrs_m"][:]
    v_itrs_mps = g["v_itrs_mps"][:]
    corrected_seed_gcrs_m = ecef_to_gcrs(fitted_lfm_corrected_points(measured_total_paths_m, x_itrs_m, v_itrs_mps), times_ns)
    fit_gcrs_m = itrs_state_to_gcrs_positions(x_itrs_m, v_itrs_mps, times_ns)

    center_m = np.nanmean(corrected_seed_gcrs_m, axis=0)
    seed_km = (corrected_seed_gcrs_m - center_m[None, :]) / 1e3
    fit_km = (fit_gcrs_m - center_m[None, :]) / 1e3
    per_pulse_rms_m = np.sqrt(np.mean(residuals_m**2.0, axis=1))

    rms = float(h["rms_total_path_residual_m"][idx])
    weighted_rms = float(h["weighted_rms"][idx])
    b_drag = float(h["b_drag_m2_per_kg"][idx])
    n_points = int(h["n_points"][idx])
    start_speed = float(h["start_speed_km_s"][idx])
    end_speed = float(h["end_speed_km_s"][idx])

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    for comp, ax in enumerate(axes.flat[:3]):
        ax.plot(t_rel_s, seed_km[:, comp], "k.", ms=4, label="LFM-corrected point solution")
        ax.plot(t_rel_s, fit_km[:, comp], color="#2ca02c", lw=1.8, label="weighted ballistic fit")
        ax.set_xlabel("Time since first matched pulse (s)")
        ax.set_ylabel(AXIS_LABELS[comp])
        ax.grid(True, alpha=0.28)
        if comp == 0:
            ax.legend(loc="best")

    ax = axes.flat[3]
    ax.plot(t_rel_s, per_pulse_rms_m, "o-", ms=3.5, color="#2ca02c")
    ax.axhline(rms, color="0.25", lw=1.2, ls="--", label="overall RMS")
    ax.set_xlabel("Time since first matched pulse (s)")
    ax.set_ylabel("Per-pulse total-path RMS (m)")
    ax.set_title(f"RMS={rms:.2f} m, weighted RMS={weighted_rms:.2f}\nB={b_drag:.2f} m$^2$ kg$^{{-1}}$")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")

    fig.suptitle(
        f"Example weighted ballistic fit: {event_id}\n"
        f"{n_points} pulses, speed {start_speed:.2f} to {end_speed:.2f} km s$^{{-1}}$",
        fontsize=12,
    )
    safe_id = event_id.replace("/", "_")
    png = os.path.join(OUTPUT_DIR, f"example_ballistic_fit_2x2_{safe_id}.png")
    pdf = os.path.join(OUTPUT_DIR, f"example_ballistic_fit_2x2_{safe_id}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description="Plot a 2x2 weighted-ballistic fit diagnostic for one event.")
    parser.add_argument("--event-id", default=None)
    args = parser.parse_args()
    with h5py.File(INPUT_H5, "r") as h:
        idx, event_id = choose_event(h, args.event_id)
        png, pdf = plot_event(h, idx, event_id)
        print(f"event_id={event_id}")
        print(f"wrote {png}")
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
