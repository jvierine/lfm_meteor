import os

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

import sanya_opts as sc
from grid_search_delays_beam_axis import (
    DAN_CENTER_US,
    WEN_CENTER_US,
    build_trajectories,
    solve_trajectory_points,
)


OUTPUT_PNG = os.path.join("results", "meteor_positions_latlon_height.png")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/meteor_positions_latlon_height.png"
INPUT_H5 = os.path.join("results", "all_tristatic_ballistic_snr_weighted_v20260611c.h5")
BEAM_AZ_DEG = 15.0
BEAM_EL_DEG = 75.0
MAX_LAT_DEG = 18.7


def ecef_to_llh(points):
    return np.asarray([jcoord.ecef2geodetic(point[0], point[1], point[2]) for point in points], dtype=np.float64)


def beam_axis_llh(max_range_km=170.0, n_points=500):
    ranges_m = np.linspace(0.0, max_range_km * 1e3, n_points)
    llh = [
        jcoord.az_el_r2geodetic(
            sc.lat0[0],
            sc.lon0[0],
            sc.alt0[0] * 1e3,
            BEAM_AZ_DEG,
            BEAM_EL_DEG,
            range_m,
        )
        for range_m in ranges_m
    ]
    return np.asarray(llh, dtype=np.float64)


def solve_all_positions(dan_delay0_us=DAN_CENTER_US, wen_delay0_us=WEN_CENTER_US):
    if os.path.exists(INPUT_H5):
        llh_chunks = []
        with h5py.File(INPUT_H5, "r") as h:
            for event_id in h["event_id"][:]:
                name = event_id.decode("utf-8") if isinstance(event_id, bytes) else str(event_id)
                group = h["points"][name]
                llh_chunks.append(
                    np.column_stack(
                        [
                            group["lat_deg"][:],
                            group["lon_deg"][:],
                            group["alt_km"][:] * 1e3,
                        ]
                    )
                )
        if not llh_chunks:
            raise RuntimeError(f"No fitted trajectory samples found in {INPUT_H5}")
        return np.vstack(llh_chunks), len(llh_chunks)

    trajectories = build_trajectories()
    llh_chunks = []
    for trajectory in trajectories:
        points = solve_trajectory_points(trajectory, dan_delay0_us, wen_delay0_us)
        llh_chunks.append(ecef_to_llh(points))
    if not llh_chunks:
        raise RuntimeError("No tri-static trajectory points were solved.")
    return np.vstack(llh_chunks), len(trajectories)


def main():
    llh, n_trajectories = solve_all_positions()
    lat_deg = llh[:, 0]
    lon_deg = llh[:, 1]
    alt_km = llh[:, 2] / 1e3

    finite = np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(alt_km)
    keep = finite & (lat_deg <= MAX_LAT_DEG)
    n_rejected = int(np.count_nonzero(finite & ~keep))
    lat_deg = lat_deg[keep]
    lon_deg = lon_deg[keep]
    alt_km = alt_km[keep]

    axis_llh = beam_axis_llh()
    axis_alt_km = axis_llh[:, 2] / 1e3
    axis_mask = (axis_alt_km >= max(0.0, np.nanmin(alt_km) - 5.0)) & (
        axis_alt_km <= np.nanmax(alt_km) + 5.0
    )
    axis_llh = axis_llh[axis_mask]
    axis_alt_km = axis_alt_km[axis_mask]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    scatter_kwargs = {
        "s": 8,
        "alpha": 0.45,
        "linewidths": 0,
        "color": "#1f77b4",
        "label": "Meteor positions",
    }
    line_kwargs = {
        "color": "black",
        "linewidth": 2.0,
        "label": f"Sanya beam axis ({BEAM_AZ_DEG:.0f} deg az, {BEAM_EL_DEG:.0f} deg el)",
    }

    axes[0].scatter(lat_deg, alt_km, **scatter_kwargs)
    axes[0].plot(axis_llh[:, 0], axis_alt_km, **line_kwargs)
    axes[0].set_xlabel("Latitude (deg)")
    axes[0].set_ylabel("Height (km)")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(lon_deg, alt_km, **scatter_kwargs)
    axes[1].plot(axis_llh[:, 1], axis_alt_km, **line_kwargs)
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.suptitle(
        "Tri-static Meteor Positions vs Sanya Beam Axis\n"
        f"Danzhou delay={DAN_CENTER_US:.3f} us, Wenchang delay={WEN_CENTER_US:.3f} us; "
        f"{lat_deg.size} points from {n_trajectories} trajectories, lat <= {MAX_LAT_DEG:.1f} deg"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    os.makedirs(os.path.dirname(PAPER_OUTPUT_PNG), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_PNG, dpi=220)
    plt.close(fig)

    print(f"points: {lat_deg.size}")
    print(f"rejected latitude outliers: {n_rejected}")
    print(f"trajectories: {n_trajectories}")
    print(f"height range: {np.nanmin(alt_km):.3f} to {np.nanmax(alt_km):.3f} km")
    print(f"latitude range: {np.nanmin(lat_deg):.6f} to {np.nanmax(lat_deg):.6f} deg")
    print(f"longitude range: {np.nanmin(lon_deg):.6f} to {np.nanmax(lon_deg):.6f} deg")
    print(INPUT_H5 if os.path.exists(INPUT_H5) else "legacy point solver")
    print(OUTPUT_PNG)
    print(PAPER_OUTPUT_PNG)


if __name__ == "__main__":
    main()
