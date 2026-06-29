import os
import glob

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import plot_memo09_antenna_gain_patterns as gain_model
import plot_sanya_beam_position_histogram as beam_hist
import sanya_opts as sc
from grid_search_delays_beam_axis import (
    DAN_PATTERN,
    DAN_CENTER_US,
    SAN_PATTERN,
    WEN_PATTERN,
    WEN_CENTER_US,
    build_trajectories,
    solve_trajectory_points,
)


OUTPUT_PNG = os.path.join("results", "meteor_positions_latlon_height.png")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/meteor_positions_latlon_height.png"
INPUT_H5 = os.path.join("results", "all_tristatic_ceplecha_snr_weighted_v20260616d.h5")
BEAM_AZ_DEG = 15.0
BEAM_EL_DEG = 75.0
MAX_LAT_DEG = 18.7
BEAM_MAX_RANGE_KM = 240.0


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


def median_pointing(pattern: str) -> tuple[float, float]:
    az = []
    el = []
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as h:
            az.append(float(h["az"][()]))
            el.append(float(h["el"][()]))
    if not az:
        raise FileNotFoundError(f"No event files matched {pattern!r}")
    return float(np.median(az)), float(np.median(el))


def beam_line_llh(site_idx: int, az_deg: float, el_deg: float, max_range_km: float) -> np.ndarray:
    ranges_m = np.linspace(0.0, max_range_km * 1e3, 700)
    lat0 = float(sc.lat0[site_idx])
    lon0 = float(sc.lon0[site_idx])
    alt0_m = float(sc.alt0[site_idx] * 1e3)
    return np.asarray(
        [
            jcoord.az_el_r2geodetic(lat0, lon0, alt0_m, az_deg, el_deg, range_m)
            for range_m in ranges_m
        ],
        dtype=np.float64,
    )


def solve_all_positions(dan_delay0_us=DAN_CENTER_US, wen_delay0_us=WEN_CENTER_US):
    if os.path.exists(INPUT_H5):
        llh_chunks = []
        speed_chunks = []
        with h5py.File(INPUT_H5, "r") as h:
            for event_id in h["event_id"][:]:
                name = event_id.decode("utf-8") if isinstance(event_id, bytes) else str(event_id)
                group = h["points"][name]
                n_rows = min(
                    group["lat_deg"].shape[0],
                    group["lon_deg"].shape[0],
                    group["alt_km"].shape[0],
                    group["speed_km_s"].shape[0],
                )
                llh_chunks.append(
                    np.column_stack(
                        [
                            group["lat_deg"][:n_rows],
                            group["lon_deg"][:n_rows],
                            group["alt_km"][:n_rows] * 1e3,
                        ]
                    )
                )
                speed_chunks.append(np.asarray(group["speed_km_s"][:n_rows], dtype=np.float64))
        if not llh_chunks:
            raise RuntimeError(f"No fitted trajectory samples found in {INPUT_H5}")
        return np.vstack(llh_chunks), np.concatenate(speed_chunks), len(llh_chunks)

    trajectories = build_trajectories()
    llh_chunks = []
    for trajectory in trajectories:
        points = solve_trajectory_points(trajectory, dan_delay0_us, wen_delay0_us)
        llh_chunks.append(ecef_to_llh(points))
    if not llh_chunks:
        raise RuntimeError("No tri-static trajectory points were solved.")
    llh = np.vstack(llh_chunks)
    return llh, np.full(llh.shape[0], np.nan), len(trajectories)


def beam_panel_data(input_h5):
    positions_ecef_m, n_events = beam_hist.collect_positions(input_h5)
    sanya_lat_deg, sanya_lon_deg, _sanya_alt_m = jcoord.ecef2geodetic(*beam_hist.gfit.LINK_TX_POSITIONS_M[0])
    los_ecef = beam_hist.unit(positions_ecef_m - beam_hist.gfit.LINK_TX_POSITIONS_M[0][None, :])
    los_enu = beam_hist.ecef_to_enu_vectors(los_ecef, sanya_lat_deg, sanya_lon_deg)

    site = gain_model.SITES[0]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    east_axis, north_axis = beam_hist.local_sky_axes(pointing)
    east_deg, north_deg = beam_hist.angular_offsets_deg(los_enu, pointing, east_axis, north_axis)
    finite = np.isfinite(east_deg) & np.isfinite(north_deg)
    east_deg = east_deg[finite]
    north_deg = north_deg[finite]

    grid = np.linspace(-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_N)
    east_grid, north_grid = np.meshgrid(grid, grid)
    gain_db = beam_hist.sanya_beam_relative_gain_db(east_grid, north_grid)
    hist_range = [[-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG], [-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG]]
    counts, east_edges, north_edges = np.histogram2d(east_deg, north_deg, bins=beam_hist.HIST_BINS, range=hist_range)
    counts = np.ma.masked_less_equal(counts.T, 0.0)
    max_count = float(np.nanmax(counts)) if np.any(counts > 0.0) else 1.0
    return {
        "east_edges": east_edges,
        "north_edges": north_edges,
        "counts": counts,
        "max_count": max_count,
        "east_grid": east_grid,
        "north_grid": north_grid,
        "gain_db": gain_db,
        "n_events": n_events,
        "n_positions": len(east_deg),
    }


def main():
    llh, speed_km_s, n_trajectories = solve_all_positions()
    lat_deg = llh[:, 0]
    lon_deg = llh[:, 1]
    alt_km = llh[:, 2] / 1e3

    finite = np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(alt_km) & np.isfinite(speed_km_s)
    keep = finite & (lat_deg <= MAX_LAT_DEG)
    n_rejected = int(np.count_nonzero(finite & ~keep))
    lat_deg = lat_deg[keep]
    lon_deg = lon_deg[keep]
    alt_km = alt_km[keep]
    speed_km_s = speed_km_s[keep]

    pointings = {
        "Sanya tx/rx": (0, *median_pointing(SAN_PATTERN), "black", "-"),
        "Danzhou rx": (1, *median_pointing(DAN_PATTERN), "#b34a2e", "--"),
        "Wenchang rx": (2, *median_pointing(WEN_PATTERN), "#2f7d4b", "--"),
    }
    alt_lo = float(np.nanmin(alt_km) - 7.0)
    alt_hi = float(np.nanmax(alt_km) + 7.0)
    beam = beam_panel_data(INPUT_H5)

    fig = plt.figure(figsize=(12.2, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes[1].sharey(axes[0])
    ax_beam = fig.add_subplot(grid[0, 2])
    scatter_kwargs = {"s": 8, "alpha": 0.58, "linewidths": 0, "cmap": "turbo"}
    vmin, vmax = np.nanpercentile(speed_km_s, [2.0, 98.0])

    sca = axes[0].scatter(
        lat_deg,
        alt_km,
        c=speed_km_s,
        vmin=vmin,
        vmax=vmax,
        label="Tri-static meteor positions",
        **scatter_kwargs,
    )
    axes[1].scatter(lon_deg, alt_km, c=speed_km_s, vmin=vmin, vmax=vmax, **scatter_kwargs)

    for label, (site_idx, az_deg, el_deg, color, linestyle) in pointings.items():
        line = beam_line_llh(site_idx, az_deg, el_deg, BEAM_MAX_RANGE_KM)
        line_alt_km = line[:, 2] / 1e3
        mask = np.isfinite(line_alt_km) & (line_alt_km >= alt_lo) & (line_alt_km <= alt_hi)
        axes[0].plot(
            line[mask, 0],
            line_alt_km[mask],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} ({az_deg:.1f}/{el_deg:.1f} deg)",
        )
        axes[1].plot(
            line[mask, 1],
            line_alt_km[mask],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} ({az_deg:.1f}/{el_deg:.1f} deg)",
        )

    axes[0].set_xlabel("Latitude (deg)")
    axes[0].set_ylabel("Height (km)")
    axes[1].set_xlabel("Longitude (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_ylim(alt_lo, alt_hi)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    cb = fig.colorbar(sca, ax=axes, pad=0.01)
    cb.set_label(r"Fitted geocentric speed, $v_g$ (km s$^{-1}$)")

    hist = ax_beam.pcolormesh(
        beam["east_edges"],
        beam["north_edges"],
        beam["counts"],
        cmap="magma",
        norm=LogNorm(vmin=1.0, vmax=max(beam["max_count"], 1.0)),
        shading="auto",
        alpha=0.82,
        rasterized=True,
    )
    ax_beam.contour(
        beam["east_grid"],
        beam["north_grid"],
        beam["gain_db"],
        levels=[-30.0, -20.0, -13.3, -10.0, -3.0],
        colors=["0.55", "0.35", "0.20", "0.10", "0.0"],
        linewidths=[0.7, 0.8, 0.9, 1.0, 1.25],
        linestyles=[":", "--", "-.", "-", "-"],
    )
    ax_beam.axhline(0.0, color="0.2", lw=0.7, alpha=0.6)
    ax_beam.axvline(0.0, color="0.2", lw=0.7, alpha=0.6)
    ax_beam.set_aspect("equal", adjustable="box")
    ax_beam.set_xlim(-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG)
    ax_beam.set_ylim(-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG)
    ax_beam.set_xlabel("East (deg)")
    ax_beam.set_ylabel("North (deg)")
    ax_beam.set_title("Tri-static beam positions")
    cb_beam = fig.colorbar(hist, ax=ax_beam, fraction=0.046, pad=0.03)
    cb_beam.set_label("Pulse count per angular bin")

    fig.suptitle(
        "Tri-static meteor head-echo positions and radar beam lines\n"
        f"{lat_deg.size} points from {n_trajectories} trajectories, lat <= {MAX_LAT_DEG:.1f} deg"
    )
    fig.savefig(OUTPUT_PNG, dpi=220)
    os.makedirs(os.path.dirname(PAPER_OUTPUT_PNG), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_PNG, dpi=220)
    plt.close(fig)

    print(f"points: {lat_deg.size}")
    print(f"rejected latitude outliers: {n_rejected}")
    print(f"trajectories: {n_trajectories}")
    print(f"beam positions: {beam['n_positions']} from {beam['n_events']} events")
    print(f"height range: {np.nanmin(alt_km):.3f} to {np.nanmax(alt_km):.3f} km")
    print(f"latitude range: {np.nanmin(lat_deg):.6f} to {np.nanmax(lat_deg):.6f} deg")
    print(f"longitude range: {np.nanmin(lon_deg):.6f} to {np.nanmax(lon_deg):.6f} deg")
    print(f"fitted geocentric speed range: {np.nanmin(speed_km_s):.3f} to {np.nanmax(speed_km_s):.3f} km/s")
    for label, (_site_idx, az_deg, el_deg, _color, _linestyle) in pointings.items():
        print(f"{label}: az={az_deg:.6f} deg el={el_deg:.6f} deg")
    print(INPUT_H5 if os.path.exists(INPUT_H5) else "legacy point solver")
    print(OUTPUT_PNG)
    print(PAPER_OUTPUT_PNG)


if __name__ == "__main__":
    main()
