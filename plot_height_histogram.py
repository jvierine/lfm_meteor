import os
import glob

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from grid_search_delays_beam_axis import DAN_CENTER_US, MAX_LAT_DEG, WEN_CENTER_US, build_trajectories, solve_trajectory_points
import plot_memo09_antenna_gain_patterns as gain_model
import plot_sanya_beam_position_histogram as beam_hist
import sanya_opts as sc


OUTPUT_PNG = os.path.join("results", "meteor_height_histogram.png")
OUTPUT_PDF = os.path.join("results", "meteor_height_histogram.pdf")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/meteor_height_histogram.png"
PAPER_OUTPUT_PDF = "/Users/jvi019/src/sanya_tristatic_paper/figures/meteor_height_histogram.pdf"
INPUT_H5 = os.path.join("results", "all_tristatic_ceplecha_snr_weighted_v20260616d.h5")
INPUT_CATALOG_DIR = os.path.join("results", "tristatic_shrinking_radius_profile_fixedmask_capped_logr61_20260708")
FALLBACK_CATALOG_DIR = os.path.join("results", "tristatic_whipple_jacchia_monotonic_bootstrap_orbit100_20260702")
BEAM_TRACE_CATALOG_DIR = FALLBACK_CATALOG_DIR
SANYA_CONTAINMENT_AREA_H5 = os.path.join("results", "sanya_gain_containment_area_v20260703b.h5")
BIN_SIZE_KM = 1.0
COMMON_VOLUME_ALT_KM = 94.988
MONOSTATIC_SANYA_H5 = os.path.join("results", "sanya_monostatic_ranges_v20260613b.h5")
MONOSTATIC_SANYA_PATTERN = os.path.join("results", "head_echoes", "sanya", "sanya_*.h5")
MONOSTATIC_SANYA_VELOCITY_CSV = os.path.join("results", "sanya_range_vs_radial_velocity.csv")
SANYA_AZ_DEG = 15.0
SANYA_EL_DEG = 75.0
HEIGHT_MIN_KM = 80.0
HEIGHT_MAX_KM = 120.0
SANYA_MONOSTATIC_LOW_HEIGHT_KM = 80.0
SANYA_MONOSTATIC_HIGH_HEIGHT_KM = 120.0
SANYA_MONOSTATIC_OUTSIDE_VELOCITY_MAX_KM_S = -10.0


def add_panel_label(ax, label):
    ax.text(
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.4,
            "boxstyle": "square,pad=0.18",
        },
    )


def sanya_slant_ranges_to_heights_km(ranges_km, az_deg=SANYA_AZ_DEG, el_deg=SANYA_EL_DEG):
    altitudes_km = []
    for range_km in np.asarray(ranges_km, dtype=np.float64):
        llh = jcoord.az_el_r2geodetic(
            sc.lat0[0],
            sc.lon0[0],
            sc.alt0[0] * 1e3,
            float(az_deg),
            float(el_deg),
            float(range_km) * 1e3,
        )
        altitudes_km.append(float(llh[2] / 1e3))
    altitudes_km = np.asarray(altitudes_km, dtype=np.float64)
    return altitudes_km[np.isfinite(altitudes_km)]


def ecef_to_enu_vectors(vectors_ecef, lat_deg, lon_deg):
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=np.float64)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    vectors = np.asarray(vectors_ecef, dtype=np.float64)
    return np.stack([vectors @ east, vectors @ north, vectors @ up], axis=-1)


def unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1e-12)


def sanya_beam_points_at_heights(height_grid_km):
    range_grid_km = np.linspace(1.0, 180.0, 6000)
    llh = np.asarray(
        [
            jcoord.az_el_r2geodetic(
                sc.lat0[0],
                sc.lon0[0],
                sc.alt0[0] * 1e3,
                SANYA_AZ_DEG,
                SANYA_EL_DEG,
                float(range_km) * 1e3,
            )
            for range_km in range_grid_km
        ],
        dtype=np.float64,
    )
    beam_height_km = llh[:, 2] / 1e3
    order = np.argsort(beam_height_km)
    target_range_km = np.interp(height_grid_km, beam_height_km[order], range_grid_km[order])
    target_llh = np.asarray(
        [
            jcoord.az_el_r2geodetic(
                sc.lat0[0],
                sc.lon0[0],
                sc.alt0[0] * 1e3,
                SANYA_AZ_DEG,
                SANYA_EL_DEG,
                float(range_km) * 1e3,
            )
            for range_km in target_range_km
        ],
        dtype=np.float64,
    )
    return np.asarray([jcoord.geodetic2ecef(lat, lon, alt) for lat, lon, alt in target_llh], dtype=np.float64)


def receiver_gain_dbi_on_sanya_beam(site_index, target_ecef_m):
    site = gain_model.SITES[site_index]
    site_ecef_m = np.asarray(jcoord.geodetic2ecef(sc.lat0[site_index], sc.lon0[site_index], sc.alt0[site_index] * 1e3), dtype=np.float64)
    los_ecef = unit_rows(target_ecef_m - site_ecef_m[None, :])
    los_enu = ecef_to_enu_vectors(los_ecef, sc.lat0[site_index], sc.lon0[site_index])
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    _normal, tilt_axis, panel_cross_axis = gain_model.panel_axes(site)
    power = gain_model.aperture_power(
        los_enu,
        pointing,
        tilt_axis,
        panel_cross_axis,
        site.dim_tilt_plane_m,
        site.dim_cross_tilt_m,
    )
    relative_db = 10.0 * np.log10(np.maximum(power, 1e-10))
    steered_peak_dbi = float(gain_model.site_summary(site)["steered_peak_gain_dbi"])
    return steered_peak_dbi + relative_db


def receiver_gain_profiles(height_grid_km):
    target_ecef_m = sanya_beam_points_at_heights(height_grid_km)
    return {
        "Danzhou": receiver_gain_dbi_on_sanya_beam(1, target_ecef_m),
        "Wenchang": receiver_gain_dbi_on_sanya_beam(2, target_ecef_m),
    }


def event_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0].replace("joint_delay_doppler_fft_", "")


def joint_fit_paths(catalog_dir, fallback_catalog_dir=FALLBACK_CATALOG_DIR):
    paths = sorted(glob.glob(os.path.join(catalog_dir, "joint_delay_doppler_fft_tri_*.h5")))
    by_event = {event_id_from_path(path): path for path in paths}
    if fallback_catalog_dir:
        for path in sorted(glob.glob(os.path.join(fallback_catalog_dir, "joint_delay_doppler_fft_tri_*.h5"))):
            by_event.setdefault(event_id_from_path(path), path)
    return [by_event[event_id] for event_id in sorted(by_event)]


def collect_joint_heights(catalog_dir):
    chunks = []
    for path in joint_fit_paths(catalog_dir):
        with h5py.File(path, "r") as h:
            if "joint_fit" not in h or "alt_km" not in h["joint_fit"]:
                continue
            alt_km = np.asarray(h["joint_fit"]["alt_km"][:], dtype=np.float64)
            chunks.append(alt_km[np.isfinite(alt_km)])
    if not chunks:
        return None
    altitudes_km = np.concatenate(chunks)
    return altitudes_km[np.isfinite(altitudes_km)], 0


def collect_heights():
    joint = collect_joint_heights(INPUT_CATALOG_DIR)
    if joint is not None:
        return joint

    if os.path.exists(INPUT_H5):
        chunks = []
        with h5py.File(INPUT_H5, "r") as h:
            for event_id in h["event_id"][:]:
                name = event_id.decode("utf-8") if isinstance(event_id, bytes) else str(event_id)
                chunks.append(np.asarray(h["points"][name]["alt_km"][:], dtype=np.float64))
        if not chunks:
            raise RuntimeError(f"No fitted trajectory samples found in {INPUT_H5}")
        altitudes_km = np.concatenate(chunks)
        altitudes_km = altitudes_km[np.isfinite(altitudes_km)]
        return altitudes_km, 0

    altitudes_km = []
    n_rejected = 0
    for trajectory in build_trajectories():
        points = solve_trajectory_points(trajectory, DAN_CENTER_US, WEN_CENTER_US)
        llh = np.asarray([jcoord.ecef2geodetic(point[0], point[1], point[2]) for point in points], dtype=np.float64)
        alt_km = llh[:, 2] / 1e3
        keep = np.isfinite(llh[:, 0]) & np.isfinite(llh[:, 1]) & np.isfinite(alt_km) & (llh[:, 0] <= MAX_LAT_DEG)
        n_rejected += int((~keep).sum())
        altitudes_km.extend(alt_km[keep].tolist())
    return np.asarray(altitudes_km, dtype=np.float64), n_rejected


def collect_joint_positions(catalog_dir):
    chunks = []
    paths = joint_fit_paths(catalog_dir)
    for path in paths:
        with h5py.File(path, "r") as h:
            if "joint_fit" not in h or "x_itrs_m" not in h["joint_fit"]:
                continue
            x_itrs_m = np.asarray(h["joint_fit"]["x_itrs_m"][:], dtype=np.float64)
            finite = np.all(np.isfinite(x_itrs_m), axis=1)
            if np.any(finite):
                chunks.append(x_itrs_m[finite])
    if not chunks:
        raise RuntimeError(f"No finite joint-fit x_itrs_m positions found in {catalog_dir}")
    return np.vstack(chunks), len(paths)


def collect_joint_position_traces(catalog_dir):
    traces = []
    paths = joint_fit_paths(catalog_dir, fallback_catalog_dir=None)
    for path in paths:
        with h5py.File(path, "r") as h:
            if "joint_fit" not in h or "x_itrs_m" not in h["joint_fit"]:
                continue
            x_itrs_m = np.asarray(h["joint_fit"]["x_itrs_m"][:], dtype=np.float64)
            finite = np.all(np.isfinite(x_itrs_m), axis=1)
            if np.count_nonzero(finite) >= 2:
                traces.append(x_itrs_m[finite])
    if not traces:
        raise RuntimeError(f"No finite joint-fit x_itrs_m traces found in {catalog_dir}")
    return traces, len(paths)


def beam_panel_data(catalog_dir):
    traces_ecef_m, n_events = collect_joint_position_traces(catalog_dir)
    positions_ecef_m = np.vstack(traces_ecef_m)
    sanya_lat_deg, sanya_lon_deg, _sanya_alt_m = jcoord.ecef2geodetic(*beam_hist.gfit.LINK_TX_POSITIONS_M[0])

    site = gain_model.SITES[0]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    east_axis, north_axis = beam_hist.local_sky_axes(pointing)

    def offsets_for_positions(x_itrs_m):
        los_ecef = beam_hist.unit(x_itrs_m - beam_hist.gfit.LINK_TX_POSITIONS_M[0][None, :])
        los_enu = beam_hist.ecef_to_enu_vectors(los_ecef, sanya_lat_deg, sanya_lon_deg)
        return beam_hist.angular_offsets_deg(los_enu, pointing, east_axis, north_axis)

    east_deg, north_deg = offsets_for_positions(positions_ecef_m)
    angular_traces = []
    for trace_ecef_m in traces_ecef_m:
        trace_east_deg, trace_north_deg = offsets_for_positions(trace_ecef_m)
        finite_trace = np.isfinite(trace_east_deg) & np.isfinite(trace_north_deg)
        if np.count_nonzero(finite_trace) >= 2:
            angular_traces.append(np.column_stack([trace_east_deg[finite_trace], trace_north_deg[finite_trace]]))

    finite = np.isfinite(east_deg) & np.isfinite(north_deg)
    east_deg = east_deg[finite]
    north_deg = north_deg[finite]

    grid = np.linspace(-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_N)
    east_grid, north_grid = np.meshgrid(grid, grid)
    gain_db = beam_hist.sanya_beam_relative_gain_db(east_grid, north_grid)
    hist_range = [
        [-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG],
        [-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG],
    ]
    counts, east_edges, north_edges = np.histogram2d(
        east_deg,
        north_deg,
        bins=beam_hist.HIST_BINS,
        range=hist_range,
    )
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
        "traces": angular_traces,
    }


def load_sanya_containment_contour(path=SANYA_CONTAINMENT_AREA_H5):
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as h:
        return {
            "cutoff_relative_gain_db": float(h.attrs["cutoff_relative_gain_db"]),
            "containment_fraction": float(h.attrs["containment_fraction"]),
            "area_horizontal_km2": float(h.attrs["area_horizontal_km2"]),
        }


def collect_sanya_monostatic_heights():
    if os.path.exists(MONOSTATIC_SANYA_VELOCITY_CSV):
        table = np.genfromtxt(MONOSTATIC_SANYA_VELOCITY_CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
        height_km = np.asarray(table["height_km"], dtype=np.float64)
        radial_velocity_km_s = np.asarray(table["sanya_radial_velocity_km_s"], dtype=np.float64)
        finite = np.isfinite(height_km) & np.isfinite(radial_velocity_km_s)
        outside_height_window = (height_km < SANYA_MONOSTATIC_LOW_HEIGHT_KM) | (height_km > SANYA_MONOSTATIC_HIGH_HEIGHT_KM)
        velocity_gate = (~outside_height_window) | (
            radial_velocity_km_s < SANYA_MONOSTATIC_OUTSIDE_VELOCITY_MAX_KM_S
        )
        keep = finite & velocity_gate
        return height_km[keep], int(np.count_nonzero(finite)), int(np.count_nonzero(finite & ~velocity_gate))

    if os.path.exists(MONOSTATIC_SANYA_H5):
        with h5py.File(MONOSTATIC_SANYA_H5, "r") as h:
            # Treat Sanya monostatic detections as slant ranges and convert
            # them to altitude along the fixed Sanya transmit beam. This keeps
            # the plotted quantity height, not range, even if an old cache has
            # ambiguous naming.
            if "range_km" in h:
                heights = sanya_slant_ranges_to_heights_km(h["range_km"][()])
                return heights, int(heights.size), 0
            altitudes_km = np.asarray(h["height_km"][()], dtype=np.float64)
            altitudes_km = altitudes_km[np.isfinite(altitudes_km)]
            return altitudes_km, int(altitudes_km.size), 0

    altitudes_km = []
    for path in sorted(glob.glob(MONOSTATIC_SANYA_PATTERN)):
        with h5py.File(path, "r") as h:
            ranges_km = np.asarray(h["range_km"][()], dtype=np.float64)
            az_deg = float(h["az"][()])
            el_deg = float(h["el"][()])
        altitudes_km.extend(sanya_slant_ranges_to_heights_km(ranges_km, az_deg, el_deg).tolist())
    altitudes_km = np.asarray(altitudes_km, dtype=np.float64)
    return altitudes_km, int(altitudes_km.size), 0


def main():
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    alt_km, n_rejected = collect_heights()
    mono_alt_km, n_mono_before_velocity_gate, n_mono_velocity_rejected = collect_sanya_monostatic_heights()
    if alt_km.size == 0:
        raise RuntimeError("No retained meteor heights available for histogram.")

    alt_km_for_plot = alt_km[(alt_km >= HEIGHT_MIN_KM) & (alt_km <= HEIGHT_MAX_KM)]
    mono_alt_km_for_plot = mono_alt_km[(mono_alt_km >= HEIGHT_MIN_KM) & (mono_alt_km <= HEIGHT_MAX_KM)]
    bin_start = HEIGHT_MIN_KM
    bin_stop = HEIGHT_MAX_KM
    bins = np.arange(bin_start, bin_stop + BIN_SIZE_KM, BIN_SIZE_KM)
    gain_height_km = np.linspace(HEIGHT_MIN_KM, HEIGHT_MAX_KM, 401)
    receiver_gain_dbi = receiver_gain_profiles(gain_height_km)
    beam = beam_panel_data(BEAM_TRACE_CATALOG_DIR)
    containment = load_sanya_containment_contour()

    fig, (ax, ax_gain, ax_beam) = plt.subplots(
        1,
        3,
        figsize=(9.2, 3.35),
        gridspec_kw={"width_ratios": [1.18, 0.92, 1.08], "wspace": 0.22},
    )

    tri_counts, _ = np.histogram(alt_km_for_plot, bins=bins)
    mono_counts = np.histogram(mono_alt_km_for_plot, bins=bins)[0] if mono_alt_km_for_plot.size > 0 else None
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.barh(
        bin_centers,
        tri_counts,
        height=np.diff(bins),
        align="center",
        color="#315f72",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        label=f"Tri-static ({alt_km_for_plot.size})",
    )
    if mono_counts is not None:
        ax.stairs(
            mono_counts,
            bins,
            orientation="horizontal",
            color="#b34a2e",
            linewidth=1.8,
            label=f"Sanya monostatic ({mono_alt_km_for_plot.size})",
        )
    max_count = float(np.nanmax(tri_counts)) if tri_counts.size else 1.0
    if mono_counts is not None and mono_counts.size:
        max_count = max(max_count, float(np.nanmax(mono_counts)))
    ax.set_xlim(0.0, max(1.0, 1.08 * max_count))
    ax.set_xlabel("Count per 1 km height bin")
    ax.set_ylabel("Height (km)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_ylim(HEIGHT_MIN_KM, HEIGHT_MAX_KM)
    add_panel_label(ax, "(a)")

    ax_gain.plot(receiver_gain_dbi["Danzhou"], gain_height_km, color="#b34a2e", lw=1.8, label="Danzhou")
    ax_gain.plot(receiver_gain_dbi["Wenchang"], gain_height_km, color="#2f7d4b", lw=1.8, label="Wenchang")
    ax_gain.set_xlabel("Receiver gain on Sanya beam (dBi)")
    ax_gain.set_ylim(HEIGHT_MIN_KM, HEIGHT_MAX_KM)
    ax_gain.set_yticklabels([])
    ax_gain.set_xlim(-5.0, 43.0)
    ax_gain.grid(True, alpha=0.25)
    ax_gain.legend(loc="lower right", frameon=False)
    add_panel_label(ax_gain, "(b)")

    ax.legend(loc="upper right", frameon=False)

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
    if beam["traces"]:
        trace_collection = LineCollection(
            beam["traces"],
            colors=[(0.96, 0.96, 0.96, 0.42)],
            linewidths=0.45,
            zorder=3,
            rasterized=True,
        )
        ax_beam.add_collection(trace_collection)
    ax_beam.contour(
        beam["east_grid"],
        beam["north_grid"],
        beam["gain_db"],
        levels=[-30.0, -20.0, -13.3, -10.0, -3.0],
        colors=["0.55", "0.35", "0.20", "0.10", "0.0"],
        linewidths=[0.7, 0.8, 0.9, 1.0, 1.25],
        linestyles=[":", "--", "-.", "-", "-"],
    )
    if containment is not None:
        containment_level = containment["cutoff_relative_gain_db"]
        containment_contour = ax_beam.contour(
            beam["east_grid"],
            beam["north_grid"],
            beam["gain_db"],
            levels=[containment_level],
            colors=["#00a6d6"],
            linewidths=[1.8],
            linestyles=["-"],
        )
        ax_beam.clabel(
            containment_contour,
            fmt={containment_level: f"95% event contour ({containment_level:.1f} dB)"},
            inline=True,
            fontsize=7.5,
            colors=["#006d8f"],
        )
        ax_beam.text(
            0.03,
            0.06,
            (
                "95% event contour\n"
                f"{containment_level:.1f} dB, "
                f"{containment['area_horizontal_km2']:.1f} km$^2$"
            ),
            transform=ax_beam.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            color="#006d8f",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "boxstyle": "square,pad=0.18",
            },
        )
    ax_beam.axhline(0.0, color="0.2", lw=0.7, alpha=0.6)
    ax_beam.axvline(0.0, color="0.2", lw=0.7, alpha=0.6)
    ax_beam.set_aspect("equal", adjustable="box")
    ax_beam.set_xlim(-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG)
    ax_beam.set_ylim(-beam_hist.GRID_LIMIT_DEG, beam_hist.GRID_LIMIT_DEG)
    ax_beam.set_xlabel("Sanya beam east offset (deg)")
    ax_beam.set_ylabel("Sanya beam north offset (deg)")
    add_panel_label(ax_beam, "(c)")
    cb_beam = fig.colorbar(hist, ax=ax_beam, fraction=0.046, pad=0.03)
    cb_beam.set_label("Pulse count")

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.93, wspace=0.24)

    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    os.makedirs(os.path.dirname(PAPER_OUTPUT_PNG), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"points: {alt_km.size}")
    print(f"points within plotted height range: {alt_km_for_plot.size}")
    print("tri-static histogram weights: unit weights, one count per fitted height sample")
    print(f"rejected latitude outliers: {n_rejected}")
    print(f"height range: {np.nanmin(alt_km):.3f} to {np.nanmax(alt_km):.3f} km")
    print(f"height mean/median: {np.nanmean(alt_km):.3f} / {np.nanmedian(alt_km):.3f} km")
    print(f"common-volume beam intersection height: {COMMON_VOLUME_ALT_KM:.3f} km")
    print(f"sanya monostatic points: {mono_alt_km.size}")
    print(f"sanya monostatic points within plotted height range: {mono_alt_km_for_plot.size}")
    print("sanya monostatic histogram weights: unit weights, one count per retained height sample")
    print(f"sanya monostatic finite points before velocity gate: {n_mono_before_velocity_gate}")
    print(f"sanya monostatic velocity-gate rejected points: {n_mono_velocity_rejected}")
    if mono_alt_km.size > 0:
        print(f"sanya monostatic height range: {np.nanmin(mono_alt_km):.3f} to {np.nanmax(mono_alt_km):.3f} km")
        print(
            "sanya monostatic velocity gate: "
            f"height < {SANYA_MONOSTATIC_LOW_HEIGHT_KM:g} km or "
            f"height > {SANYA_MONOSTATIC_HIGH_HEIGHT_KM:g} km requires "
            f"radial velocity < {SANYA_MONOSTATIC_OUTSIDE_VELOCITY_MAX_KM_S:g} km/s"
        )
    print(f"bins: {bin_start:.0f} to {bin_stop:.0f} km in {BIN_SIZE_KM:.0f} km steps")
    print(INPUT_CATALOG_DIR if joint_fit_paths(INPUT_CATALOG_DIR) else (INPUT_H5 if os.path.exists(INPUT_H5) else "legacy point solver"))
    print(f"beam positions: {beam['n_positions']} from {beam['n_events']} events")
    if containment is not None:
        print(
            "sanya 95% event containment contour: "
            f"{containment['cutoff_relative_gain_db']:.3f} dB, "
            f"{containment['area_horizontal_km2']:.3f} km^2 horizontal area"
        )
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)
    print(PAPER_OUTPUT_PNG)
    print(PAPER_OUTPUT_PDF)


if __name__ == "__main__":
    main()
