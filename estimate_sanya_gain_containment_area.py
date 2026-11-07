import argparse
import glob
import shutil
from pathlib import Path

import h5py
import jcoord
import numpy as np

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_memo09_antenna_gain_patterns as gain_model
from estimate_empirical_beam_area_from_tristatic_positions import local_enu_to_ecef_matrix, unit


DEFAULT_CATALOG_DIR = Path("results/tristatic")
DEFAULT_OUTPUT_BASE = Path("results/sanya_gain_containment_area_v20260703a")
DEFAULT_PAPER_MEMO_FIGURE_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper/memos/figures")
COMMON_VOLUME_ALTITUDE_KM = 95.0


def sanya_geometry():
    tx = np.asarray(gfit.LINK_TX_POSITIONS_M[0], dtype=np.float64)
    lat_deg, lon_deg, _alt_m = jcoord.ecef2geodetic(*tx)
    enu_to_ecef = local_enu_to_ecef_matrix(lat_deg, lon_deg)
    site = gain_model.SITES[0]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    normal, tilt_axis, panel_cross_axis = gain_model.panel_axes(site)
    scan_axis, cross_axis = gain_model.offset_basis(pointing, tilt_axis, panel_cross_axis)
    summary = gain_model.site_summary(site)
    return {
        "tx_ecef_m": tx,
        "enu_to_ecef": enu_to_ecef,
        "site": site,
        "pointing_enu": pointing,
        "scan_axis_enu": scan_axis,
        "cross_axis_enu": cross_axis,
        "pointing_el_deg": float(site.pointing_el_deg),
        "summary": summary,
    }


def ecef_to_enu(vectors_ecef, enu_to_ecef):
    return np.asarray(vectors_ecef, dtype=np.float64) @ enu_to_ecef


def offsets_for_positions(positions_ecef_m, geom):
    rel = np.asarray(positions_ecef_m, dtype=np.float64) - geom["tx_ecef_m"][None, :]
    directions_ecef = unit(rel)
    directions_enu = ecef_to_enu(directions_ecef, geom["enu_to_ecef"])
    denom = np.maximum(directions_enu @ geom["pointing_enu"], 1e-12)
    scan_deg = np.rad2deg(np.arctan2(directions_enu @ geom["scan_axis_enu"], denom))
    cross_deg = np.rad2deg(np.arctan2(directions_enu @ geom["cross_axis_enu"], denom))
    gain_dbi, summary = gain_model.gain_pattern_dbi(geom["site"], scan_deg, cross_deg)
    rel_gain_db = gain_dbi - float(summary["steered_peak_gain_dbi"])
    return scan_deg, cross_deg, rel_gain_db


def load_event_maxima(catalog_dir, geom):
    paths = sorted(glob.glob(str(Path(catalog_dir) / "joint_delay_doppler_fft_tri_*.h5")))
    event_id = []
    max_gain_db = []
    max_scan_deg = []
    max_cross_deg = []
    all_scan = []
    all_cross = []
    all_gain = []
    for path in paths:
        with h5py.File(path, "r") as h:
            if "joint_fit/x_itrs_m" not in h:
                continue
            x = np.asarray(h["joint_fit/x_itrs_m"][:], dtype=np.float64)
            keep = np.asarray(h["joint_fit/keep_rows"][:], dtype=bool) if "joint_fit/keep_rows" in h else np.ones(len(x), dtype=bool)
            finite = keep & np.all(np.isfinite(x), axis=1)
            if not np.any(finite):
                continue
            scan, cross, rel_gain = offsets_for_positions(x[finite], geom)
            finite_gain = np.isfinite(rel_gain)
            if not np.any(finite_gain):
                continue
            scan = scan[finite_gain]
            cross = cross[finite_gain]
            rel_gain = rel_gain[finite_gain]
            i = int(np.nanargmax(rel_gain))
            eid = h.attrs.get("event_id", Path(path).stem.replace("joint_delay_doppler_fft_", ""))
            eid = eid.decode("utf-8") if isinstance(eid, bytes) else str(eid)
            event_id.append(eid)
            max_gain_db.append(float(rel_gain[i]))
            max_scan_deg.append(float(scan[i]))
            max_cross_deg.append(float(cross[i]))
            all_scan.append(scan)
            all_cross.append(cross)
            all_gain.append(rel_gain)
    if not event_id:
        raise RuntimeError(f"No fitted positions found in {catalog_dir}")
    return {
        "event_id": np.asarray(event_id, dtype=object),
        "max_gain_db": np.asarray(max_gain_db, dtype=np.float64),
        "max_scan_deg": np.asarray(max_scan_deg, dtype=np.float64),
        "max_cross_deg": np.asarray(max_cross_deg, dtype=np.float64),
        "all_scan_deg": np.concatenate(all_scan),
        "all_cross_deg": np.concatenate(all_cross),
        "all_gain_db": np.concatenate(all_gain),
    }


def contour_area(cutoff_db, geom, altitude_km, grid_half_width_deg, grid_n):
    scan = np.linspace(-grid_half_width_deg, grid_half_width_deg, grid_n)
    cross = np.linspace(-grid_half_width_deg, grid_half_width_deg, grid_n)
    scan_grid, cross_grid = np.meshgrid(scan, cross)
    gain_dbi, summary = gain_model.gain_pattern_dbi(geom["site"], scan_grid, cross_grid)
    rel_gain = gain_dbi - float(summary["steered_peak_gain_dbi"])
    inside = rel_gain >= cutoff_db

    slant_km = float(altitude_km) / np.sin(np.deg2rad(geom["pointing_el_deg"]))
    x_edges = slant_km * np.tan(np.deg2rad(0.5 * (scan[:-1] + scan[1:])))
    y_edges = slant_km * np.tan(np.deg2rad(0.5 * (cross[:-1] + cross[1:])))
    # Use center-spacing in tangent-plane coordinates; the grid is fine enough
    # that this is stable for the small angular extent used here.
    x = slant_km * np.tan(np.deg2rad(scan))
    y = slant_km * np.tan(np.deg2rad(cross))
    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    area_perp = float(np.count_nonzero(inside) * dx * dy)
    area_horizontal = area_perp / np.sin(np.deg2rad(geom["pointing_el_deg"]))
    _ = x_edges, y_edges
    return scan_grid, cross_grid, rel_gain, inside, area_perp, area_horizontal


def write_h5(path, args, geom, events, cutoff_db, area_perp, area_horizontal):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["source_program"] = str(Path(__file__).resolve())
        h.attrs["catalog_dir"] = str(Path(args.catalog_dir).resolve())
        h.attrs["containment_fraction"] = float(args.containment_fraction)
        h.attrs["cutoff_relative_gain_db"] = float(cutoff_db)
        h.attrs["area_perpendicular_km2"] = float(area_perp)
        h.attrs["area_horizontal_km2"] = float(area_horizontal)
        h.attrs["common_volume_altitude_km"] = float(args.altitude_km)
        h.attrs["grid_half_width_deg"] = float(args.grid_half_width_deg)
        h.attrs["pointing_elevation_deg"] = float(geom["pointing_el_deg"])
        h.attrs["scan_hpbw_deg"] = float(geom["summary"]["scan_hpbw_deg"])
        h.attrs["cross_hpbw_deg"] = float(geom["summary"]["cross_hpbw_deg"])
        h.create_dataset("event_id", data=np.asarray(events["event_id"], dtype=object), dtype=string_dtype)
        for key in ("max_gain_db", "max_scan_deg", "max_cross_deg", "all_scan_deg", "all_cross_deg", "all_gain_db"):
            h.create_dataset(key, data=events[key])


def plot(output_base, events, cutoff_db, scan_grid, cross_grid, rel_gain, area_horizontal):
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    ax = axes[0]
    levels = np.r_[-40, -30, -20, -10, -6, -3]
    mesh = ax.pcolormesh(scan_grid, cross_grid, rel_gain, shading="auto", cmap="viridis", vmin=-40, vmax=0)
    cs = ax.contour(scan_grid, cross_grid, rel_gain, levels=levels, colors="white", linewidths=0.65, alpha=0.85)
    ax.contour(scan_grid, cross_grid, rel_gain, levels=[cutoff_db], colors="#ff4f4f", linewidths=1.8)
    ax.clabel(cs, fmt=lambda x: f"{x:.0f} dB", fontsize=7)
    ax.scatter(events["all_scan_deg"], events["all_cross_deg"], s=3, c="0.8", alpha=0.16, edgecolors="none")
    ax.scatter(events["max_scan_deg"], events["max_cross_deg"], s=13, c="#ffb000", alpha=0.85, edgecolors="black", linewidths=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Sanya TX scan offset (deg)")
    ax.set_ylabel("Sanya TX cross-scan offset (deg)")
    ax.set_title("Sanya TX gain containment contour")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Relative TX gain (dB)")

    ax = axes[1]
    ax.hist(events["max_gain_db"], bins=30, color="#2f7f6f", edgecolor="white", alpha=0.78)
    ax.axvline(cutoff_db, color="#b23a48", lw=2.0, label=f"95% cutoff = {cutoff_db:.1f} dB")
    ax.set_xlabel("Best point in event: relative TX gain (dB)")
    ax.set_ylabel("Events")
    ax.set_title(f"Horizontal area = {area_horizontal:.2f} km$^2$")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png, pdf


def copy_to_paper(paths):
    DEFAULT_PAPER_MEMO_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in paths:
        dst = DEFAULT_PAPER_MEMO_FIGURE_DIR / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Estimate Sanya TX gain-containment footprint area.")
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--containment-fraction", type=float, default=0.95)
    parser.add_argument("--altitude-km", type=float, default=COMMON_VOLUME_ALTITUDE_KM)
    parser.add_argument("--grid-half-width-deg", type=float, default=20.0)
    parser.add_argument("--grid-n", type=int, default=901)
    parser.add_argument("--copy-to-paper", action="store_true")
    args = parser.parse_args()

    geom = sanya_geometry()
    events = load_event_maxima(args.catalog_dir, geom)
    cutoff_db = float(np.nanquantile(events["max_gain_db"], 1.0 - float(args.containment_fraction)))
    scan_grid, cross_grid, rel_gain, _inside, area_perp, area_horizontal = contour_area(
        cutoff_db,
        geom,
        args.altitude_km,
        args.grid_half_width_deg,
        args.grid_n,
    )
    h5_path = Path(args.output_base).with_suffix(".h5")
    write_h5(h5_path, args, geom, events, cutoff_db, area_perp, area_horizontal)
    png, pdf = plot(args.output_base, events, cutoff_db, scan_grid, cross_grid, rel_gain, area_horizontal)
    copied = copy_to_paper([png, pdf]) if args.copy_to_paper else []

    print(f"n_events={len(events['event_id'])}")
    print(f"containment_fraction={args.containment_fraction:.6g}")
    print(f"cutoff_relative_gain_db={cutoff_db:.6g}")
    print(f"area_perpendicular_km2={area_perp:.6g}")
    print(f"area_horizontal_km2={area_horizontal:.6g}")
    print(f"scan_hpbw_deg={float(geom['summary']['scan_hpbw_deg']):.6g}")
    print(f"cross_hpbw_deg={float(geom['summary']['cross_hpbw_deg']):.6g}")
    print(f"output_h5={h5_path}")
    print(f"output_png={png}")
    print(f"output_pdf={pdf}")
    for path in copied:
        print(f"paper_copy={path}")


if __name__ == "__main__":
    main()
