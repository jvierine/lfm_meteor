import argparse
import csv
import os
import shutil

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_article_event_fit as event_plot
import plot_memo09_antenna_gain_patterns as gain_model
import plot_sanya_beam_position_histogram as beam_plot


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
OUTPUT_BASE = "results/sanya_rcs_from_snr"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
ARTICLE_TABLE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/tables"

LIGHT_SPEED_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380_649e-23
TX_POWER_W = 4.7e6
TX_PULSE_LENGTH_S = 200e-6
NOISE_BANDWIDTH_HZ = 1.0 / TX_PULSE_LENGTH_S
SANYA_SYSTEM_TEMPERATURE_K = 120.0
CONSERVATIVE_SYSTEM_TEMPERATURE_K = 130.0
MIN_RELATIVE_GAIN_DB = -3.0


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def sanya_gain_for_positions(points_ecef_m):
    site = gain_model.SITES[0]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    _normal, tilt_axis, panel_cross_axis = gain_model.panel_axes(site)
    summary = gain_model.site_summary(site)

    los_ecef = beam_plot.unit(np.asarray(points_ecef_m, dtype=np.float64) - gfit.LINK_TX_POSITIONS_M[0][None, :])
    san_lat, san_lon, _san_alt = jcoord.ecef2geodetic(*gfit.LINK_TX_POSITIONS_M[0])
    los_enu = beam_plot.ecef_to_enu_vectors(los_ecef, san_lat, san_lon)
    relative_power = gain_model.aperture_power(
        los_enu,
        pointing,
        tilt_axis,
        panel_cross_axis,
        site.dim_tilt_plane_m,
        site.dim_cross_tilt_m,
    )
    relative_gain_db = 10.0 * np.log10(np.maximum(relative_power, 1e-12))
    gain_dbi = float(summary["steered_peak_gain_dbi"]) + relative_gain_db
    gain_linear = 10.0 ** (gain_dbi / 10.0)
    return relative_gain_db, gain_dbi, gain_linear


def rcs_from_sanya_snr(snr_db, ranges_m, gain_linear, system_temperature_k):
    snr_linear = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)
    received_power_w = snr_linear * BOLTZMANN_J_K * float(system_temperature_k) * NOISE_BANDWIDTH_HZ
    wavelength_m = gain_model.WAVELENGTH_M
    sigma_m2 = (
        received_power_w
        * (4.0 * np.pi) ** 3
        * np.asarray(ranges_m, dtype=np.float64) ** 4
        / (TX_POWER_W * np.asarray(gain_linear, dtype=np.float64) ** 2 * wavelength_m**2)
    )
    return sigma_m2


def collect_estimates(input_h5, system_temperature_k=SANYA_SYSTEM_TEMPERATURE_K):
    rows = []
    with h5py.File(input_h5, "r") as h:
        for event_id in decode_strings(h["event_id"][:]):
            group = h["points"][event_id]
            time_ns = np.asarray(group["time_ns"][:], dtype=np.int64)
            measured_total_paths_m = np.asarray(group["measured_total_paths_m"][:], dtype=np.float64)
            fit_itrs_m = np.asarray(group["x_itrs_m"][:], dtype=np.float64)
            fit_v_itrs_mps = np.asarray(group["v_itrs_mps"][:], dtype=np.float64)
            snr_db_all = event_plot.retained_sanya_snr_db(group, len(time_ns))
            points_itrs_m = event_plot.lfm_corrected_point_solutions(
                measured_total_paths_m,
                fit_itrs_m,
                fit_v_itrs_mps,
            )

            relative_gain_db, gain_dbi, gain_linear = sanya_gain_for_positions(points_itrs_m)
            ranges_m = np.linalg.norm(points_itrs_m - gfit.LINK_TX_POSITIONS_M[0][None, :], axis=1)
            rcs_m2 = rcs_from_sanya_snr(snr_db_all, ranges_m, gain_linear, system_temperature_k)
            good = (
                np.isfinite(snr_db_all)
                & np.isfinite(relative_gain_db)
                & np.isfinite(rcs_m2)
                & (relative_gain_db >= MIN_RELATIVE_GAIN_DB)
                & (rcs_m2 > 0.0)
            )
            for idx in np.flatnonzero(good):
                lat_deg, lon_deg, alt_m = jcoord.ecef2geodetic(*points_itrs_m[idx])
                rows.append(
                    {
                        "event_id": event_id,
                        "time_ns": int(time_ns[idx]),
                        "sanya_snr_db": float(snr_db_all[idx]),
                        "sanya_range_km": float(ranges_m[idx] / 1e3),
                        "lat_deg": float(lat_deg),
                        "lon_deg": float(lon_deg),
                        "alt_km": float(alt_m / 1e3),
                        "sanya_relative_gain_db": float(relative_gain_db[idx]),
                        "sanya_gain_dbi": float(gain_dbi[idx]),
                        "rcs_m2": float(rcs_m2[idx]),
                        "rcs_dbsm": float(10.0 * np.log10(rcs_m2[idx])),
                    }
                )
    if not rows:
        raise RuntimeError("No Sanya -3 dB in-beam RCS estimates were found.")
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.nanmin(values)),
        "p25": float(np.nanpercentile(values, 25.0)),
        "median": float(np.nanmedian(values)),
        "p75": float(np.nanpercentile(values, 75.0)),
        "max": float(np.nanmax(values)),
    }


def write_summary_tex(rows, path):
    rcs_m2 = np.asarray([row["rcs_m2"] for row in rows], dtype=np.float64)
    rcs_dbsm = np.asarray([row["rcs_dbsm"] for row in rows], dtype=np.float64)
    snr_db = np.asarray([row["sanya_snr_db"] for row in rows], dtype=np.float64)
    alt_km = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
    summaries = {
        "RCS (m$^2$)": summarize(rcs_m2),
        "RCS (dBsm)": summarize(rcs_dbsm),
        "Sanya SNR (dB)": summarize(snr_db),
        "Altitude (km)": summarize(alt_km),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% Auto-generated by /Users/jvi019/src/lfm_meteor/estimate_sanya_rcs_from_snr.py\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("Quantity & Min. & 25\\% & Median & 75\\% & Max. \\\\\n")
        f.write("\\midrule\n")
        for name, stats in summaries.items():
            f.write(
                f"{name} & {stats['min']:.3g} & {stats['p25']:.3g} & "
                f"{stats['median']:.3g} & {stats['p75']:.3g} & {stats['max']:.3g} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def make_plot(rows, output_base):
    rcs_dbsm = np.asarray([row["rcs_dbsm"] for row in rows], dtype=np.float64)
    snr_db = np.asarray([row["sanya_snr_db"] for row in rows], dtype=np.float64)
    alt_km = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
    with plt.rc_context(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
        bins = np.linspace(np.nanpercentile(rcs_dbsm, 1.0), np.nanpercentile(rcs_dbsm, 99.0), 34)
        axes[0].hist(rcs_dbsm, bins=bins, color="#4c78a8", alpha=0.86)
        axes[0].axvline(np.nanmedian(rcs_dbsm), color="0.15", lw=1.2, ls="--")
        axes[0].set_xlabel("Sanya RCS estimate (dBsm)")
        axes[0].set_ylabel("Pulse count")
        axes[0].set_title("In-beam RCS distribution")
        axes[0].text(
            0.03,
            0.96,
            f"{len(rows):,} pulses inside Sanya -3 dB contour\nTsys={SANYA_SYSTEM_TEMPERATURE_K:.0f} K",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        )
        sc = axes[1].scatter(snr_db, rcs_dbsm, c=alt_km, s=10, cmap="viridis", alpha=0.55, edgecolors="none")
        axes[1].set_xlabel("Sanya SNR (dB)")
        axes[1].set_ylabel("Sanya RCS estimate (dBsm)")
        axes[1].set_title("RCS versus SNR")
        axes[1].grid(True, color="0.88", lw=0.7)
        cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.035)
        cb.set_label("Altitude (km)")
        png = f"{output_base}.png"
        pdf = f"{output_base}.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description="Estimate Sanya monostatic meteor RCS from Sanya SNR for in-beam pulses.")
    parser.add_argument("--input", default=INPUT_H5)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--system-temperature-k", type=float, default=SANYA_SYSTEM_TEMPERATURE_K)
    parser.add_argument("--copy-to-article", action="store_true")
    args = parser.parse_args()

    rows = collect_estimates(args.input, system_temperature_k=args.system_temperature_k)
    csv_path = f"{args.output_base}.csv"
    tex_path = f"{args.output_base}_summary.tex"
    write_csv(rows, csv_path)
    write_summary_tex(rows, tex_path)
    png, pdf = make_plot(rows, args.output_base)

    rcs_m2 = np.asarray([row["rcs_m2"] for row in rows], dtype=np.float64)
    rcs_dbsm = np.asarray([row["rcs_dbsm"] for row in rows], dtype=np.float64)
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"n_in_3db={len(rows)}")
    print(f"rcs_m2_median={np.nanmedian(rcs_m2):.6e}")
    print(f"rcs_dbsm_median={np.nanmedian(rcs_dbsm):.3f}")
    print(f"tsys_130k_scale={CONSERVATIVE_SYSTEM_TEMPERATURE_K / args.system_temperature_k:.6f}")

    if args.copy_to_article:
        os.makedirs(ARTICLE_FIGURE_DIR, exist_ok=True)
        os.makedirs(ARTICLE_TABLE_DIR, exist_ok=True)
        for path in (png, pdf):
            dest = os.path.join(ARTICLE_FIGURE_DIR, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"copied {dest}")
        table_dest = os.path.join(ARTICLE_TABLE_DIR, os.path.basename(tex_path))
        shutil.copy2(tex_path, table_dest)
        print(f"copied {table_dest}")


if __name__ == "__main__":
    main()
