import argparse
import os
import shutil
import sys
from pathlib import Path

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_article_event_fit as event_plot
import plot_memo09_antenna_gain_patterns as gain_model
import plot_sanya_beam_position_histogram as beam_plot
import noise_model


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
OUTPUT_BASE = "results/sanya_rcs_from_snr"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
ARTICLE_TABLE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/tables"
FALCON9_DIR = Path("/Users/jvi019/src/falcon9")

LIGHT_SPEED_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380_649e-23
TX_POWER_W = 4.7e6
TX_PULSE_LENGTH_S = 200e-6
NOISE_BANDWIDTH_HZ = 1.0 / TX_PULSE_LENGTH_S
SANYA_SYSTEM_TEMPERATURE_K = 120.0
MIN_RELATIVE_GAIN_DB = -3.0
COMMON_VOLUME_LAT_DEG = 18.567821
COMMON_VOLUME_LON_DEG = 109.683719
REFERENCE_TIME_UTC = np.datetime64("2024-04-22T18:00:00")


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def import_falcon9_mfp():
    sys.path.insert(0, str(FALCON9_DIR))
    import mean_free_path  # noqa: PLC0415

    return mean_free_path


def mean_free_path_profile(alt_min_km, alt_max_km, n_alt=100):
    mfp = import_falcon9_mfp()
    alt_grid_km = np.linspace(float(alt_min_km), float(alt_max_km), int(n_alt))
    lambda_m = mfp.mean_free_path_m(
        time_dt64=REFERENCE_TIME_UTC,
        lat_deg=COMMON_VOLUME_LAT_DEG,
        lon_deg=COMMON_VOLUME_LON_DEG,
        alt_km=alt_grid_km,
    )
    return alt_grid_km, np.asarray(lambda_m, dtype=np.float64)


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
    received_power_w = snr_linear * BOLTZMANN_J_K * np.asarray(system_temperature_k, dtype=np.float64) * NOISE_BANDWIDTH_HZ
    wavelength_m = gain_model.WAVELENGTH_M
    sigma_m2 = (
        received_power_w
        * (4.0 * np.pi) ** 3
        * np.asarray(ranges_m, dtype=np.float64) ** 4
        / (TX_POWER_W * np.asarray(gain_linear, dtype=np.float64) ** 2 * wavelength_m**2)
    )
    return sigma_m2


def collect_estimates(input_h5, measured_noise: noise_model.MeasuredSystemNoise | None = None, system_temperature_k=None):
    rows = []
    with h5py.File(input_h5, "r") as h:
        for event_id in decode_strings(h["event_id"][:]):
            group = h["points"][event_id]
            time_ns = np.asarray(group["time_ns"][:], dtype=np.int64)
            measured_total_paths_m = np.asarray(group["measured_total_paths_m"][:], dtype=np.float64)
            fit_itrs_m = np.asarray(group["x_itrs_m"][:], dtype=np.float64)
            fit_v_itrs_mps = np.asarray(group["v_itrs_mps"][:], dtype=np.float64)
            speed_km_s = np.asarray(group["speed_km_s"][:], dtype=np.float64)
            snr_db_all = event_plot.retained_sanya_snr_db(group, len(time_ns))
            points_itrs_m = event_plot.lfm_corrected_point_solutions(
                measured_total_paths_m,
                fit_itrs_m,
                fit_v_itrs_mps,
            )

            relative_gain_db, gain_dbi, gain_linear = sanya_gain_for_positions(points_itrs_m)
            ranges_m = np.linalg.norm(points_itrs_m - gfit.LINK_TX_POSITIONS_M[0][None, :], axis=1)
            if measured_noise is not None:
                tsys_k = measured_noise.tsys_k("Sanya", time_ns)
            else:
                tsys_k = np.full(len(time_ns), float(system_temperature_k), dtype=np.float64)
            rcs_m2 = rcs_from_sanya_snr(snr_db_all, ranges_m, gain_linear, tsys_k)
            good = (
                np.isfinite(snr_db_all)
                & np.isfinite(tsys_k)
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
                        "speed_km_s": float(speed_km_s[idx]),
                        "sanya_relative_gain_db": float(relative_gain_db[idx]),
                        "sanya_gain_dbi": float(gain_dbi[idx]),
                        "sanya_tsys_k": float(tsys_k[idx]),
                        "rcs_m2": float(rcs_m2[idx]),
                        "rcs_dbsm": float(10.0 * np.log10(rcs_m2[idx])),
                    }
                )
    if not rows:
        raise RuntimeError("No Sanya -3 dB in-beam RCS estimates were found.")
    return rows


def write_hdf5(rows, path, measured_noise: noise_model.MeasuredSystemNoise | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    fieldnames = list(rows[0].keys())
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Sanya monostatic meteor RCS estimates from SNR using measured system-noise model."
        h.attrs["source_script"] = "/Users/jvi019/src/lfm_meteor/estimate_sanya_rcs_from_snr.py"
        if measured_noise is not None:
            h.attrs["system_noise_model"] = "noise_model.MeasuredSystemNoise"
            h.attrs["system_noise_h5"] = measured_noise.path
            h.attrs["noise_model_summary"] = measured_noise.summary_text()
        else:
            h.attrs["system_noise_model"] = "fixed constant"
            h.attrs["fixed_system_temperature_k"] = float(SANYA_SYSTEM_TEMPERATURE_K)
        for key in fieldnames:
            values = [row[key] for row in rows]
            if isinstance(values[0], str):
                h[key] = np.asarray(values, dtype=string_dtype)
            elif isinstance(values[0], (int, np.integer)):
                h[key] = np.asarray(values, dtype=np.int64)
            else:
                h[key] = np.asarray(values, dtype=np.float64)


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
    tsys_k = np.asarray([row["sanya_tsys_k"] for row in rows], dtype=np.float64)
    alt_km = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
    speed_km_s = np.asarray([row["speed_km_s"] for row in rows], dtype=np.float64)
    summaries = {
        "RCS (m$^2$)": summarize(rcs_m2),
        "RCS (dBsm)": summarize(rcs_dbsm),
        "Sanya SNR (dB)": summarize(snr_db),
        r"Sanya \(T_{\mathrm{sys}}\) (K)": summarize(tsys_k),
        "Altitude (km)": summarize(alt_km),
        r"Speed (km s\(^{-1}\))": summarize(speed_km_s),
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
    speed_km_s = np.asarray([row["speed_km_s"] for row in rows], dtype=np.float64)
    n_events = len({row["event_id"] for row in rows})
    with plt.rc_context(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2), constrained_layout=True)
        bins = np.linspace(np.nanpercentile(rcs_dbsm, 1.0), np.nanpercentile(rcs_dbsm, 99.0), 34)
        axes[0, 0].hist(rcs_dbsm, bins=bins, color="#4c78a8", alpha=0.86)
        axes[0, 0].set_xlabel("Sanya RCS estimate (dBsm)")
        axes[0, 0].set_ylabel("Pulse count")
        axes[0, 0].set_title("In-beam RCS distribution")

        sc_snr = axes[0, 1].scatter(snr_db, rcs_dbsm, c=alt_km, s=10, cmap="viridis", alpha=0.55, edgecolors="none")
        axes[0, 1].set_xlabel("Sanya SNR (dB)")
        axes[0, 1].set_ylabel("Sanya RCS estimate (dBsm)")
        axes[0, 1].set_title("RCS versus SNR")
        axes[0, 1].grid(True, color="0.88", lw=0.7)
        cb = fig.colorbar(sc_snr, ax=axes[0, 1], fraction=0.046, pad=0.035)
        cb.set_label("Altitude (km)")

        sc_speed = axes[1, 0].scatter(
            speed_km_s,
            rcs_dbsm,
            c=alt_km,
            s=10,
            cmap="viridis",
            alpha=0.58,
            edgecolors="none",
        )
        axes[1, 0].set_xlabel(r"Fitted speed (km s$^{-1}$)")
        axes[1, 0].set_ylabel("Sanya RCS estimate (dBsm)")
        axes[1, 0].set_title("RCS versus speed")
        axes[1, 0].grid(True, color="0.88", lw=0.7)
        cb = fig.colorbar(sc_speed, ax=axes[1, 0], fraction=0.046, pad=0.035)
        cb.set_label("Altitude (km)")

        sc_height = axes[1, 1].scatter(
            rcs_dbsm,
            alt_km,
            c=speed_km_s,
            s=10,
            cmap="turbo",
            alpha=0.58,
            edgecolors="none",
        )
        axes[1, 1].set_xlabel("Sanya RCS estimate (dBsm)")
        axes[1, 1].set_ylabel("Altitude (km)")
        axes[1, 1].set_title("RCS versus altitude")
        axes[1, 1].grid(True, color="0.88", lw=0.7)
        cb = fig.colorbar(sc_height, ax=axes[1, 1], fraction=0.046, pad=0.035)
        cb.set_label(r"Fitted speed (km s$^{-1}$)")

        png = f"{output_base}.png"
        pdf = f"{output_base}.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
    return png, pdf


def make_article_plot(rows, output_base):
    rcs_dbsm = np.asarray([row["rcs_dbsm"] for row in rows], dtype=np.float64)
    rcs_m2 = np.asarray([row["rcs_m2"] for row in rows], dtype=np.float64)
    alt_km = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
    speed_km_s = np.asarray([row["speed_km_s"] for row in rows], dtype=np.float64)
    alt_profile_km, mfp_m = mean_free_path_profile(
        np.nanmin(alt_km) - 0.5,
        np.nanmax(alt_km) + 0.5,
    )
    bragg_wavelength_m = 0.5 * gain_model.WAVELENGTH_M

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
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.05), constrained_layout=True)
        bins = np.linspace(np.nanpercentile(rcs_dbsm, 1.0), np.nanpercentile(rcs_dbsm, 99.0), 34)
        axes[0].hist(rcs_dbsm, bins=bins, color="#4c78a8", alpha=0.88, edgecolor="white", linewidth=0.45)
        axes[0].set_xlabel("Sanya RCS estimate (dBsm)")
        axes[0].set_ylabel("Pulse count")
        axes[0].set_title("In-beam RCS distribution")
        axes[0].grid(True, axis="y", color="0.88", lw=0.7)

        sc = axes[1].scatter(
            rcs_dbsm,
            alt_km,
            c=speed_km_s,
            s=12,
            cmap="turbo",
            alpha=0.62,
            edgecolors="none",
        )
        axes[1].set_xlabel("Sanya RCS estimate (dBsm)")
        axes[1].set_ylabel("Altitude (km)")
        axes[1].set_title("RCS versus altitude")
        axes[1].grid(True, color="0.88", lw=0.7)
        cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.035)
        cb.set_label(r"Fitted speed (km s$^{-1}$)")

        mfp_ax = axes[1].twiny()
        mfp_ax.plot(mfp_m, alt_profile_km, color="0.2", lw=1.2)
        mfp_ax.axvline(bragg_wavelength_m, color="#d62728", lw=1.1, ls="--")
        mfp_ax.set_xscale("log")
        mfp_ax.set_xlim(np.nanmin(mfp_m) * 0.7, np.nanmax(mfp_m) * 1.3)
        mfp_ax.set_ylim(axes[1].get_ylim())
        mfp_ax.set_xlabel(r"Mean free path (m)")
        mfp_ax.tick_params(axis="x", labelsize=9.5, pad=2)
        mfp_ax.text(
            bragg_wavelength_m * 1.08,
            0.94,
            r"$\lambda/2$",
            transform=mfp_ax.get_xaxis_transform(),
            color="#d62728",
            fontsize=8.5,
            va="top",
            ha="left",
        )

        sc_v = axes[2].scatter(
            speed_km_s,
            alt_km,
            c=rcs_dbsm,
            s=12,
            cmap="turbo",
            alpha=0.62,
            edgecolors="none",
        )
        axes[2].set_xlabel(r"Fitted speed (km s$^{-1}$)")
        axes[2].set_ylabel("Altitude (km)")
        axes[2].set_title("Altitude versus speed")
        axes[2].grid(True, color="0.88", lw=0.7)
        cb = fig.colorbar(sc_v, ax=axes[2], fraction=0.046, pad=0.035)
        cb.set_label("Sanya RCS estimate (dBsm)")

        png = f"{output_base}_article.png"
        pdf = f"{output_base}_article.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
    return png, pdf


def make_histogram(rows, output_base):
    rcs_dbsm = np.asarray([row["rcs_dbsm"] for row in rows], dtype=np.float64)
    rcs_m2 = np.asarray([row["rcs_m2"] for row in rows], dtype=np.float64)

    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
        bins = np.linspace(np.nanpercentile(rcs_dbsm, 1.0), np.nanpercentile(rcs_dbsm, 99.0), 36)
        ax.hist(rcs_dbsm, bins=bins, color="#4c78a8", alpha=0.88, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Sanya RCS estimate (dBsm)")
        ax.set_ylabel("Pulse count")
        ax.set_title("Sanya in-beam RCS distribution")
        ax.grid(True, axis="y", color="0.88", lw=0.7)

        png = f"{output_base}_histogram.png"
        pdf = f"{output_base}_histogram.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description="Estimate Sanya monostatic meteor RCS from Sanya SNR for in-beam pulses.")
    parser.add_argument("--input", default=INPUT_H5)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--system-noise-h5", default=noise_model.SYSTEM_NOISE_H5)
    parser.add_argument("--use-fixed-system-temperature", action="store_true")
    parser.add_argument("--system-temperature-k", type=float, default=SANYA_SYSTEM_TEMPERATURE_K)
    parser.add_argument("--copy-to-article", action="store_true")
    args = parser.parse_args()

    measured_noise = None if args.use_fixed_system_temperature else noise_model.MeasuredSystemNoise(args.system_noise_h5)
    rows = collect_estimates(args.input, measured_noise=measured_noise, system_temperature_k=args.system_temperature_k)
    h5_path = f"{args.output_base}.h5"
    tex_path = f"{args.output_base}_summary.tex"
    write_hdf5(rows, h5_path, measured_noise)
    write_summary_tex(rows, tex_path)
    png, pdf = make_plot(rows, args.output_base)
    article_png, article_pdf = make_article_plot(rows, args.output_base)
    hist_png, hist_pdf = make_histogram(rows, args.output_base)

    rcs_m2 = np.asarray([row["rcs_m2"] for row in rows], dtype=np.float64)
    rcs_dbsm = np.asarray([row["rcs_dbsm"] for row in rows], dtype=np.float64)
    print(f"wrote {h5_path}")
    print(f"wrote {tex_path}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {article_png}")
    print(f"wrote {article_pdf}")
    print(f"wrote {hist_png}")
    print(f"wrote {hist_pdf}")
    print(f"n_in_3db={len(rows)}")
    print(f"n_events_in_3db={len({row['event_id'] for row in rows})}")
    print(f"rcs_m2_median={np.nanmedian(rcs_m2):.6e}")
    print(f"rcs_dbsm_median={np.nanmedian(rcs_dbsm):.3f}")
    print(f"median_tsys_k={np.nanmedian([row['sanya_tsys_k'] for row in rows]):.3f}")

    if args.copy_to_article:
        os.makedirs(ARTICLE_FIGURE_DIR, exist_ok=True)
        os.makedirs(ARTICLE_TABLE_DIR, exist_ok=True)
        for path in (png, pdf, article_png, article_pdf, hist_png, hist_pdf):
            dest = os.path.join(ARTICLE_FIGURE_DIR, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"copied {dest}")
        table_dest = os.path.join(ARTICLE_TABLE_DIR, os.path.basename(tex_path))
        shutil.copy2(tex_path, table_dest)
        print(f"copied {table_dest}")


if __name__ == "__main__":
    main()
