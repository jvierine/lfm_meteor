import argparse
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
import noise_model


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
OUTPUT_BASE = "results/tristatic_rcs_link_comparison"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
ARTICLE_TABLE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/tables"

BOLTZMANN_J_K = 1.380_649e-23
TX_POWER_W = 4.7e6
TX_PULSE_LENGTH_S = 200e-6
NOISE_BANDWIDTH_HZ = 1.0 / TX_PULSE_LENGTH_S
SYSTEM_TEMPERATURE_K = np.asarray([120.0, 130.0, 130.0], dtype=np.float64)
MIN_RELATIVE_GAIN_DB = -3.0
LINK_KEYS = ("sanya", "danzhou", "wenchang")
LINK_LABELS = ("Sanya", "Danzhou", "Wenchang")
PAIR_SPECS = ((0, 1), (0, 2), (2, 1))


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def station_gain_for_positions(points_ecef_m, station_index):
    site = gain_model.SITES[station_index]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    _normal, tilt_axis, panel_cross_axis = gain_model.panel_axes(site)
    summary = gain_model.site_summary(site)
    station_ecef_m = gfit.LINK_RX_POSITIONS_M[station_index]
    site_lat, site_lon, _site_alt = jcoord.ecef2geodetic(*station_ecef_m)

    los_ecef = beam_plot.unit(np.asarray(points_ecef_m, dtype=np.float64) - station_ecef_m[None, :])
    los_enu = beam_plot.ecef_to_enu_vectors(los_ecef, site_lat, site_lon)
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


def rcs_from_link_snr(snr_db, tx_range_m, rx_range_m, tx_gain_linear, rx_gain_linear, system_temperature_k):
    snr_linear = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)
    received_power_w = snr_linear * BOLTZMANN_J_K * np.asarray(system_temperature_k, dtype=np.float64) * NOISE_BANDWIDTH_HZ
    wavelength_m = gain_model.WAVELENGTH_M
    sigma_m2 = (
        received_power_w
        * (4.0 * np.pi) ** 3
        * np.asarray(tx_range_m, dtype=np.float64) ** 2
        * np.asarray(rx_range_m, dtype=np.float64) ** 2
        / (
            TX_POWER_W
            * np.asarray(tx_gain_linear, dtype=np.float64)
            * np.asarray(rx_gain_linear, dtype=np.float64)
            * wavelength_m**2
        )
    )
    return sigma_m2


def bragg_wave_vectors(points_ecef_m, receiver_ecef_m):
    incident_hat = beam_plot.unit(np.asarray(points_ecef_m, dtype=np.float64) - gfit.LINK_TX_POSITIONS_M[0][None, :])
    scattered_hat = beam_plot.unit(np.asarray(receiver_ecef_m, dtype=np.float64)[None, :] - points_ecef_m)
    wave_number = 2.0 * np.pi / gain_model.WAVELENGTH_M
    return wave_number * (scattered_hat - incident_hat)


def bragg_wave_vector_length(points_ecef_m, receiver_ecef_m):
    return np.linalg.norm(bragg_wave_vectors(points_ecef_m, receiver_ecef_m), axis=1)


def aspect_angle_deg(bragg_vectors_m_inv, velocity_ecef_mps):
    bragg_hat = beam_plot.unit(np.asarray(bragg_vectors_m_inv, dtype=np.float64))
    velocity_hat = beam_plot.unit(np.asarray(velocity_ecef_mps, dtype=np.float64))
    dot = np.sum(bragg_hat * velocity_hat, axis=1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def retained_station_snr_db(group, n_points):
    snr = np.asarray(group["snr_db"][:], dtype=np.float64)
    if snr.ndim == 1:
        snr = snr[:, None]
    if snr.shape[0] == n_points:
        return snr
    out = np.full((n_points, snr.shape[1]), np.nan, dtype=np.float64)
    n_copy = min(n_points, snr.shape[0])
    out[:n_copy, :] = snr[:n_copy, :]
    return out


def collect_estimates(input_h5, measured_noise: noise_model.MeasuredSystemNoise | None = None):
    rows = []
    with h5py.File(input_h5, "r") as h:
        for event_id in decode_strings(h["event_id"][:]):
            group = h["points"][event_id]
            time_ns = np.asarray(group["time_ns"][:], dtype=np.int64)
            measured_total_paths_m = np.asarray(group["measured_total_paths_m"][:], dtype=np.float64)
            fit_itrs_m = np.asarray(group["x_itrs_m"][:], dtype=np.float64)
            fit_v_itrs_mps = np.asarray(group["v_itrs_mps"][:], dtype=np.float64)
            snr_db = retained_station_snr_db(group, len(time_ns))
            points_itrs_m = event_plot.lfm_corrected_point_solutions(
                measured_total_paths_m,
                fit_itrs_m,
                fit_v_itrs_mps,
            )

            tx_relative_gain_db, tx_gain_dbi, tx_gain_linear = station_gain_for_positions(points_itrs_m, 0)
            tx_range_m = np.linalg.norm(points_itrs_m - gfit.LINK_TX_POSITIONS_M[0][None, :], axis=1)
            station_relative_gain_db = []
            station_gain_dbi = []
            station_gain_linear = []
            station_range_m = []
            station_bragg_k_m = []
            station_aspect_angle_deg = []
            station_tsys_k = []
            for station_index in range(3):
                rel_db, gain_dbi, gain_linear = station_gain_for_positions(points_itrs_m, station_index)
                bragg_vectors = bragg_wave_vectors(points_itrs_m, gfit.LINK_RX_POSITIONS_M[station_index])
                station_relative_gain_db.append(rel_db)
                station_gain_dbi.append(gain_dbi)
                station_gain_linear.append(gain_linear)
                station_range_m.append(
                    np.linalg.norm(points_itrs_m - gfit.LINK_RX_POSITIONS_M[station_index][None, :], axis=1)
                )
                station_bragg_k_m.append(np.linalg.norm(bragg_vectors, axis=1))
                station_aspect_angle_deg.append(aspect_angle_deg(bragg_vectors, fit_v_itrs_mps))
                if measured_noise is not None:
                    station_tsys_k.append(measured_noise.tsys_k(LINK_LABELS[station_index], time_ns))
                else:
                    station_tsys_k.append(np.full(len(time_ns), SYSTEM_TEMPERATURE_K[station_index], dtype=np.float64))

            station_relative_gain_db = np.asarray(station_relative_gain_db, dtype=np.float64).T
            station_gain_dbi = np.asarray(station_gain_dbi, dtype=np.float64).T
            station_gain_linear = np.asarray(station_gain_linear, dtype=np.float64).T
            station_range_m = np.asarray(station_range_m, dtype=np.float64).T
            station_bragg_k_m = np.asarray(station_bragg_k_m, dtype=np.float64).T
            station_aspect_angle_deg = np.asarray(station_aspect_angle_deg, dtype=np.float64).T
            station_tsys_k = np.asarray(station_tsys_k, dtype=np.float64).T

            rcs_m2 = np.full_like(snr_db, np.nan, dtype=np.float64)
            for link_index in range(3):
                rcs_m2[:, link_index] = rcs_from_link_snr(
                    snr_db[:, link_index],
                    tx_range_m,
                    station_range_m[:, link_index],
                    tx_gain_linear,
                    station_gain_linear[:, link_index],
                    station_tsys_k[:, link_index],
                )

            good = (
                np.isfinite(snr_db)
                & np.isfinite(station_tsys_k)
                & np.isfinite(rcs_m2)
                & (rcs_m2 > 0.0)
                & np.isfinite(tx_relative_gain_db[:, None])
                & np.isfinite(station_relative_gain_db)
                & (tx_relative_gain_db[:, None] >= MIN_RELATIVE_GAIN_DB)
                & (station_relative_gain_db >= MIN_RELATIVE_GAIN_DB)
            )

            for pulse_index in range(len(time_ns)):
                lat_deg, lon_deg, alt_m = jcoord.ecef2geodetic(*points_itrs_m[pulse_index])
                row = {
                    "event_id": event_id,
                    "time_ns": int(time_ns[pulse_index]),
                    "lat_deg": float(lat_deg),
                    "lon_deg": float(lon_deg),
                    "alt_km": float(alt_m / 1e3),
                    "sanya_tx_relative_gain_db": float(tx_relative_gain_db[pulse_index]),
                }
                any_good = False
                for link_index, key in enumerate(LINK_KEYS):
                    if good[pulse_index, link_index]:
                        any_good = True
                        row[f"{key}_snr_db"] = float(snr_db[pulse_index, link_index])
                        row[f"{key}_range_km"] = float(station_range_m[pulse_index, link_index] / 1e3)
                        row[f"{key}_relative_gain_db"] = float(station_relative_gain_db[pulse_index, link_index])
                        row[f"{key}_gain_dbi"] = float(station_gain_dbi[pulse_index, link_index])
                        row[f"{key}_tsys_k"] = float(station_tsys_k[pulse_index, link_index])
                        row[f"{key}_bragg_k_m_inv"] = float(station_bragg_k_m[pulse_index, link_index])
                        row[f"{key}_aspect_angle_deg"] = float(station_aspect_angle_deg[pulse_index, link_index])
                        row[f"{key}_rcs_m2"] = float(rcs_m2[pulse_index, link_index])
                        row[f"{key}_rcs_dbsm"] = float(10.0 * np.log10(rcs_m2[pulse_index, link_index]))
                    else:
                        row[f"{key}_snr_db"] = np.nan
                        row[f"{key}_range_km"] = np.nan
                        row[f"{key}_relative_gain_db"] = np.nan
                        row[f"{key}_gain_dbi"] = np.nan
                        row[f"{key}_tsys_k"] = np.nan
                        row[f"{key}_bragg_k_m_inv"] = np.nan
                        row[f"{key}_aspect_angle_deg"] = np.nan
                        row[f"{key}_rcs_m2"] = np.nan
                        row[f"{key}_rcs_dbsm"] = np.nan
                if any_good:
                    rows.append(row)

    if not rows:
        raise RuntimeError("No overlapping in-beam RCS estimates were found.")
    return rows


def add_danzhou_median_normalized_rcs(rows):
    by_event = {}
    for row in rows:
        value = row["danzhou_rcs_dbsm"]
        if np.isfinite(value):
            by_event.setdefault(row["event_id"], []).append(value)

    event_medians = {event_id: float(np.nanmedian(values)) for event_id, values in by_event.items()}
    for row in rows:
        event_median = event_medians.get(row["event_id"], np.nan)
        row["event_danzhou_median_rcs_dbsm"] = event_median
        for key in LINK_KEYS:
            value = row[f"{key}_rcs_dbsm"]
            row[f"{key}_rcs_danzhou_normalized_db"] = (
                float(value - event_median) if np.isfinite(value) and np.isfinite(event_median) else np.nan
            )
    return rows


def finite_pair(rows, x_index, y_index):
    x_key = f"{LINK_KEYS[x_index]}_rcs_danzhou_normalized_db"
    y_key = f"{LINK_KEYS[y_index]}_rcs_danzhou_normalized_db"
    x = np.asarray([row[x_key] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_key] for row in rows], dtype=np.float64)
    alt = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good], alt[good]


def finite_pair_with_bragg(rows, x_index, y_index):
    x_key = f"{LINK_KEYS[x_index]}_rcs_danzhou_normalized_db"
    y_key = f"{LINK_KEYS[y_index]}_rcs_danzhou_normalized_db"
    k_key = f"{LINK_KEYS[y_index]}_bragg_k_m_inv"
    x = np.asarray([row[x_key] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_key] for row in rows], dtype=np.float64)
    bragg_k = np.asarray([row[k_key] for row in rows], dtype=np.float64)
    alt = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(bragg_k)
    return bragg_k[good], (y - x)[good], alt[good]


def perpendicular_closeness_deg(aspect_angle_deg_values):
    return 90.0 - np.abs(np.asarray(aspect_angle_deg_values, dtype=np.float64) - 90.0)


def finite_pair_with_perpendicularity_difference(rows, x_index, y_index):
    x_rcs_key = f"{LINK_KEYS[x_index]}_rcs_danzhou_normalized_db"
    y_rcs_key = f"{LINK_KEYS[y_index]}_rcs_danzhou_normalized_db"
    x_alpha_key = f"{LINK_KEYS[x_index]}_aspect_angle_deg"
    y_alpha_key = f"{LINK_KEYS[y_index]}_aspect_angle_deg"
    x_rcs = np.asarray([row[x_rcs_key] for row in rows], dtype=np.float64)
    y_rcs = np.asarray([row[y_rcs_key] for row in rows], dtype=np.float64)
    x_alpha = np.asarray([row[x_alpha_key] for row in rows], dtype=np.float64)
    y_alpha = np.asarray([row[y_alpha_key] for row in rows], dtype=np.float64)
    alt = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
    good = np.isfinite(x_rcs) & np.isfinite(y_rcs) & np.isfinite(x_alpha) & np.isfinite(y_alpha)
    x_perpendicularity = perpendicular_closeness_deg(x_alpha[good])
    y_perpendicularity = perpendicular_closeness_deg(y_alpha[good])
    return (y_perpendicularity - x_perpendicularity), (y_rcs[good] - x_rcs[good]), alt[good]


def pair_stats(rows):
    stats = []
    for x_index, y_index in PAIR_SPECS:
        x, y, _alt = finite_pair(rows, x_index, y_index)
        diff = y - x
        stats.append(
            {
                "pair": f"{LINK_LABELS[y_index]} - {LINK_LABELS[x_index]}",
                "n": int(len(diff)),
                "median_difference_db": float(np.nanmedian(diff)) if len(diff) else np.nan,
                "p16_difference_db": float(np.nanpercentile(diff, 16.0)) if len(diff) else np.nan,
                "p84_difference_db": float(np.nanpercentile(diff, 84.0)) if len(diff) else np.nan,
                "rms_difference_db": float(np.sqrt(np.nanmean(diff**2))) if len(diff) else np.nan,
            }
        )
    return stats


def write_hdf5(rows, path, measured_noise: noise_model.MeasuredSystemNoise | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    fieldnames = list(rows[0].keys())
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Tri-static link-by-link meteor RCS estimates from SNR."
        h.attrs["source_script"] = "/Users/jvi019/src/lfm_meteor/estimate_tristatic_rcs_from_snr.py"
        if measured_noise is not None:
            h.attrs["system_noise_model"] = "noise_model.MeasuredSystemNoise"
            h.attrs["system_noise_h5"] = measured_noise.path
            h.attrs["noise_model_summary"] = measured_noise.summary_text()
        else:
            h.attrs["system_noise_model"] = "fixed constants"
            h.attrs["fixed_system_temperature_k"] = SYSTEM_TEMPERATURE_K
        for key in fieldnames:
            values = [row[key] for row in rows]
            if isinstance(values[0], str):
                h[key] = np.asarray(values, dtype=string_dtype)
            elif isinstance(values[0], (int, np.integer)):
                h[key] = np.asarray(values, dtype=np.int64)
            else:
                h[key] = np.asarray(values, dtype=np.float64)


def write_summary_tex(stats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% Auto-generated by /Users/jvi019/src/lfm_meteor/estimate_tristatic_rcs_from_snr.py\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("Comparison & Pulses & Median $\\Delta$ & 16\\% $\\Delta$ & 84\\% $\\Delta$ & RMS $\\Delta$ \\\\\n")
        f.write("\\midrule\n")
        for row in stats:
            f.write(
                f"{row['pair']} & {row['n']} & {row['median_difference_db']:.2f} & "
                f"{row['p16_difference_db']:.2f} & {row['p84_difference_db']:.2f} & "
                f"{row['rms_difference_db']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def make_plot(rows, output_base):
    all_values = []
    for link_key in LINK_KEYS:
        values = np.asarray([row[f"{link_key}_rcs_danzhou_normalized_db"] for row in rows], dtype=np.float64)
        all_values.append(values[np.isfinite(values)])
    all_values = np.concatenate(all_values)
    lo = float(np.nanpercentile(all_values, 1.0))
    hi = float(np.nanpercentile(all_values, 99.0))
    pad = 0.08 * (hi - lo)
    limits = (lo - pad, hi + pad)

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
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), sharex=True, sharey=True, constrained_layout=True)
        for ax, (x_index, y_index) in zip(axes, PAIR_SPECS):
            x, y, alt = finite_pair(rows, x_index, y_index)
            diff = y - x
            sc = ax.scatter(x, y, c=alt, s=12, cmap="viridis", alpha=0.6, edgecolors="none")
            ax.plot(limits, limits, color="0.2", lw=1.0, ls="--")
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="0.88", lw=0.7)
            ax.set_title(f"{LINK_LABELS[y_index]} vs {LINK_LABELS[x_index]}")
            ax.set_xlabel(f"{LINK_LABELS[x_index]} norm. RCS (dB)")
            ax.text(
                0.04,
                0.96,
                f"n={len(x):,}\nmedian Δ={np.nanmedian(diff):.1f} dB",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
            )
        axes[0].set_ylabel("Comparison-link norm. RCS (dB)")
        cb = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.015)
        cb.set_label("Altitude (km)")

        png = f"{output_base}.png"
        pdf = f"{output_base}.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png)
        fig.savefig(pdf)
        plt.close(fig)
    return png, pdf


def make_aspect_plot(rows, output_base):
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
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), sharey=True, constrained_layout=True)
        mappable = None
        for ax, (x_index, y_index) in zip(axes, PAIR_SPECS):
            delta_perpendicularity, delta_rcs, alt = finite_pair_with_perpendicularity_difference(
                rows, x_index, y_index
            )
            mappable = ax.scatter(
                delta_perpendicularity,
                delta_rcs,
                c=alt,
                s=12,
                cmap="viridis",
                alpha=0.62,
                edgecolors="none",
            )
            ax.axhline(0.0, color="0.2", lw=1.0, ls="--")
            ax.axvline(0.0, color="0.2", lw=0.8, ls=":")
            ax.grid(True, color="0.88", lw=0.7)
            ax.set_title(f"{LINK_LABELS[y_index]} - {LINK_LABELS[x_index]}")
            ax.set_xlabel(r"$\Delta q_{90}$ (deg)")
            corr = (
                np.corrcoef(delta_perpendicularity, delta_rcs)[0, 1]
                if len(delta_perpendicularity) > 1
                else np.nan
            )
            ax.text(
                0.04,
                0.96,
                f"n={len(delta_rcs):,}\nr={corr:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
            )
        axes[0].set_ylabel(r"$\Delta$ normalized RCS (dB)")
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.015)
            cb.set_label("Altitude (km)")

        png = f"{output_base}_aspect.png"
        pdf = f"{output_base}_aspect.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png)
        fig.savefig(pdf)
        plt.close(fig)
    return png, pdf


def make_rcs_vs_aspect_plot(rows, output_base):
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
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), sharex=True, sharey=True, constrained_layout=True)
        mappable = None
        for ax, link_key, link_label in zip(axes, LINK_KEYS, LINK_LABELS):
            aspect = np.asarray([row[f"{link_key}_aspect_angle_deg"] for row in rows], dtype=np.float64)
            rcs = np.asarray([row[f"{link_key}_rcs_danzhou_normalized_db"] for row in rows], dtype=np.float64)
            alt = np.asarray([row["alt_km"] for row in rows], dtype=np.float64)
            good = np.isfinite(aspect) & np.isfinite(rcs)
            aspect = aspect[good]
            rcs = rcs[good]
            alt = alt[good]
            mappable = ax.scatter(aspect, rcs, c=alt, s=12, cmap="viridis", alpha=0.62, edgecolors="none")
            ax.axvline(90.0, color="0.2", lw=1.0, ls="--")
            ax.grid(True, color="0.88", lw=0.7)
            ax.set_title(link_label)
            ax.set_xlabel(r"$\alpha$ (deg)")
            ax.text(
                0.04,
                0.96,
                f"n={len(rcs):,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
            )
        axes[0].set_ylabel("RCS - median Danzhou (dB)")
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.015)
            cb.set_label("Altitude (km)")

        png = f"{output_base}_rcs_vs_aspect.png"
        pdf = f"{output_base}_rcs_vs_aspect.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png)
        fig.savefig(pdf)
        plt.close(fig)
    return png, pdf


def make_bragg_plot(rows, output_base):
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
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), sharey=True, constrained_layout=True)
        mappable = None
        for ax, (x_index, y_index) in zip(axes, PAIR_SPECS):
            bragg_k, diff, alt = finite_pair_with_bragg(rows, x_index, y_index)
            mappable = ax.scatter(bragg_k, diff, c=alt, s=12, cmap="viridis", alpha=0.62, edgecolors="none")
            ax.axhline(0.0, color="0.2", lw=1.0, ls="--")
            ax.grid(True, color="0.88", lw=0.7)
            ax.set_title(f"{LINK_LABELS[y_index]} - {LINK_LABELS[x_index]}")
            ax.set_xlabel(rf"{LINK_LABELS[y_index]} $|\mathbf{{K}}_B|$ (m$^{{-1}}$)")
            ax.text(
                0.04,
                0.96,
                f"n={len(diff):,}\nmedian Δ={np.nanmedian(diff):.1f} dB",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
            )
        axes[0].set_ylabel("Normalized RCS difference (dB)")
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.015)
            cb.set_label("Altitude (km)")

        png = f"{output_base}_bragg.png"
        pdf = f"{output_base}_bragg.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png)
        fig.savefig(pdf)
        plt.close(fig)
    return png, pdf


def copy_to_article(paths):
    for path in paths:
        if path.endswith("_summary.tex"):
            destination_dir = ARTICLE_TABLE_DIR
        else:
            destination_dir = ARTICLE_FIGURE_DIR
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(destination_dir, os.path.basename(path))
        shutil.copy2(path, destination)
        print(f"copied {destination}")


def main():
    parser = argparse.ArgumentParser(description="Estimate station-by-station meteor RCS from tri-static SNR.")
    parser.add_argument("--input", default=INPUT_H5)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--system-noise-h5", default=noise_model.SYSTEM_NOISE_H5)
    parser.add_argument("--use-fixed-system-temperature", action="store_true")
    parser.add_argument("--copy-to-article", action="store_true")
    args = parser.parse_args()

    measured_noise = None if args.use_fixed_system_temperature else noise_model.MeasuredSystemNoise(args.system_noise_h5)
    rows = add_danzhou_median_normalized_rcs(collect_estimates(args.input, measured_noise=measured_noise))
    stats = pair_stats(rows)
    h5_path = f"{args.output_base}.h5"
    tex_path = f"{args.output_base}_summary.tex"
    write_hdf5(rows, h5_path, measured_noise)
    write_summary_tex(stats, tex_path)
    png, pdf = make_plot(rows, args.output_base)
    bragg_png, bragg_pdf = make_bragg_plot(rows, args.output_base)
    aspect_png, aspect_pdf = make_aspect_plot(rows, args.output_base)
    rcs_aspect_png, rcs_aspect_pdf = make_rcs_vs_aspect_plot(rows, args.output_base)

    print(f"wrote {h5_path}")
    print(f"wrote {tex_path}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {bragg_png}")
    print(f"wrote {bragg_pdf}")
    print(f"wrote {aspect_png}")
    print(f"wrote {aspect_pdf}")
    print(f"wrote {rcs_aspect_png}")
    print(f"wrote {rcs_aspect_pdf}")
    for row in stats:
        print(
            f"{row['pair']}: n={row['n']} median_delta_db={row['median_difference_db']:.3f} "
            f"rms_delta_db={row['rms_difference_db']:.3f}"
        )
    if args.copy_to_article:
        copy_to_article(
            [png, pdf, bragg_png, bragg_pdf, aspect_png, aspect_pdf, rcs_aspect_png, rcs_aspect_pdf, tex_path]
        )


if __name__ == "__main__":
    main()
