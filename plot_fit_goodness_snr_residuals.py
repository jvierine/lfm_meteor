import argparse
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np

import plot_article_event_fit as event_plot


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
OUTPUT_BASE = "results/fit_goodness_snr_residuals"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
SNR_BIN_WIDTH_DB = 10.0
MIN_BIN_COUNT = 25


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def collect_residuals(input_h5):
    rows = []
    with h5py.File(input_h5, "r") as h:
        for event_id in decode_strings(h["event_id"][:]):
            group = h["points"][event_id]
            time_ns = np.asarray(group["time_ns"][:], dtype=np.int64)
            measured_total_paths_m = np.asarray(group["measured_total_paths_m"][:], dtype=np.float64)
            fit_gcrs_m = np.asarray(group["x_gcrs_m"][:], dtype=np.float64)
            fit_v_gcrs_mps = np.asarray(group["v_gcrs_mps"][:], dtype=np.float64)
            fit_itrs_m = np.asarray(group["x_itrs_m"][:], dtype=np.float64)
            fit_v_itrs_mps = np.asarray(group["v_itrs_mps"][:], dtype=np.float64)
            snr_db = event_plot.retained_sanya_snr_db(group, len(time_ns))

            measured_itrs_m = event_plot.lfm_corrected_point_solutions(
                measured_total_paths_m,
                fit_itrs_m,
                fit_v_itrs_mps,
            )
            measured_gcrs_m = event_plot.ecef_to_gcrs(measured_itrs_m, time_ns)

            along_axis, cross_axis = event_plot.event_axes(fit_gcrs_m, fit_v_gcrs_mps)
            origin = fit_gcrs_m[0]
            fit_along_m = (fit_gcrs_m - origin) @ along_axis
            fit_cross_m = (fit_gcrs_m - origin) @ cross_axis
            measured_along_m = (measured_gcrs_m - origin) @ along_axis
            measured_cross_m = (measured_gcrs_m - origin) @ cross_axis

            along_residual_m = measured_along_m - fit_along_m
            cross_residual_m = measured_cross_m - fit_cross_m
            good = np.isfinite(snr_db) & np.isfinite(along_residual_m) & np.isfinite(cross_residual_m)
            if np.any(good):
                rows.append(
                    np.column_stack(
                        [
                            snr_db[good],
                            along_residual_m[good],
                            cross_residual_m[good],
                        ]
                    )
                )
    if not rows:
        raise RuntimeError(f"No finite residuals found in {input_h5}")
    return np.vstack(rows)


def robust_ylim(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return (-1.0, 1.0)
    limit = float(np.nanpercentile(np.abs(finite), 99.0))
    limit = max(limit, 10.0)
    return (-1.08 * limit, 1.08 * limit)


def mean_standard_deviation(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < MIN_BIN_COUNT:
        return np.nan, np.nan
    center = float(np.nanmean(finite))
    sigma = float(np.nanstd(finite, ddof=1))
    return center, sigma


def snr_bin_summaries(snr_db, values):
    first_edge = SNR_BIN_WIDTH_DB * np.floor(np.nanmin(snr_db) / SNR_BIN_WIDTH_DB)
    last_edge = SNR_BIN_WIDTH_DB * np.ceil(np.nanmax(snr_db) / SNR_BIN_WIDTH_DB)
    edges = np.arange(first_edge, last_edge + SNR_BIN_WIDTH_DB, SNR_BIN_WIDTH_DB)
    summaries = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (snr_db >= lo) & (snr_db < hi) & np.isfinite(values)
        if np.count_nonzero(use) < MIN_BIN_COUNT:
            continue
        mean, sigma = mean_standard_deviation(values[use])
        if np.isfinite(mean) and np.isfinite(sigma):
            summaries.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "center": 0.5 * float(lo + hi),
                    "mean": mean,
                    "sigma": sigma,
                    "n": int(np.count_nonzero(use)),
                }
            )
    return summaries


def draw_snr_summaries(ax, snr_db, values, color="#b2182b"):
    summaries = snr_bin_summaries(snr_db, values)
    if not summaries:
        return summaries

    x = np.asarray([row["center"] for row in summaries], dtype=np.float64)
    y = np.asarray([row["mean"] for row in summaries], dtype=np.float64)
    sigma = np.asarray([row["sigma"] for row in summaries], dtype=np.float64)
    ax.errorbar(
        x,
        y,
        yerr=sigma,
        fmt="s",
        ms=4.2,
        mfc=color,
        mec="white",
        mew=0.6,
        ecolor=color,
        elinewidth=1.4,
        capsize=3.0,
        label=r"10 dB bin mean $\pm$ std.",
        zorder=5,
    )

    y0, y1 = ax.get_ylim()
    text_offset = 0.025 * (y1 - y0)
    for row in summaries:
        sign = 1.0 if row["mean"] >= 0.0 else -1.0
        label_y = row["mean"] + sign * (row["sigma"] + text_offset)
        label_y = float(np.clip(label_y, y0 + 0.04 * (y1 - y0), y1 - 0.04 * (y1 - y0)))
        ax.text(
            row["center"],
            label_y,
            f"{row['sigma']:.0f} m",
            ha="center",
            va="bottom" if sign > 0 else "top",
            fontsize=7.5,
            color=color,
        )
    return summaries


def make_plot(input_h5, output_base, copy_to_article=False):
    residuals = collect_residuals(input_h5)
    snr_db = residuals[:, 0]
    along_residual_m = residuals[:, 1]
    cross_residual_m = residuals[:, 2]

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
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), constrained_layout=True, sharex=True)
        panels = [
            (axes[0], along_residual_m, "Along-track residual (m)"),
            (axes[1], cross_residual_m, "Cross-track residual (m)"),
        ]
        all_summaries = {}
        for ax, values, ylabel in panels:
            ax.scatter(
                snr_db,
                values,
                s=8,
                c="#2166ac",
                alpha=0.34,
                edgecolors="none",
                rasterized=True,
            )
            ax.axhline(0.0, color="0.25", lw=0.9)
            ax.grid(True, color="0.88", lw=0.7)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(*robust_ylim(values))
            ax.set_yscale("symlog", linthresh=2.0, linscale=0.7)
            all_summaries[ylabel] = draw_snr_summaries(ax, snr_db, values)

        axes[0].set_title("Along-track")
        axes[1].set_title("Cross-track")
        axes[0].legend(loc="lower right", frameon=True, framealpha=0.86)
        axes[0].text(
            0.02,
            0.98,
            f"{len(snr_db):,} pulse positions",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
        )

        png = f"{output_base}.png"
        pdf = f"{output_base}.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"positions={len(snr_db)}")
    print(f"along_rms_m={np.sqrt(np.nanmean(along_residual_m**2)):.3f}")
    print(f"cross_rms_m={np.sqrt(np.nanmean(cross_residual_m**2)):.3f}")
    for name, summaries in all_summaries.items():
        print(name)
        for row in summaries:
            print(
                f"  {row['lo']:.0f}-{row['hi']:.0f} dB: "
                f"mean={row['mean']:.2f} m std={row['sigma']:.2f} m n={row['n']}"
            )

    if copy_to_article:
        os.makedirs(ARTICLE_FIGURE_DIR, exist_ok=True)
        for path in (png, pdf):
            dest = os.path.join(ARTICLE_FIGURE_DIR, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"copied {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot trajectory-centered position residuals against Sanya SNR for fit-goodness checks."
    )
    parser.add_argument("--input", default=INPUT_H5)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--copy-to-article", action="store_true")
    args = parser.parse_args()
    make_plot(args.input, args.output_base, copy_to_article=args.copy_to_article)


if __name__ == "__main__":
    main()
