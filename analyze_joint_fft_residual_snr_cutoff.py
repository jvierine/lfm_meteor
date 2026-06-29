import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as joint
import sanya_opts as sc
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, load_events, pair_tristatic_events


DEFAULT_CATALOG_DIR = "results/tristatic"
DEFAULT_OUTPUT_BASE = "results/joint_fft_residual_snr_cutoff_v20260618a"


def collect_ok_event_ids(catalog_dir):
    paths = sorted(glob.glob(os.path.join(catalog_dir, "joint_delay_doppler_fft_tri_*.h5")))
    event_ids = []
    for path in paths:
        try:
            with h5py.File(path, "r") as h:
                event_ids.append(str(h.attrs["event_id"]))
        except Exception:
            continue
    return event_ids


def reconstruct_snr_for_event(event_id, triplets, ref_fits, args):
    _idx, triplet = joint.choose_triplet(event_id, triplets, ref_fits)
    san_event, dan_event, wen_event = triplet
    fit0 = base.match_reference_fit(san_event, ref_fits)
    site_data = {
        "sanya": joint.load_site_h5_with_pulse(san_event.path, fit0, "sanya"),
        "danzhou": joint.load_site_h5_with_pulse(dan_event.path, fit0, "danzhou"),
        "wenchang": joint.load_site_h5_with_pulse(wen_event.path, fit0, "wenchang"),
    }
    refined = {}
    for site in joint.SITE_ORDER:
        gate, range_km, _power_db = joint.refine_site_without_doppler(
            site_data[site],
            upsample_factor=args.range_upsample_factor,
            same_mode_offset_samples=0.0,
        )
        refined[f"{site}_gate"] = gate
        refined[f"{site}_range_km"] = range_km
    refined["sanya_range_km"] = refined["sanya_range_km"] + sc.SANYA_RANGE_CORRECTION_KM
    measured, times_ns, _beijing_ns, snr_db, source_indices = base.matched_measurements_from_sites(
        san_event,
        dan_event,
        wen_event,
        site_data,
        refined,
    )
    order = np.argsort(times_ns)
    measured = measured[order]
    times_ns = times_ns[order]
    snr_db = snr_db[order]
    source_indices = source_indices[order]
    _points, keep_geo = base.triangulate_points(measured, san_event.az_deg, san_event.el_deg)
    return times_ns[keep_geo], snr_db[keep_geo], source_indices[keep_geo]


def nearest_time_rows(reference_ns, target_ns):
    reference_ns = np.asarray(reference_ns, dtype=np.int64)
    target_ns = np.asarray(target_ns, dtype=np.int64)
    out = np.full(len(target_ns), -1, dtype=np.int64)
    for idx, value in enumerate(target_ns):
        j = int(np.argmin(np.abs(reference_ns - value)))
        if abs(int(reference_ns[j]) - int(value)) <= 2_000_000:
            out[idx] = j
    return out


def robust_stats_by_threshold(snr_db, abs_resid_hz, edge_rank, thresholds):
    rows = []
    for threshold in thresholds:
        keep = snr_db >= threshold
        if np.count_nonzero(keep) == 0:
            continue
        edge_keep = keep & (edge_rank <= 2)
        mid_keep = keep & (edge_rank > 2)
        rows.append(
            {
                "threshold_snr_db": float(threshold),
                "n": int(np.count_nonzero(keep)),
                "median_abs_hz": float(np.nanmedian(abs_resid_hz[keep])),
                "p90_abs_hz": float(np.nanpercentile(abs_resid_hz[keep], 90)),
                "p95_abs_hz": float(np.nanpercentile(abs_resid_hz[keep], 95)),
                "frac_gt_2khz": float(np.nanmean(abs_resid_hz[keep] > 2000.0)),
                "edge_frac_gt_2khz": float(np.nanmean(abs_resid_hz[edge_keep] > 2000.0)) if np.any(edge_keep) else np.nan,
                "mid_frac_gt_2khz": float(np.nanmean(abs_resid_hz[mid_keep] > 2000.0)) if np.any(mid_keep) else np.nan,
            }
        )
    return rows


def write_h5(output_base, samples, threshold_rows):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_base + ".h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.create_dataset("event_id", data=np.asarray(samples["event_id"], dtype=object), dtype=string_dtype)
        for key, value in samples.items():
            if key == "event_id":
                continue
            h[key] = np.asarray(value)
        g = h.create_group("threshold_scan")
        for key in threshold_rows[0].keys():
            g[key] = np.asarray([row[key] for row in threshold_rows], dtype=np.float64)


def plot_summary(output_base, samples, threshold_rows):
    snr = np.asarray(samples["snr_db"], dtype=np.float64)
    resid = np.asarray(samples["abs_fft_residual_hz"], dtype=np.float64)
    edge_rank = np.asarray(samples["edge_rank"], dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    edge = edge_rank <= 2
    axes[0].scatter(snr[~edge], resid[~edge] / 1e3, s=8, alpha=0.22, label="interior")
    axes[0].scatter(snr[edge], resid[edge] / 1e3, s=10, alpha=0.35, label="edge")
    axes[0].axhline(2.0, color="0.2", lw=1.0, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Matched-filter SNR (dB)")
    axes[0].set_ylabel("|beat residual| (kHz)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    thresholds = np.asarray([row["threshold_snr_db"] for row in threshold_rows], dtype=np.float64)
    frac = np.asarray([row["frac_gt_2khz"] for row in threshold_rows], dtype=np.float64)
    edge_frac = np.asarray([row["edge_frac_gt_2khz"] for row in threshold_rows], dtype=np.float64)
    mid_frac = np.asarray([row["mid_frac_gt_2khz"] for row in threshold_rows], dtype=np.float64)
    axes[1].plot(thresholds, frac, label="all")
    axes[1].plot(thresholds, edge_frac, label="edge")
    axes[1].plot(thresholds, mid_frac, label="interior")
    axes[1].set_xlabel("Minimum SNR (dB)")
    axes[1].set_ylabel("Fraction |residual| > 2 kHz")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(output_base + ".png", dpi=220)
    fig.savefig(output_base + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze dechirped-FFT residuals versus matched-filter SNR.")
    parser.add_argument("--catalog-dir", default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--range-upsample-factor", type=int, default=32)
    args = parser.parse_args()

    event_ids = collect_ok_event_ids(args.catalog_dir)
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]
    ref_fits = base.load_reference_fits()
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    samples = {
        "event_id": [],
        "site_index": [],
        "row_index": [],
        "edge_rank": [],
        "snr_db": [],
        "fft_residual_hz": [],
        "abs_fft_residual_hz": [],
        "fft_prominence_db": [],
    }
    failures = []
    for count, event_id in enumerate(event_ids, start=1):
        path = os.path.join(args.catalog_dir, f"joint_delay_doppler_fft_{event_id}.h5")
        try:
            recon_time_ns, recon_snr_db, _source_indices = reconstruct_snr_for_event(event_id, triplets, ref_fits, args)
            with h5py.File(path, "r") as h:
                fit = h["joint_fit"]
                obs = h["fft_observations"]
                fit_time_ns = fit["time_ns"][:]
                row_map = nearest_time_rows(recon_time_ns, fit_time_ns)
                residual = fit["fft_residuals_hz"][:]
                keep = fit["fft_keep"][:]
                prom = obs["fft_prominence_db"][:]
                for row_idx, recon_row in enumerate(row_map):
                    if recon_row < 0:
                        continue
                    edge_rank = min(row_idx, len(row_map) - 1 - row_idx)
                    for site_idx in range(3):
                        if keep[row_idx, site_idx] and np.isfinite(residual[row_idx, site_idx]):
                            samples["event_id"].append(event_id)
                            samples["site_index"].append(site_idx)
                            samples["row_index"].append(row_idx)
                            samples["edge_rank"].append(edge_rank)
                            samples["snr_db"].append(float(recon_snr_db[recon_row, site_idx]))
                            samples["fft_residual_hz"].append(float(residual[row_idx, site_idx]))
                            samples["abs_fft_residual_hz"].append(float(abs(residual[row_idx, site_idx])))
                            samples["fft_prominence_db"].append(float(prom[row_idx, site_idx]))
            if count % 25 == 0:
                print(f"processed {count}/{len(event_ids)}", flush=True)
        except Exception as exc:
            failures.append((event_id, repr(exc)))
            print(f"failed {event_id}: {exc!r}", flush=True)

    thresholds = np.arange(8.0, 31.0, 1.0)
    threshold_rows = robust_stats_by_threshold(
        np.asarray(samples["snr_db"], dtype=np.float64),
        np.asarray(samples["abs_fft_residual_hz"], dtype=np.float64),
        np.asarray(samples["edge_rank"], dtype=np.int64),
        thresholds,
    )
    write_h5(args.output_base, samples, threshold_rows)
    plot_summary(args.output_base, samples, threshold_rows)

    print(f"n_samples={len(samples['snr_db'])}")
    print(f"n_failures={len(failures)}")
    for row in threshold_rows:
        if row["threshold_snr_db"] in (12.0, 15.0, 18.0, 20.0, 22.0, 25.0):
            print(
                "threshold={threshold_snr_db:.0f} n={n} median={median_abs_hz:.1f}Hz "
                "p90={p90_abs_hz:.1f}Hz p95={p95_abs_hz:.1f}Hz frac_gt_2khz={frac_gt_2khz:.3f} "
                "edge_gt_2khz={edge_frac_gt_2khz:.3f} mid_gt_2khz={mid_frac_gt_2khz:.3f}".format(**row)
            )
    print(f"output_h5={args.output_base}.h5")
    print(f"output_png={args.output_base}.png")


if __name__ == "__main__":
    main()
