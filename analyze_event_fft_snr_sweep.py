import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as joint
import sanya_opts as sc
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, load_events, pair_tristatic_events


DEFAULT_CATALOG_DIR = "results/tristatic"
DEFAULT_EVENT_ID = "tri_0016_1713800709264349937"
DEFAULT_OUTPUT_BASE = "results/event_fft_snr_sweep_v20260618a"


def load_event_measurement_context(event_id, range_upsample_factor):
    ref_fits = base.load_reference_fits()
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
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
            upsample_factor=range_upsample_factor,
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
    return site_data, refined, times_ns[keep_geo], snr_db[keep_geo], source_indices[keep_geo]


def nearest_time_rows(reference_ns, target_ns):
    out = np.full(len(target_ns), -1, dtype=np.int64)
    for idx, value in enumerate(target_ns):
        j = int(np.argmin(np.abs(reference_ns - value)))
        if abs(int(reference_ns[j]) - int(value)) <= 2_000_000:
            out[idx] = j
    return out


def main():
    parser = argparse.ArgumentParser(description="Recompute event FFT beats over an SNR sweep and compare to the fitted model.")
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID)
    parser.add_argument("--catalog-dir", default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--snr-min-db", type=float, default=5.0)
    parser.add_argument("--prominence-min-db", type=float, default=0.0)
    parser.add_argument("--range-upsample-factor", type=int, default=32)
    parser.add_argument("--fft-gate-upsample-factor", type=int, default=32)
    parser.add_argument("--zero-pad-factor", type=int, default=64)
    args = parser.parse_args()

    site_data, refined, recon_time_ns, snr_db, source_indices = load_event_measurement_context(
        args.event_id,
        args.range_upsample_factor,
    )
    fft_obs = joint.estimate_fft_observations(
        site_data,
        refined,
        source_indices,
        args.zero_pad_factor,
        args.snr_min_db,
        args.prominence_min_db,
        gate_upsample_factor=args.fft_gate_upsample_factor,
        center_offset_samples=0.0,
    )
    fit_path = os.path.join(args.catalog_dir, f"joint_delay_doppler_fft_{args.event_id}.h5")
    with h5py.File(fit_path, "r") as h:
        fit = h["joint_fit"]
        fit_time_ns = fit["time_ns"][:]
        row_map = nearest_time_rows(recon_time_ns, fit_time_ns)
        model = fit["model_fft_peak_hz"][:]
        fit_keep = fit["fft_keep"][:]
    rows = []
    for fit_row, recon_row in enumerate(row_map):
        if recon_row < 0:
            continue
        edge_rank = min(fit_row, len(row_map) - 1 - fit_row)
        for site_idx, site in enumerate(joint.SITE_ORDER):
            obs = fft_obs["fft_offset_hz"][recon_row, site_idx]
            if not np.isfinite(obs):
                continue
            residual = model[fit_row, site_idx] - obs
            rows.append(
                (
                    fit_row,
                    site_idx,
                    edge_rank,
                    float(snr_db[recon_row, site_idx]),
                    float(fft_obs["fft_prominence_db"][recon_row, site_idx]),
                    float(residual),
                    bool(fit_keep[fit_row, site_idx]),
                )
            )
    data = np.asarray(rows, dtype=[
        ("row", "i4"),
        ("site", "i4"),
        ("edge_rank", "i4"),
        ("snr_db", "f8"),
        ("prominence_db", "f8"),
        ("residual_hz", "f8"),
        ("catalog_keep", "?"),
    ])
    os.makedirs(os.path.dirname(args.output_base), exist_ok=True)
    with h5py.File(args.output_base + ".h5", "w") as h:
        h.attrs["event_id"] = args.event_id
        for name in data.dtype.names:
            h[name] = data[name]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    edge = data["edge_rank"] <= 2
    axes[0].scatter(data["snr_db"][~edge], np.abs(data["residual_hz"][~edge]) / 1e3, s=22, alpha=0.55, label="interior")
    axes[0].scatter(data["snr_db"][edge], np.abs(data["residual_hz"][edge]) / 1e3, s=24, alpha=0.75, label="edge")
    axes[0].axhline(2.0, color="0.2", lw=1.0, ls="--")
    axes[0].axvline(15.0, color="0.4", lw=1.0, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Matched-filter SNR (dB)")
    axes[0].set_ylabel("|model - beat| (kHz)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].scatter(data["prominence_db"][~edge], np.abs(data["residual_hz"][~edge]) / 1e3, s=22, alpha=0.55, label="interior")
    axes[1].scatter(data["prominence_db"][edge], np.abs(data["residual_hz"][edge]) / 1e3, s=24, alpha=0.75, label="edge")
    axes[1].axhline(2.0, color="0.2", lw=1.0, ls="--")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("FFT peak prominence (dB)")
    axes[1].set_ylabel("|model - beat| (kHz)")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(args.event_id)
    fig.savefig(args.output_base + ".png", dpi=220)
    fig.savefig(args.output_base + ".pdf")
    plt.close(fig)

    print(f"event_id={args.event_id}")
    print(f"n_fft={len(data)}")
    for threshold in (5, 8, 10, 12, 15, 18, 20, 22, 25):
        keep = data["snr_db"] >= threshold
        if np.any(keep):
            print(
                f"snr_min={threshold} n={np.count_nonzero(keep)} "
                f"median_abs_hz={np.nanmedian(np.abs(data['residual_hz'][keep])):.1f} "
                f"p90_abs_hz={np.nanpercentile(np.abs(data['residual_hz'][keep]),90):.1f} "
                f"frac_gt_2khz={np.nanmean(np.abs(data['residual_hz'][keep])>2000):.3f}"
            )
    print(f"output_h5={args.output_base}.h5")
    print(f"output_png={args.output_base}.png")


if __name__ == "__main__":
    main()
