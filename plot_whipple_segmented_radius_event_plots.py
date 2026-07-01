import argparse
import concurrent.futures
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import h5py
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as fit
from fit_segmented_radius_catalog_from_whipple import event_id_from_path
from fit_whipple_jacchia_catalog_from_h5 import load_group


DEFAULT_SOURCE_DIR = Path("results/tristatic_student_t_bootstrap_orbit100_20260630")
DEFAULT_WHIPPLE_DIR = Path("results/tristatic_whipple_jacchia_bootstrap_orbit100_20260701")
DEFAULT_SEGMENTED_DIR = Path("results/tristatic_segmented_radius_from_whipple_20260701")
DEFAULT_OUTPUT_DIR = Path("results/tristatic_whipple_jacchia_segmented_radius_event_plots_20260701")
EVENT_PREFIX = "joint_delay_doppler_fft_"


def inject_segmented_radius(joint_fit, segmented_h5):
    segmented_h5 = Path(segmented_h5)
    if not segmented_h5.exists():
        joint_fit["segmented_radius_available"] = False
        return joint_fit
    with h5py.File(segmented_h5, "r") as h:
        n_segments = int(h.attrs["best_n_segments"])
        group = h[f"segments_{n_segments}"]
        joint_fit["segmented_radius_available"] = True
        joint_fit["segmented_radius_best_n_segments"] = n_segments
        joint_fit["segmented_radius_best_bic"] = float(h.attrs.get("best_bic", np.nan))
        joint_fit["segmented_radius_segment_start_indices"] = np.asarray(
            group["segment_start_indices"][()],
            dtype=np.int64,
        )
        joint_fit["segmented_radius_segment_initial_radius_m"] = np.asarray(
            group["segment_initial_radius_m"][()],
            dtype=np.float64,
        )
    return joint_fit


def make_delay_fit(source_joint):
    return {
        "params": np.asarray(source_joint.get("params", np.full(7, np.nan)), dtype=np.float64),
        "rms_total_path_residual_m": float(source_joint.get("rms_total_path_residual_m", np.nan)),
        "weighted_rms": float(source_joint.get("weighted_rms", np.nan)),
        "initial_radius_m": float(source_joint.get("initial_radius_m", np.nan)),
        "initial_mass_kg": float(source_joint.get("initial_mass_kg", np.nan)),
    }


def plot_one(whipple_h5, source_dir, segmented_dir, output_dir, overwrite):
    whipple_h5 = Path(whipple_h5)
    event_id = event_id_from_path(whipple_h5)
    source_h5 = Path(source_dir) / whipple_h5.name
    segmented_h5 = Path(segmented_dir) / f"segmented_radius_{event_id}.h5"
    output_base = Path(output_dir) / whipple_h5.stem
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    if png_path.exists() and pdf_path.exists() and not overwrite:
        return event_id, "exists", str(png_path)

    with h5py.File(source_h5, "r") as h:
        source_joint = load_group(h["joint_fit"])
        source_fft = load_group(h["fft_observations"])
    with h5py.File(whipple_h5, "r") as h:
        joint_fit = load_group(h["joint_fit"])

    joint_fit = inject_segmented_radius(joint_fit, segmented_h5)
    n_params = len(np.asarray(joint_fit.get("params", []), dtype=np.float64))
    if n_params:
        joint_fit["parameter_covariance"] = np.asarray(
            joint_fit.get("parameter_covariance", np.full((n_params, n_params), np.nan)),
            dtype=np.float64,
        )
        if joint_fit["parameter_covariance"].shape != (n_params, n_params):
            joint_fit["parameter_covariance"] = np.full((n_params, n_params), np.nan)
        joint_fit["parameter_std"] = np.asarray(
            joint_fit.get("parameter_std", np.full(n_params, np.nan)),
            dtype=np.float64,
        )
    delay_fit = make_delay_fit(source_joint)
    rho_of_alt_m, _meta = base.density_interpolator(
        np.asarray(joint_fit["time_ns"], dtype=np.int64),
        np.asarray(joint_fit["x_itrs_m"], dtype=np.float64),
    )
    os.makedirs(output_dir, exist_ok=True)
    fit.plot_joint_fit(
        event_id,
        delay_fit,
        joint_fit,
        str(output_base),
        rho_of_alt_m,
        snr_db=source_fft.get("fft_snr_db"),
    )
    return event_id, "ok", str(png_path)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Whipple-Jacchia event plots with BIC-selected segmented-radius r0 annotations."
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--whipple-dir", default=DEFAULT_WHIPPLE_DIR)
    parser.add_argument("--segmented-dir", default=DEFAULT_SEGMENTED_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    whipple_paths = sorted(Path(args.whipple_dir).glob(f"{EVENT_PREFIX}tri_*.h5"))
    if args.event_id:
        wanted = set(args.event_id)
        whipple_paths = [p for p in whipple_paths if event_id_from_path(p) in wanted]
    if not whipple_paths:
        raise SystemExit("No Whipple-Jacchia event HDF5 files matched.")

    rows = []
    if args.jobs <= 1:
        for idx, path in enumerate(whipple_paths, 1):
            row = plot_one(path, args.source_dir, args.segmented_dir, args.output_dir, args.overwrite)
            rows.append(row)
            print(f"[{idx}/{len(whipple_paths)}] {row[1]} {row[0]} {row[2]}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(plot_one, path, args.source_dir, args.segmented_dir, args.output_dir, args.overwrite)
                for path in whipple_paths
            ]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                print(f"[{idx}/{len(whipple_paths)}] {row[1]} {row[0]} {row[2]}", flush=True)
    status_counts = {}
    for _event_id, status, _path in rows:
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"status_counts={status_counts}", flush=True)
    print(f"output_dir={Path(args.output_dir).resolve()}", flush=True)


if __name__ == "__main__":
    main()
