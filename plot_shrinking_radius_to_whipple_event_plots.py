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
from fit_whipple_jacchia_catalog_from_h5 import event_id_from_path, load_group


DEFAULT_SOURCE_DIR = Path("results/tristatic_student_t_bootstrap_orbit100_20260630")
DEFAULT_WHIPPLE_DIR = Path("results/tristatic_whipple_jacchia_bootstrap_orbit100_20260701")
DEFAULT_SHRINKING_DIR = Path("results/tristatic_shrinking_radius_to_whipple_synthetic_20260701")
DEFAULT_OUTPUT_DIR = Path("results/tristatic_shrinking_radius_to_whipple_event_plots_20260701")
EVENT_PREFIX = "joint_delay_doppler_fft_"
DEFAULT_MAX_SHRINKING_VELOCITY_RMS_MPS = 1000.0
DEFAULT_MAX_SHRINKING_VELOCITY_MAX_MPS = 1000.0
DEFAULT_MAX_SHRINKING_PATH_RATE_RMS_MPS = 1000.0
DEFAULT_MAX_SHRINKING_PATH_RMS_M = 2.0


def inject_shrinking_radius_fit(
    joint_fit,
    shrinking_h5,
    max_velocity_rms_mps=DEFAULT_MAX_SHRINKING_VELOCITY_RMS_MPS,
    max_velocity_max_mps=DEFAULT_MAX_SHRINKING_VELOCITY_MAX_MPS,
    max_path_rate_rms_mps=DEFAULT_MAX_SHRINKING_PATH_RATE_RMS_MPS,
    max_path_rms_m=DEFAULT_MAX_SHRINKING_PATH_RMS_M,
):
    shrinking_h5 = Path(shrinking_h5)
    if not shrinking_h5.exists():
        joint_fit["segmented_radius_available"] = False
        return joint_fit
    with h5py.File(shrinking_h5, "r") as h:
        velocity_rms = float(h.attrs.get("synthetic_velocity_rms_mps", np.nan))
        path_rms = float(h.attrs.get("synthetic_path_rms_m", np.nan))
        path_rate_rms = float(h.attrs.get("synthetic_path_rate_rms_mps", np.nan))
        optimizer_success = bool(h.attrs.get("optimizer_success", False))
        segmented_v_gcrs_mps = np.asarray(h["v_gcrs_mps"][()], dtype=np.float64)
        nominal_v_gcrs_mps = np.asarray(joint_fit.get("v_gcrs_mps", []), dtype=np.float64)
        if segmented_v_gcrs_mps.shape == nominal_v_gcrs_mps.shape:
            velocity_max = float(np.nanmax(np.linalg.norm(segmented_v_gcrs_mps - nominal_v_gcrs_mps, axis=1)))
        else:
            velocity_max = float(h.attrs.get("synthetic_velocity_max_mps", np.nan))
        fit_quality_ok = (
            optimizer_success
            and np.isfinite(velocity_rms)
            and np.isfinite(velocity_max)
            and np.isfinite(path_rms)
            and np.isfinite(path_rate_rms)
            and velocity_rms <= float(max_velocity_rms_mps)
            and velocity_max <= float(max_velocity_max_mps)
            and path_rms <= float(max_path_rms_m)
            and path_rate_rms <= float(max_path_rate_rms_mps)
        )
        joint_fit["segmented_radius_available"] = True
        joint_fit["segmented_radius_fit_quality_ok"] = bool(fit_quality_ok)
        joint_fit["segmented_radius_best_n_segments"] = 1
        joint_fit["segmented_radius_initial_radius_m"] = float(h.attrs.get("initial_radius_m", np.nan))
        joint_fit["segmented_radius_initial_mass_kg"] = float(h.attrs.get("initial_mass_kg", np.nan))
        joint_fit["segmented_radius_bootstrap_samples_successful"] = int(h.attrs.get("bootstrap_samples_successful", 0))
        interval_map = {
            "bootstrap_initial_radius_lo95_m": "segmented_radius_initial_radius_lo95_m",
            "bootstrap_initial_radius_hi95_m": "segmented_radius_initial_radius_hi95_m",
            "bootstrap_initial_mass_lo95_kg": "segmented_radius_initial_mass_lo95_kg",
            "bootstrap_initial_mass_hi95_kg": "segmented_radius_initial_mass_hi95_kg",
        }
        for h5_key, plot_key in interval_map.items():
            joint_fit[plot_key] = float(h.attrs.get(h5_key, np.nan))
        joint_fit["segmented_radius_v_gcrs_mps"] = segmented_v_gcrs_mps
        joint_fit["segmented_radius_radius_m"] = np.asarray(h["radius_m"][()], dtype=np.float64)
        joint_fit["segmented_radius_mass_kg"] = np.asarray(h["mass_kg"][()], dtype=np.float64)
        joint_fit["shrinking_radius_synthetic_velocity_rms_mps"] = float(
            h.attrs.get("synthetic_velocity_rms_mps", np.nan)
        )
        joint_fit["shrinking_radius_synthetic_path_rate_rms_mps"] = float(
            h.attrs.get("synthetic_path_rate_rms_mps", np.nan)
        )
        joint_fit["shrinking_radius_synthetic_path_rms_m"] = path_rms
        joint_fit["shrinking_radius_synthetic_velocity_max_mps"] = velocity_max
    return joint_fit


def make_delay_fit(source_joint):
    return {
        "params": np.asarray(source_joint.get("params", np.full(7, np.nan)), dtype=np.float64),
        "rms_total_path_residual_m": float(source_joint.get("rms_total_path_residual_m", np.nan)),
        "weighted_rms": float(source_joint.get("weighted_rms", np.nan)),
        "initial_radius_m": float(source_joint.get("initial_radius_m", np.nan)),
        "initial_mass_kg": float(source_joint.get("initial_mass_kg", np.nan)),
    }


def plot_one(
    whipple_h5,
    source_dir,
    shrinking_dir,
    output_dir,
    overwrite,
    max_shrinking_velocity_rms_mps,
    max_shrinking_velocity_max_mps,
    max_shrinking_path_rate_rms_mps,
    max_shrinking_path_rms_m,
):
    whipple_h5 = Path(whipple_h5)
    event_id = event_id_from_path(whipple_h5)
    source_h5 = Path(source_dir) / whipple_h5.name
    shrinking_h5 = Path(shrinking_dir) / f"shrinking_radius_to_whipple_{event_id}.h5"
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

    joint_fit = inject_shrinking_radius_fit(
        joint_fit,
        shrinking_h5,
        max_velocity_rms_mps=max_shrinking_velocity_rms_mps,
        max_velocity_max_mps=max_shrinking_velocity_max_mps,
        max_path_rate_rms_mps=max_shrinking_path_rate_rms_mps,
        max_path_rms_m=max_shrinking_path_rms_m,
    )
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
        description="Regenerate Whipple-Jacchia event plots with single shrinking-radius synthetic-fit overlays."
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--whipple-dir", default=DEFAULT_WHIPPLE_DIR)
    parser.add_argument("--shrinking-dir", default=DEFAULT_SHRINKING_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--max-shrinking-velocity-rms-mps", type=float, default=DEFAULT_MAX_SHRINKING_VELOCITY_RMS_MPS)
    parser.add_argument("--max-shrinking-velocity-max-mps", type=float, default=DEFAULT_MAX_SHRINKING_VELOCITY_MAX_MPS)
    parser.add_argument("--max-shrinking-path-rate-rms-mps", type=float, default=DEFAULT_MAX_SHRINKING_PATH_RATE_RMS_MPS)
    parser.add_argument("--max-shrinking-path-rms-m", type=float, default=DEFAULT_MAX_SHRINKING_PATH_RMS_M)
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
            row = plot_one(
                path,
                args.source_dir,
                args.shrinking_dir,
                args.output_dir,
                args.overwrite,
                args.max_shrinking_velocity_rms_mps,
                args.max_shrinking_velocity_max_mps,
                args.max_shrinking_path_rate_rms_mps,
                args.max_shrinking_path_rms_m,
            )
            rows.append(row)
            print(f"[{idx}/{len(whipple_paths)}] {row[1]} {row[0]} {row[2]}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(
                    plot_one,
                    path,
                    args.source_dir,
                    args.shrinking_dir,
                    args.output_dir,
                    args.overwrite,
                    args.max_shrinking_velocity_rms_mps,
                    args.max_shrinking_velocity_max_mps,
                    args.max_shrinking_path_rate_rms_mps,
                    args.max_shrinking_path_rms_m,
                )
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
