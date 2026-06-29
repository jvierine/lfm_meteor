import argparse
import glob
import os

import h5py
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as joint


def load_group(group):
    out = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        out[key] = value
    for key, dataset in group.items():
        if isinstance(dataset, h5py.Dataset):
            out[key] = dataset[()]
    return out


def mean_abs(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(np.abs(finite)))


def update_mean_residual_attrs(h5_group, joint_fit):
    path_keep = np.asarray(joint_fit["path_keep"], dtype=bool)
    fft_keep = np.asarray(joint_fit["fft_keep"], dtype=bool)
    path_resid = np.asarray(joint_fit["path_residuals_m"], dtype=np.float64)[path_keep]
    fft_resid = np.asarray(joint_fit["fft_residuals_hz"], dtype=np.float64)[fft_keep]
    path_rate_resid = np.asarray(joint_fit["path_rate_residuals_mps"], dtype=np.float64)[fft_keep]
    updates = {
        "mean_abs_total_path_residual_m": mean_abs(path_resid),
        "mean_abs_fft_residual_hz": mean_abs(fft_resid),
        "mean_abs_path_rate_residual_mps": mean_abs(path_rate_resid),
    }
    for key, value in updates.items():
        h5_group.attrs[key] = value
        joint_fit[key] = value


def regenerate_one(path):
    with h5py.File(path, "r+") as h:
        event_id = h.attrs["event_id"]
        if isinstance(event_id, bytes):
            event_id = event_id.decode("utf-8")
        delay_fit = load_group(h["delay_only_fit"])
        joint_fit = load_group(h["joint_fit"])
        fft_obs = load_group(h["fft_observations"])
        update_mean_residual_attrs(h["joint_fit"], joint_fit)
        snr_db = fft_obs.get("fft_snr_db")
        times_ns = np.asarray(joint_fit["time_ns"], dtype=np.int64)
        points = np.asarray(joint_fit["x_itrs_m"], dtype=np.float64)
    try:
        rho_of_alt_m, _meta = base.density_interpolator(times_ns, points)
    except Exception:
        rho_of_alt_m = lambda alt_m: np.zeros_like(np.asarray(alt_m, dtype=np.float64))
    joint.plot_joint_fit(event_id, delay_fit, joint_fit, path[:-3], rho_of_alt_m, snr_db=snr_db)


def main():
    parser = argparse.ArgumentParser(description="Regenerate joint FFT event plots from stored fit HDF5 products.")
    parser.add_argument("result_dir")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.result_dir, "joint_delay_doppler_fft_tri_*.h5")))
    if not paths:
        raise SystemExit(f"No event HDF5 files found in {args.result_dir}")
    for idx, path in enumerate(paths, 1):
        regenerate_one(path)
        if idx % 25 == 0 or idx == len(paths):
            print(f"{idx}/{len(paths)} plots regenerated", flush=True)


if __name__ == "__main__":
    main()
