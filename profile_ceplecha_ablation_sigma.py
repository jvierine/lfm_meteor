import os

import h5py
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_all_ceplecha_snr_weighted as cepl
import fit_event_joint_delay_doppler_fft as fit


EVENT_ID = "tri_0142_1713821835054349899"
SOURCE_H5 = "results/tristatic_student_t_bootstrap_orbit100_20260630/joint_delay_doppler_fft_tri_0142_1713821835054349899.h5"
WHIPPLE_H5 = "results/whipple_speed_test_20260630/joint_delay_doppler_fft_tri_0142_1713821835054349899.h5"
OUTPUT_DIR = "results/ceplecha_ablation_sigma_profile_20260630"
SIGMA_GRID_KG_J = np.asarray([1e-10, 1e-9, 1e-8, 1e-7, 1e-6], dtype=np.float64)
RADIUS_GRID_UM = np.asarray([100.0], dtype=np.float64)


def load_group(group):
    out = {}
    for key, value in group.attrs.items():
        out[key] = value.decode("utf-8") if isinstance(value, bytes) else value
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            arr = value[()]
            if getattr(arr, "dtype", None) is not None and arr.dtype.kind == "S":
                arr = np.asarray([item.decode("utf-8") for item in arr])
            out[key] = arr
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with h5py.File(SOURCE_H5, "r") as h:
        source_joint = load_group(h["joint_fit"])
    with h5py.File(WHIPPLE_H5, "r") as h:
        whipple_joint = load_group(h["joint_fit"])

    measured = np.asarray(source_joint["measured_total_paths_m"], dtype=np.float64)
    times_ns = np.asarray(source_joint["time_ns"], dtype=np.int64)
    sigma_m = np.asarray(source_joint["path_sigma_m"], dtype=np.float64)
    fft_offset_hz = np.asarray(source_joint["observed_fft_beat_hz"], dtype=np.float64)
    fft_keep = np.asarray(source_joint["fft_keep"], dtype=bool)
    sigma_fft_hz = np.asarray(source_joint["fft_sigma_hz"], dtype=np.float64)
    path_keep = np.asarray(source_joint["path_keep"], dtype=bool)

    whipple_x0 = np.asarray(whipple_joint["x_gcrs_m"], dtype=np.float64)[0]
    whipple_v0 = np.asarray(whipple_joint["v_gcrs_mps"], dtype=np.float64)[0]
    whipple_x_itrs = np.asarray(whipple_joint["x_itrs_m"], dtype=np.float64)
    rho_of_alt_m, _meta = base.density_interpolator(times_ns, whipple_x_itrs)

    original_sigma = cepl.ABLATION_SIGMA_KG_J
    rows = []
    try:
        for fixed_sigma in SIGMA_GRID_KG_J:
            cepl.ABLATION_SIGMA_KG_J = float(fixed_sigma)
            best = None
            for radius_um in RADIUS_GRID_UM:
                p0 = np.concatenate([whipple_x0, whipple_v0, [np.log10(radius_um * 1e-6)]])
                try:
                    candidate = fit.fit_joint_delay_doppler(
                        measured,
                        times_ns,
                        rho_of_alt_m,
                        p0,
                        sigma_m,
                        fft_offset_hz,
                        fft_keep,
                        sigma_fft_hz,
                        keep_rows=np.ones(len(times_ns), dtype=bool),
                        epoch_time_ns=int(times_ns[0]),
                        fit_station_bias=True,
                        fft_model="range_offset_corrected_beat",
                        reference_chirp_rate_scale=1.0,
                        path_keep=path_keep,
                        model_kind="ceplecha",
                        residual_likelihood="student_t",
                        student_t_nu_delay=1.5,
                        student_t_nu_fft=3.0,
                    )
                except Exception:
                    continue
                if best is None or candidate["weighted_rms"] < best["weighted_rms"]:
                    best = candidate
            if best is None:
                rows.append((fixed_sigma, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                print(f"sigma={fixed_sigma:.1e}: failed")
                continue
            rows.append(
                (
                    fixed_sigma,
                    best["weighted_rms"],
                    best["rms_total_path_residual_m"],
                    best["rms_path_rate_residual_mps"],
                    best["rms_fft_residual_hz"],
                    best["initial_radius_m"],
                    best["initial_mass_kg"],
                )
            )
            print(
                f"sigma={fixed_sigma:.1e}: "
                f"wrms={best['weighted_rms']:.3f} "
                f"path={best['rms_total_path_residual_m']:.2f} m "
                f"rate={best['rms_path_rate_residual_mps']:.1f} m/s "
                f"fft={best['rms_fft_residual_hz']:.1f} Hz "
                f"r0={best['initial_radius_m'] * 1e6:.1f} um"
            )
    finally:
        cepl.ABLATION_SIGMA_KG_J = original_sigma

    rows = np.asarray(rows, dtype=np.float64)
    summary_path = os.path.join(OUTPUT_DIR, f"{EVENT_ID}_profile.h5")
    with h5py.File(summary_path, "w") as h:
        h.attrs["event_id"] = EVENT_ID
        h.attrs["source_h5"] = SOURCE_H5
        h.attrs["whipple_h5"] = WHIPPLE_H5
        h["ablation_sigma_kg_j"] = rows[:, 0]
        h["weighted_rms"] = rows[:, 1]
        h["path_rms_m"] = rows[:, 2]
        h["path_rate_rms_mps"] = rows[:, 3]
        h["fft_rms_hz"] = rows[:, 4]
        h["initial_radius_m"] = rows[:, 5]
        h["initial_mass_kg"] = rows[:, 6]
    print(summary_path)


if __name__ == "__main__":
    main()
