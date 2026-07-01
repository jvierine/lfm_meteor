import os

import h5py
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_all_ceplecha_snr_weighted as cepl
import fit_event_joint_delay_doppler_fft as fit


EVENT_ID = "tri_0142_1713821835054349899"
SOURCE_H5 = "results/tristatic_student_t_bootstrap_orbit100_20260630/joint_delay_doppler_fft_tri_0142_1713821835054349899.h5"
WHIPPLE_H5 = "results/whipple_speed_test_20260630/joint_delay_doppler_fft_tri_0142_1713821835054349899.h5"
OUTPUT_DIR = "results/ceplecha_ablation_sigma_test_20260630"
RADIUS_GRID_UM = np.asarray([100.0, 200.0, 500.0], dtype=np.float64)
SIGMA_GRID_KG_J = np.asarray([1e-9, cepl.ABLATION_SIGMA_KG_J, 1e-7], dtype=np.float64)


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
        source_fft = load_group(h["fft_observations"])
    with h5py.File(WHIPPLE_H5, "r") as h:
        whipple_joint = load_group(h["joint_fit"])
    source_fft.pop("link_names", None)

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

    best = None
    rows = []
    for radius_um in RADIUS_GRID_UM:
        for ablation_sigma in SIGMA_GRID_KG_J:
            p0 = np.concatenate(
                [
                    whipple_x0,
                    whipple_v0,
                    [np.log10(radius_um * 1e-6), np.log10(ablation_sigma)],
                ]
            )
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
                    model_kind="ceplecha_ablation_sigma",
                    residual_likelihood="student_t",
                    student_t_nu_delay=1.5,
                    student_t_nu_fft=3.0,
                )
            except Exception as exc:
                rows.append((radius_um, ablation_sigma, np.nan, np.nan, np.nan, np.nan, str(exc)))
                continue
            rows.append(
                (
                    radius_um,
                    ablation_sigma,
                    candidate["weighted_rms"],
                    candidate["rms_total_path_residual_m"],
                    candidate["rms_path_rate_residual_mps"],
                    candidate["rms_fft_residual_hz"],
                    "",
                )
            )
            print(
                f"start r={radius_um:.0f} um sigma={ablation_sigma:.1e}: "
                f"path={candidate['rms_total_path_residual_m']:.3f} m "
                f"rate={candidate['rms_path_rate_residual_mps']:.1f} m/s "
                f"fft={candidate['rms_fft_residual_hz']:.1f} Hz "
                f"wrms={candidate['weighted_rms']:.3f} "
                f"fit_r={candidate['initial_radius_m'] * 1e6:.2f} um "
                f"fit_sigma={candidate['ablation_sigma_kg_j']:.3e}"
            )
            if best is None or candidate["weighted_rms"] < best["weighted_rms"]:
                best = candidate

    if best is None:
        raise RuntimeError("No successful ceplecha_ablation_sigma fits")

    out_base = os.path.join(OUTPUT_DIR, EVENT_ID)
    delay_fit = {
        "params": np.concatenate([whipple_x0, whipple_v0, [np.nan]]),
        "rms_total_path_residual_m": np.nan,
        "weighted_rms": np.nan,
    }
    fit.write_h5(
        out_base,
        EVENT_ID,
        delay_fit,
        best,
        source_fft,
        best["fft_sigma_hz"],
        512,
        "cached_full_event_ablation_sigma",
        32,
        0.0,
        fit.DEFAULT_FFT_TIME_PAD_US,
        1.0,
    )
    best_for_plot = dict(best)
    best_for_plot["parameter_covariance"] = np.full((len(best["params"]), len(best["params"])), np.nan)
    fit.plot_joint_fit(EVENT_ID, delay_fit, best_for_plot, out_base, rho_of_alt_m, snr_db=source_fft.get("fft_snr_db"))

    summary_path = os.path.join(OUTPUT_DIR, f"{EVENT_ID}_summary.h5")
    with h5py.File(summary_path, "w") as h:
        h.attrs["event_id"] = EVENT_ID
        h.attrs["source_h5"] = SOURCE_H5
        h.attrs["whipple_h5"] = WHIPPLE_H5
        h.attrs["best_output_base"] = out_base
        h.create_dataset("start_radius_um", data=np.asarray([row[0] for row in rows], dtype=np.float64))
        h.create_dataset("start_ablation_sigma_kg_j", data=np.asarray([row[1] for row in rows], dtype=np.float64))
        h.create_dataset("weighted_rms", data=np.asarray([row[2] for row in rows], dtype=np.float64))
        h.create_dataset("path_rms_m", data=np.asarray([row[3] for row in rows], dtype=np.float64))
        h.create_dataset("path_rate_rms_mps", data=np.asarray([row[4] for row in rows], dtype=np.float64))
        h.create_dataset("fft_rms_hz", data=np.asarray([row[5] for row in rows], dtype=np.float64))
    print(f"best={out_base}.h5")
    print(summary_path)


if __name__ == "__main__":
    main()
