import os

import h5py
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_all_ceplecha_snr_weighted as cepl
import fit_event_joint_delay_doppler_fft as fit


EVENT_ID = "tri_0142_1713821835054349899"
SOURCE_H5 = "results/tristatic_student_t_bootstrap_orbit100_20260630/joint_delay_doppler_fft_tri_0142_1713821835054349899.h5"
WHIPPLE_H5 = "results/whipple_speed_test_20260630/joint_delay_doppler_fft_tri_0142_1713821835054349899.h5"
OUTPUT_DIR = "results/whipple_segmented_ceplecha_test_20260630"
RADIUS_GRID_UM = np.asarray([5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0], dtype=np.float64)


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


def segment_masks(t_rel_s, n_segments):
    edges = np.linspace(float(np.nanmin(t_rel_s)), float(np.nanmax(t_rel_s)), int(n_segments) + 1)
    masks = []
    for idx in range(int(n_segments)):
        if idx == int(n_segments) - 1:
            mask = (t_rel_s >= edges[idx]) & (t_rel_s <= edges[idx + 1])
        else:
            mask = (t_rel_s >= edges[idx]) & (t_rel_s < edges[idx + 1])
        masks.append((edges[idx], edges[idx + 1], mask))
    return masks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with h5py.File(SOURCE_H5, "r") as h:
        source_joint = load_group(h["joint_fit"])
        source_fft = load_group(h["fft_observations"])
    with h5py.File(WHIPPLE_H5, "r") as h:
        whipple_joint = load_group(h["joint_fit"])

    measured = np.asarray(source_joint["measured_total_paths_m"], dtype=np.float64)
    times_ns = np.asarray(source_joint["time_ns"], dtype=np.int64)
    sigma_m = np.asarray(source_joint["path_sigma_m"], dtype=np.float64)
    fft_offset_hz = np.asarray(source_joint["observed_fft_beat_hz"], dtype=np.float64)
    fft_keep = np.asarray(source_joint["fft_keep"], dtype=bool)
    sigma_fft_hz = np.asarray(source_joint["fft_sigma_hz"], dtype=np.float64)
    path_keep = np.asarray(source_joint["path_keep"], dtype=bool)
    t_rel_s = np.asarray(source_joint["t_rel_s"], dtype=np.float64)

    whipple_x = np.asarray(whipple_joint["x_gcrs_m"], dtype=np.float64)
    whipple_v = np.asarray(whipple_joint["v_gcrs_mps"], dtype=np.float64)
    whipple_x_itrs = np.asarray(whipple_joint["x_itrs_m"], dtype=np.float64)
    rho_of_alt_m, _meta = base.density_interpolator(times_ns, whipple_x_itrs)

    rows = []
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for n_segments in (2, 3):
        for seg_idx, t0, t1, mask in (
            (idx, start, stop, seg_mask)
            for idx, (start, stop, seg_mask) in enumerate(segment_masks(t_rel_s, n_segments))
        ):
            if np.count_nonzero(path_keep[mask]) < 10 or np.count_nonzero(fft_keep[mask]) < 10:
                continue
            first = int(np.flatnonzero(mask)[0])
            epoch_time_ns = int(times_ns[first])
            best = None
            attempts = []
            for radius_um in RADIUS_GRID_UM:
                p0 = np.concatenate([whipple_x[first], whipple_v[first], [np.log10(radius_um * 1e-6)]])
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
                        keep_rows=mask,
                        epoch_time_ns=epoch_time_ns,
                        fit_station_bias=True,
                        fft_model="range_offset_corrected_beat",
                        reference_chirp_rate_scale=1.0,
                        path_keep=path_keep,
                        model_kind="ceplecha",
                        residual_likelihood="student_t",
                        student_t_nu_delay=1.5,
                        student_t_nu_fft=3.0,
                    )
                except Exception as exc:
                    attempts.append((float(radius_um), np.nan, str(exc)))
                    continue
                attempts.append((float(radius_um), float(candidate["weighted_rms"]), ""))
                if best is None or candidate["weighted_rms"] < best["weighted_rms"]:
                    best = candidate
            if best is None:
                continue
            out_base = os.path.join(OUTPUT_DIR, f"{EVENT_ID}_seg{n_segments}_{seg_idx + 1}")
            seg_fft = {
                key: np.asarray(value)[mask]
                for key, value in source_fft.items()
                if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape
            }
            delay_fit = {
                "params": np.concatenate([whipple_x[first], whipple_v[first], [np.nan]]),
                "rms_total_path_residual_m": np.nan,
                "weighted_rms": np.nan,
            }
            fit.write_h5(
                out_base,
                EVENT_ID,
                delay_fit,
                best,
                seg_fft,
                best["fft_sigma_hz"],
                512,
                "cached_segment",
                32,
                0.0,
                fit.DEFAULT_FFT_TIME_PAD_US,
                1.0,
            )
            best_for_plot = dict(best)
            best_for_plot["parameter_covariance"] = np.full((len(best["params"]), len(best["params"])), np.nan)
            fit.plot_joint_fit(EVENT_ID, delay_fit, best_for_plot, out_base, rho_of_alt_m, snr_db=seg_fft.get("fft_snr_db"))
            rows.append(
                {
                    "n_segments": n_segments,
                    "segment_index": seg_idx + 1,
                    "t_start_s": float(t0),
                    "t_stop_s": float(t1),
                    "n_points": int(best["n_points"]),
                    "n_path_observations": int(best["n_path_observations"]),
                    "n_fft_observations": int(best["n_fft_observations"]),
                    "weighted_rms": float(best["weighted_rms"]),
                    "path_rms_m": float(best["rms_total_path_residual_m"]),
                    "path_rate_rms_mps": float(best["rms_path_rate_residual_mps"]),
                    "fft_rms_hz": float(best["rms_fft_residual_hz"]),
                    "initial_radius_m": float(best["initial_radius_m"]),
                    "initial_mass_kg": float(best["initial_mass_kg"]),
                    "start_speed_km_s": float(best["speed_km_s"][0]),
                    "end_speed_km_s": float(best["speed_km_s"][-1]),
                    "output_base": out_base,
                }
            )
            print(
                f"seg{n_segments}.{seg_idx + 1}: "
                f"path={best['rms_total_path_residual_m']:.3f} m "
                f"rate={best['rms_path_rate_residual_mps']:.1f} m/s "
                f"fft={best['rms_fft_residual_hz']:.1f} Hz "
                f"wrms={best['weighted_rms']:.3f} "
                f"r0={best['initial_radius_m'] * 1e6:.2f} um "
                f"v={best['speed_km_s'][0]:.2f}->{best['speed_km_s'][-1]:.2f} km/s"
            )

    summary_path = os.path.join(OUTPUT_DIR, f"{EVENT_ID}_summary.h5")
    with h5py.File(summary_path, "w") as h:
        h.attrs["event_id"] = EVENT_ID
        h.attrs["source_h5"] = SOURCE_H5
        h.attrs["whipple_h5"] = WHIPPLE_H5
        if rows:
            for key in rows[0]:
                data = [row[key] for row in rows]
                if isinstance(data[0], str):
                    h.create_dataset(key, data=np.asarray(data, dtype=object), dtype=string_dtype)
                else:
                    h[key] = np.asarray(data)
    print(summary_path)


if __name__ == "__main__":
    main()
