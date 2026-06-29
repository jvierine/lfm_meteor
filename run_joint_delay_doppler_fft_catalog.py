import argparse
import concurrent.futures
import os
import subprocess
import sys

import h5py
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as joint
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, load_events, pair_tristatic_events


SCRIPT_VERSION = "v20260618a"
DEFAULT_OUTPUT_DIR = os.path.join("results", "tristatic")
CANONICAL_SNR_MIN_DB = 15.0
CANONICAL_CLIP_FFT_RESIDUAL_KHZ = 2.0
CANONICAL_SIGMA_FFT_HZ = 5000.0
CANONICAL_REFERENCE_CHIRP_RATE_SCALE = joint.DEFAULT_REFERENCE_CHIRP_RATE_SCALE


def parse_key_value_stdout(stdout):
    out = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def numeric_or_nan(mapping, key):
    try:
        return float(mapping[key])
    except Exception:
        return np.nan


def load_existing_result_summary(output_h5):
    out = {}
    try:
        with h5py.File(output_h5, "r") as h:
            if "joint_fit" not in h:
                return out
            jg = h["joint_fit"]
            for key in (
                "n_points",
                "n_path_observations",
                "n_fft_observations",
                "n_delay_clipped_observations",
                "rms_total_path_residual_m",
                "mean_abs_total_path_residual_m",
                "rms_fft_residual_hz",
                "mean_abs_fft_residual_hz",
                "rms_path_rate_residual_mps",
                "mean_abs_path_rate_residual_mps",
                "initial_radius_m",
                "initial_mass_kg",
                "fallback_log10_radius_std",
                "all_finite_path_residual_rms_m",
                "coincident_delay_weight",
                "n_coincident_delay_constraint_rows",
                "min_geometric_points",
                "default_min_geometric_points",
            ):
                if key in jg.attrs:
                    out[key] = float(jg.attrs[key])
            if "radius_m" in jg:
                out["joint_final_radius_um"] = float(jg["radius_m"][-1] * 1e6)
            if "mass_kg" in jg:
                out["joint_final_mass_kg"] = float(jg["mass_kg"][-1])
            out["joint_path_rms_m"] = out.get("rms_total_path_residual_m", np.nan)
            out["joint_path_mean_abs_m"] = out.get("mean_abs_total_path_residual_m", np.nan)
            out["joint_fft_rms_hz"] = out.get("rms_fft_residual_hz", np.nan)
            out["joint_fft_mean_abs_hz"] = out.get("mean_abs_fft_residual_hz", np.nan)
            out["joint_path_rate_rms_mps"] = out.get("rms_path_rate_residual_mps", np.nan)
            out["joint_path_rate_mean_abs_mps"] = out.get("mean_abs_path_rate_residual_mps", np.nan)
            out["joint_radius_um"] = out.get("initial_radius_m", np.nan) * 1e6
            out["joint_initial_mass_kg"] = out.get("initial_mass_kg", np.nan)
            for key in ("dynamical_model", "fallback_reason", "bad_fit_reasons", "bad_fit_recovery_step", "pre_recovery_bad_fit_reasons"):
                if key in jg.attrs:
                    value = jg.attrs[key]
                    out[key] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            if "bad_fit_detected" in jg.attrs:
                out["bad_fit_detected"] = bool(jg.attrs["bad_fit_detected"])
    except Exception:
        return out
    return out


def catalog_events(selected_event_ids=None):
    ref_fits = base.load_reference_fits()
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    rows = []
    selected = set(selected_event_ids) if selected_event_ids else None
    for idx, triplet in enumerate(triplets):
        fit0 = base.match_reference_fit(triplet[0], ref_fits)
        if fit0 is None:
            continue
        event_id = fit0["event_id"]
        if selected is not None and event_id not in selected and f"tri_{idx:04d}_{triplet[0].t0_ns}" not in selected:
            continue
        rows.append((idx, event_id))
    return rows


def write_summary(path, results, args_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["event_fit_script_version"] = joint.SCRIPT_VERSION
        for key, value in args_dict.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                h.attrs[key] = value
        h.create_dataset("event_id", data=np.asarray([r["event_id"] for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset("status", data=np.asarray([r["status"] for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset("output_base", data=np.asarray([r.get("output_base", "") for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset("dynamical_model", data=np.asarray([r.get("dynamical_model", "") for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset("fallback_reason", data=np.asarray([r.get("fallback_reason", "") for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset("bad_fit_reasons", data=np.asarray([r.get("bad_fit_reasons", "") for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset("bad_fit_recovery_step", data=np.asarray([r.get("bad_fit_recovery_step", "") for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset("pre_recovery_bad_fit_reasons", data=np.asarray([r.get("pre_recovery_bad_fit_reasons", "") for r in results], dtype=object), dtype=string_dtype)
        h.create_dataset(
            "bad_fit_detected",
            data=np.asarray([str(r.get("bad_fit_detected", "False")).lower() in {"true", "1"} for r in results], dtype=bool),
        )
        for key in (
            "n_points",
            "n_path_observations",
            "n_fft_observations",
            "n_delay_clipped_observations",
            "delay_only_path_rms_m",
            "joint_path_rms_m",
            "joint_path_mean_abs_m",
            "joint_fft_rms_hz",
            "joint_fft_mean_abs_hz",
            "joint_path_rate_rms_mps",
            "joint_path_rate_mean_abs_mps",
            "all_finite_path_residual_rms_m",
            "delay_only_radius_um",
            "joint_radius_um",
            "joint_initial_mass_kg",
            "joint_final_radius_um",
            "joint_final_mass_kg",
            "fallback_log10_radius_std",
            "coincident_delay_weight",
            "n_coincident_delay_constraint_rows",
            "min_geometric_points",
            "default_min_geometric_points",
            "returncode",
        ):
            h.create_dataset(key, data=np.asarray([numeric_or_nan(r, key) for r in results], dtype=np.float64))
        h.create_dataset("stderr", data=np.asarray([r.get("stderr", "") for r in results], dtype=object), dtype=string_dtype)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical Sanya tri-static joint delay + dechirped FFT "
            "beat-frequency fit over tri-static events."
        )
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-h5", default=None)
    parser.add_argument("--event-id", action="append", default=None, help="Restrict to one or more event ids.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zero-pad-factor", type=int, default=64)
    parser.add_argument("--fft-gate-upsample-factor", type=int, default=32)
    parser.add_argument("--range-upsample-factor", type=int, default=32)
    parser.add_argument("--sigma-fft-hz", type=float, default=CANONICAL_SIGMA_FFT_HZ)
    parser.add_argument("--clip-fft-residual-khz", type=float, default=CANONICAL_CLIP_FFT_RESIDUAL_KHZ)
    parser.add_argument("--snr-min-db", type=float, default=CANONICAL_SNR_MIN_DB)
    parser.add_argument("--prominence-min-db", type=float, default=8.0)
    parser.add_argument("--reference-chirp-rate-scale", type=float, default=CANONICAL_REFERENCE_CHIRP_RATE_SCALE)
    parser.add_argument("--fit-station-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bad-fit-retained-path-rms-m", type=float, default=joint.DEFAULT_BAD_FIT_RETAINED_PATH_RMS_M)
    parser.add_argument("--bad-fit-retained-fft-rms-hz", type=float, default=joint.DEFAULT_BAD_FIT_RETAINED_FFT_RMS_HZ)
    parser.add_argument("--bad-fit-max-retry", type=int, default=joint.DEFAULT_BAD_FIT_MAX_RETRY)
    parser.add_argument("--manual-outlier-h5", default=None)
    parser.add_argument("--random-initial-guesses", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--force-model-reevaluation", action="store_true")
    parser.add_argument("--coincident-delay-weight", type=float, default=joint.DEFAULT_COINCIDENT_DELAY_WEIGHT)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_h5 = args.summary_h5 or os.path.join(args.output_dir, "joint_delay_doppler_fft_catalog_summary.h5")
    rows = catalog_events(args.event_id)
    rows = rows[args.start_index :]
    if args.max_events is not None:
        rows = rows[: args.max_events]

    results = []
    script_path = os.path.join(os.path.dirname(__file__), "fit_event_joint_delay_doppler_fft.py")

    def run_one_event(ordinal, idx, event_id):
        output_base = os.path.join(args.output_dir, f"joint_delay_doppler_fft_{event_id}")
        output_h5 = f"{output_base}.h5"
        result = {"event_id": event_id, "triplet_index": idx, "output_base": output_base}
        if os.path.exists(output_h5) and not args.overwrite:
            result["status"] = "ok"
            result["loaded_existing"] = True
            result["returncode"] = 0
            result.update(load_existing_result_summary(output_h5))
            return ordinal, f"[{ordinal}/{len(rows)}] skip {event_id} -> {output_h5}", result

        cmd = [
            sys.executable,
            script_path,
            "--event-id",
            event_id,
            "--output-base",
            output_base,
            "--zero-pad-factor",
            str(args.zero_pad_factor),
            "--fft-gate-upsample-factor",
            str(args.fft_gate_upsample_factor),
            "--range-upsample-factor",
            str(args.range_upsample_factor),
            "--sigma-fft-hz",
            str(args.sigma_fft_hz),
            "--clip-fft-residual-khz",
            str(args.clip_fft_residual_khz),
            "--snr-min-db",
            str(args.snr_min_db),
            "--prominence-min-db",
            str(args.prominence_min_db),
            "--reference-chirp-rate-scale",
            str(args.reference_chirp_rate_scale),
            "--bad-fit-retained-path-rms-m",
            str(args.bad_fit_retained_path_rms_m),
            "--bad-fit-retained-fft-rms-hz",
            str(args.bad_fit_retained_fft_rms_hz),
            "--bad-fit-max-retry",
            str(args.bad_fit_max_retry),
            "--random-initial-guesses",
            str(args.random_initial_guesses),
            "--coincident-delay-weight",
            str(args.coincident_delay_weight),
        ]
        if args.random_seed is not None:
            cmd.extend(["--random-seed", str(args.random_seed + idx)])
        if args.manual_outlier_h5:
            cmd.extend(["--manual-outlier-h5", args.manual_outlier_h5])
        if args.force_model_reevaluation:
            cmd.append("--force-model-reevaluation")
        if args.fit_station_bias:
            cmd.append("--fit-station-bias")
        else:
            cmd.append("--no-fit-station-bias")
        proc = subprocess.run(cmd, cwd=os.path.dirname(__file__), text=True, capture_output=True)
        result.update(parse_key_value_stdout(proc.stdout))
        result["returncode"] = proc.returncode
        result["stderr"] = proc.stderr.strip()
        result["status"] = "ok" if proc.returncode == 0 else "error"
        if proc.returncode == 0:
            message = (
                f"  ok n={result.get('n_points', '?')} fft={result.get('n_fft_observations', '?')} "
                f"path={result.get('joint_path_rms_m', '?')} m beat={result.get('joint_fft_rms_hz', '?')} Hz"
            )
        else:
            message = f"  error returncode={proc.returncode}: {proc.stderr.strip()[:300]}"
        return ordinal, f"[{ordinal}/{len(rows)}] fit {event_id}\n{message}", result

    indexed_rows = [(ordinal, idx, event_id) for ordinal, (idx, event_id) in enumerate(rows, start=1)]
    if args.jobs <= 1:
        for ordinal, idx, event_id in indexed_rows:
            _, message, result = run_one_event(ordinal, idx, event_id)
            print(message, flush=True)
            results.append(result)
            write_summary(summary_h5, sorted(results, key=lambda r: r["triplet_index"]), vars(args))
    else:
        max_workers = max(1, int(args.jobs))
        print(f"running with jobs={max_workers}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(run_one_event, ordinal, idx, event_id): (ordinal, event_id)
                for ordinal, idx, event_id in indexed_rows
            }
            for future in concurrent.futures.as_completed(future_map):
                ordinal, event_id = future_map[future]
                try:
                    _, message, result = future.result()
                except Exception as exc:
                    result = {"event_id": event_id, "triplet_index": np.nan, "status": "error", "returncode": np.nan, "stderr": str(exc)}
                    message = f"[{ordinal}/{len(rows)}] fit {event_id}\n  error driver_exception={exc}"
                print(message, flush=True)
                results.append(result)
                write_summary(summary_h5, sorted(results, key=lambda r: numeric_or_nan(r, "triplet_index")), vars(args))

    results = sorted(results, key=lambda r: numeric_or_nan(r, "triplet_index"))
    write_summary(summary_h5, results, vars(args))
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped_existing")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"summary_h5={summary_h5}")
    print(f"n_ok={ok}")
    print(f"n_skipped_existing={skipped}")
    print(f"n_error={errors}")


if __name__ == "__main__":
    main()
