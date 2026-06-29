"""Estimate one-sided meteoroid size limits for weak-drag fits.

This script is intentionally separate from the fitting routines.  It reads the
existing constant-velocity, fixed-drag/ballistic, and Ceplecha HDF5 products
and classifies each event as either a measured drag/size estimate or a
censored limit.

Procedure
---------
1. Compare each one-parameter drag/ablation fit with the constant-velocity
   baseline using the same fitted rows and the path uncertainties stored in
   the drag-product HDF5 group.
2. Treat the extra physical scale parameter as detected only when the
   improvement over constant velocity exceeds a one-parameter likelihood-ratio
   threshold and the linearized uncertainty is finite and not effectively
   pinned to the no-drag side of the model.
3. For a weak fixed-drag fit, report an upper limit on the fitted molecular
   free-flow drag coefficient b = C_D A / m.  Convert that to a lower limit on
   spherical compact-particle diameter using d > 3 C_D / (2 rho_m b_upper).
4. For a weak Ceplecha shrinking-radius fit, report a lower limit on the
   initial radius r0 and diameter d0 = 2 r0.

The default confidence level is 95 percent one-sided.  The output is a CSV
event catalogue plus a JSON summary with counts and median limits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from statistics import NormalDist

import h5py
import numpy as np


DEFAULT_CONST_VELOCITY_H5 = "results/gcrs_trajectory_fits_lfm_ambiguity_v20260613b.h5"
DEFAULT_BALLISTIC_GLOB = "results/all_tristatic_ballistic_snr_weighted_*.h5"
DEFAULT_CEPLECHA_GLOB = "results/all_tristatic_ceplecha_snr_weighted_*.h5"
DEFAULT_OUTPUT_H5 = "results/drag_size_limit_catalog.h5"
DEFAULT_OUTPUT_CSV = "results/drag_size_limit_catalog.csv"
DEFAULT_OUTPUT_JSON = "results/drag_size_limit_summary.json"

MIN_B_M2_PER_KG = 1e-4
MAX_RADIUS_M = 1e-2


def decode_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def newest_existing(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match {pattern!r}")
    return matches[-1]


def chi2_from_residuals(residuals_m: np.ndarray, sigma_m: np.ndarray | None) -> float:
    residuals = np.asarray(residuals_m, dtype=float)
    if sigma_m is None:
        return float(np.nansum(residuals**2.0))
    sigma = np.asarray(sigma_m, dtype=float)
    return float(np.nansum((residuals / sigma) ** 2.0))


def finite_power10(log_value: float) -> float:
    if not np.isfinite(log_value):
        return np.nan
    if log_value > 308.0:
        return np.inf
    if log_value < -324.0:
        return 0.0
    return float(10.0**log_value)


def conservative_floor_125(value: float) -> float:
    """Round a positive lower limit downward to a 1-2-5 reporting grid."""

    if not np.isfinite(value) or value <= 0.0:
        return np.nan
    exponent = math.floor(math.log10(value))
    scaled = value / (10.0**exponent)
    if scaled >= 5.0:
        mantissa = 5.0
    elif scaled >= 2.0:
        mantissa = 2.0
    else:
        mantissa = 1.0
    return float(mantissa * 10.0**exponent)


def match_constant_velocity_residuals(cv_group: h5py.Group, model_group: h5py.Group) -> np.ndarray | None:
    """Return constant-velocity residuals on the same rows as a drag fit."""

    if "time_ns" not in cv_group or "time_ns" not in model_group:
        return None
    if "total_path_residuals_m" not in cv_group:
        return None

    cv_time = np.asarray(cv_group["time_ns"][:], dtype=np.int64)
    model_time = np.asarray(model_group["time_ns"][:], dtype=np.int64)
    cv_resid = np.asarray(cv_group["total_path_residuals_m"][:], dtype=float)

    by_time: dict[int, deque[int]] = defaultdict(deque)
    for idx, time_ns in enumerate(cv_time):
        by_time[int(time_ns)].append(idx)
    indices = [by_time[int(t)].popleft() if by_time[int(t)] else -1 for t in model_time]
    if any(i < 0 for i in indices):
        return None
    return cv_resid[np.asarray(indices, dtype=int)]


def delta_chi2_against_constant_velocity(
    cv_h5: h5py.File,
    model_h5: h5py.File,
    event_id: str,
) -> tuple[float, float, float] | tuple[None, None, None]:
    if "points" not in cv_h5 or "points" not in model_h5:
        return None, None, None
    if event_id not in cv_h5["points"] or event_id not in model_h5["points"]:
        return None, None, None

    model_group = model_h5["points"][event_id]
    cv_resid = match_constant_velocity_residuals(cv_h5["points"][event_id], model_group)
    if cv_resid is None or "residuals_m" not in model_group:
        return None, None, None

    model_resid = np.asarray(model_group["residuals_m"][:], dtype=float)
    if cv_resid.shape != model_resid.shape:
        return None, None, None
    sigma_m = np.asarray(model_group["sigma_m"][:], dtype=float) if "sigma_m" in model_group else None
    if sigma_m is not None and sigma_m.shape != model_resid.shape:
        sigma_m = None
    chi2_cv = chi2_from_residuals(cv_resid, sigma_m)
    chi2_model = chi2_from_residuals(model_resid, sigma_m)
    return chi2_cv - chi2_model, chi2_cv, chi2_model


@dataclass
class Classification:
    kind: str
    reason: str


def classify_parameter(
    *,
    delta_chi2: float | None,
    delta_chi2_threshold: float,
    log_value: float,
    log_std: float,
    non_drag_log_bound: float,
    bound_side: str,
    z: float,
    max_log_std: float,
    edge_tolerance_dex: float,
) -> Classification:
    reasons = []
    if delta_chi2 is None or not np.isfinite(delta_chi2):
        reasons.append("no_constant_velocity_comparison")
    elif delta_chi2 < delta_chi2_threshold:
        reasons.append("insignificant_improvement_over_constant_velocity")

    if not np.isfinite(log_std) or log_std <= 0.0:
        reasons.append("covariance_unavailable")
    elif log_std > max_log_std:
        reasons.append("large_log_parameter_uncertainty")

    if bound_side == "low":
        near_edge = log_value <= non_drag_log_bound + edge_tolerance_dex
        includes_edge = np.isfinite(log_std) and (log_value - z * log_std <= non_drag_log_bound)
    elif bound_side == "high":
        near_edge = log_value >= non_drag_log_bound - edge_tolerance_dex
        includes_edge = np.isfinite(log_std) and (log_value + z * log_std >= non_drag_log_bound)
    else:
        raise ValueError(f"unknown bound_side={bound_side!r}")

    if near_edge:
        reasons.append("near_no_drag_parameter_bound")
    if includes_edge:
        reasons.append("confidence_interval_reaches_no_drag_bound")

    if reasons:
        return Classification("limit", ";".join(dict.fromkeys(reasons)))
    return Classification("measured", "significant_drag_parameter")


def event_ids(h5: h5py.File) -> list[str]:
    return [decode_string(x) for x in h5["event_id"][:]]


def ballistic_rows(
    cv_h5: h5py.File,
    ballistic_h5: h5py.File,
    *,
    confidence: float,
    delta_chi2_threshold: float,
    density_kg_m3: float,
    drag_coefficient: float,
    max_log_std: float,
    edge_tolerance_dex: float,
    min_informative_diameter_um: float,
    max_informative_diameter_um: float,
) -> list[dict[str, object]]:
    z = NormalDist().inv_cdf(confidence)
    ids = event_ids(ballistic_h5)
    out = []
    for idx, event_id in enumerate(ids):
        b = float(ballistic_h5["b_drag_m2_per_kg"][idx])
        log_b = math.log10(b) if b > 0.0 else np.nan
        log_std = float(ballistic_h5["log10_b_std"][idx])
        sigma_log_b = log_std
        b_upper = finite_power10(log_b + z * sigma_log_b) if np.isfinite(log_b) and np.isfinite(sigma_log_b) else np.nan
        diameter_lower_m = (
            3.0 * drag_coefficient / (2.0 * density_kg_m3 * b_upper)
            if np.isfinite(b_upper) and b_upper > 0.0
            else np.nan
        )
        lower_diameter_um = diameter_lower_m * 1e6 if np.isfinite(diameter_lower_m) else np.nan
        lower_radius_um = 0.5 * lower_diameter_um if np.isfinite(lower_diameter_um) else np.nan
        measured_diameter_um = (
            3.0 * drag_coefficient / (2.0 * density_kg_m3 * b) * 1e6 if b > 0.0 else np.nan
        )
        measured_radius_um = 0.5 * measured_diameter_um if np.isfinite(measured_diameter_um) else np.nan
        delta_chi2, chi2_cv, chi2_model = delta_chi2_against_constant_velocity(cv_h5, ballistic_h5, event_id)
        cls = classify_parameter(
            delta_chi2=delta_chi2,
            delta_chi2_threshold=delta_chi2_threshold,
            log_value=log_b,
            log_std=log_std,
            non_drag_log_bound=math.log10(MIN_B_M2_PER_KG),
            bound_side="low",
            z=z,
            max_log_std=max_log_std,
            edge_tolerance_dex=edge_tolerance_dex,
        )
        informative_limit = bool(
            cls.kind == "limit"
            and delta_chi2 is not None
            and np.isfinite(delta_chi2)
            and delta_chi2 >= delta_chi2_threshold
            and np.isfinite(lower_diameter_um)
            and lower_diameter_um >= min_informative_diameter_um
            and lower_diameter_um <= max_informative_diameter_um
        )
        reported_lower_diameter_um = conservative_floor_125(lower_diameter_um) if informative_limit else np.nan
        out.append(
            {
                "event_id": event_id,
                "model": "fixed_drag",
                "classification": cls.kind,
                "reason": cls.reason,
                "confidence": confidence,
                "delta_chi2_vs_constant_velocity": delta_chi2,
                "chi2_constant_velocity": chi2_cv,
                "chi2_model": chi2_model,
                "parameter_value": b,
                "parameter_name": "C_D_A_over_m_m2_per_kg",
                "log10_parameter_std": log_std,
                "upper_limit_parameter": b_upper if cls.kind == "limit" else "",
                "lower_limit_radius_um": lower_radius_um if informative_limit else "",
                "lower_limit_diameter_um": lower_diameter_um if informative_limit else "",
                "reported_lower_limit_diameter_um": reported_lower_diameter_um if informative_limit else "",
                "reported_lower_limit_radius_um": 0.5 * reported_lower_diameter_um if informative_limit else "",
                "limit_is_informative": informative_limit if cls.kind == "limit" else "",
                "measured_radius_um": measured_radius_um if cls.kind == "measured" else "",
                "measured_diameter_um": measured_diameter_um if cls.kind == "measured" else "",
                "n_points": int(ballistic_h5["n_points"][idx]) if "n_points" in ballistic_h5 else "",
                "rms_total_path_residual_m": float(ballistic_h5["rms_total_path_residual_m"][idx]),
                "start_speed_km_s": float(ballistic_h5["start_speed_km_s"][idx])
                if "start_speed_km_s" in ballistic_h5
                else "",
            }
        )
    return out


def ceplecha_rows(
    cv_h5: h5py.File,
    ceplecha_h5: h5py.File,
    *,
    confidence: float,
    delta_chi2_threshold: float,
    max_log_std: float,
    edge_tolerance_dex: float,
    min_informative_diameter_um: float,
    max_informative_diameter_um: float,
) -> list[dict[str, object]]:
    z = NormalDist().inv_cdf(confidence)
    ids = event_ids(ceplecha_h5)
    out = []
    for idx, event_id in enumerate(ids):
        radius = float(ceplecha_h5["initial_radius_m"][idx])
        log_radius = math.log10(radius) if radius > 0.0 else np.nan
        log_std = float(ceplecha_h5["log10_radius_std"][idx])
        radius_lower = (
            finite_power10(log_radius - z * log_std)
            if np.isfinite(log_radius) and np.isfinite(log_std)
            else np.nan
        )
        lower_radius_um = radius_lower * 1e6 if np.isfinite(radius_lower) else np.nan
        lower_diameter_um = 2.0 * lower_radius_um if np.isfinite(lower_radius_um) else np.nan
        measured_radius_um = radius * 1e6 if np.isfinite(radius) else np.nan
        measured_diameter_um = 2.0 * measured_radius_um if np.isfinite(measured_radius_um) else np.nan
        delta_chi2, chi2_cv, chi2_model = delta_chi2_against_constant_velocity(cv_h5, ceplecha_h5, event_id)
        cls = classify_parameter(
            delta_chi2=delta_chi2,
            delta_chi2_threshold=delta_chi2_threshold,
            log_value=log_radius,
            log_std=log_std,
            non_drag_log_bound=math.log10(MAX_RADIUS_M),
            bound_side="high",
            z=z,
            max_log_std=max_log_std,
            edge_tolerance_dex=edge_tolerance_dex,
        )
        informative_limit = bool(
            cls.kind == "limit"
            and delta_chi2 is not None
            and np.isfinite(delta_chi2)
            and delta_chi2 >= delta_chi2_threshold
            and np.isfinite(lower_diameter_um)
            and lower_diameter_um >= min_informative_diameter_um
            and lower_diameter_um <= max_informative_diameter_um
        )
        reported_lower_diameter_um = conservative_floor_125(lower_diameter_um) if informative_limit else np.nan
        out.append(
            {
                "event_id": event_id,
                "model": "ceplecha_shrinking_radius",
                "classification": cls.kind,
                "reason": cls.reason,
                "confidence": confidence,
                "delta_chi2_vs_constant_velocity": delta_chi2,
                "chi2_constant_velocity": chi2_cv,
                "chi2_model": chi2_model,
                "parameter_value": radius,
                "parameter_name": "initial_radius_m",
                "log10_parameter_std": log_std,
                "upper_limit_parameter": "",
                "lower_limit_radius_um": lower_radius_um if informative_limit else "",
                "lower_limit_diameter_um": lower_diameter_um if informative_limit else "",
                "reported_lower_limit_diameter_um": reported_lower_diameter_um if informative_limit else "",
                "reported_lower_limit_radius_um": 0.5 * reported_lower_diameter_um if informative_limit else "",
                "limit_is_informative": informative_limit if cls.kind == "limit" else "",
                "measured_radius_um": measured_radius_um if cls.kind == "measured" else "",
                "measured_diameter_um": measured_diameter_um if cls.kind == "measured" else "",
                "n_points": int(ceplecha_h5["n_points"][idx]) if "n_points" in ceplecha_h5 else "",
                "rms_total_path_residual_m": float(ceplecha_h5["rms_total_path_residual_m"][idx]),
                "start_speed_km_s": float(ceplecha_h5["start_speed_km_s"][idx])
                if "start_speed_km_s" in ceplecha_h5
                else "",
            }
        )
    return out


def finite_float_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=float)


def catalogue_fieldnames() -> list[str]:
    return [
        "event_id",
        "model",
        "classification",
        "reason",
        "confidence",
        "delta_chi2_vs_constant_velocity",
        "chi2_constant_velocity",
        "chi2_model",
        "parameter_name",
        "parameter_value",
        "log10_parameter_std",
        "upper_limit_parameter",
        "lower_limit_radius_um",
        "lower_limit_diameter_um",
        "reported_lower_limit_diameter_um",
        "reported_lower_limit_radius_um",
        "measured_radius_um",
        "measured_diameter_um",
        "limit_is_informative",
        "n_points",
        "rms_total_path_residual_m",
        "start_speed_km_s",
    ]


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = catalogue_fieldnames()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_summary(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    summary: dict[str, object] = {
        "script": os.path.basename(__file__),
        "constant_velocity_h5": args.constant_velocity_h5,
        "ballistic_h5": args.ballistic_h5,
        "ceplecha_h5": args.ceplecha_h5,
        "confidence": args.confidence,
        "delta_chi2_threshold": args.delta_chi2_threshold,
        "density_kg_m3": args.density_kg_m3,
        "drag_coefficient": args.drag_coefficient,
        "max_log_std": args.max_log_std,
        "edge_tolerance_dex": args.edge_tolerance_dex,
        "min_informative_diameter_um": args.min_informative_diameter_um,
        "max_informative_diameter_um": args.max_informative_diameter_um,
        "models": {},
    }
    for model in sorted({row["model"] for row in rows}):
        sub = [row for row in rows if row["model"] == model]
        limits = [row for row in sub if row["classification"] == "limit"]
        informative_limits = [row for row in limits if row.get("limit_is_informative") is True]
        measured = [row for row in sub if row["classification"] == "measured"]
        limit_diam = finite_float_values(informative_limits, "reported_lower_limit_diameter_um")
        measured_diam = finite_float_values(measured, "measured_diameter_um")
        summary["models"][model] = {
            "n_total": len(sub),
            "n_limit": len(limits),
            "n_informative_limit": len(informative_limits),
            "n_uninformative_limit": len(limits) - len(informative_limits),
            "n_measured": len(measured),
            "median_reported_informative_lower_limit_diameter_um": float(np.nanmedian(limit_diam)) if limit_diam.size else None,
            "median_measured_diameter_um": float(np.nanmedian(measured_diam)) if measured_diam.size else None,
        }
    return summary


def as_float_column(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            values.append(np.nan)
        else:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def as_bool_column(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        values.append(bool(value) if value != "" else False)
    return np.asarray(values, dtype=bool)


def write_h5(path: str, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    string_columns = {"event_id", "model", "classification", "reason", "parameter_name"}
    bool_columns = {"limit_is_informative"}

    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["constant_velocity_h5"] = str(summary["constant_velocity_h5"])
        h.attrs["ballistic_h5"] = str(summary["ballistic_h5"])
        h.attrs["ceplecha_h5"] = str(summary["ceplecha_h5"])
        h.attrs["confidence"] = float(summary["confidence"])
        h.attrs["delta_chi2_threshold"] = float(summary["delta_chi2_threshold"])
        h.attrs["density_kg_m3"] = float(summary["density_kg_m3"])
        h.attrs["drag_coefficient"] = float(summary["drag_coefficient"])
        h.attrs["max_log_std"] = float(summary["max_log_std"])
        h.attrs["edge_tolerance_dex"] = float(summary["edge_tolerance_dex"])
        h.attrs["min_informative_diameter_um"] = float(summary["min_informative_diameter_um"])
        h.attrs["max_informative_diameter_um"] = float(summary["max_informative_diameter_um"])
        h.attrs["limit_reporting"] = (
            "Raw one-sided limits are retained, but reported lower limits are rounded downward "
            "to a 1-2-5 grid and only marked informative inside the configured diameter window."
        )
        h.attrs["fixed_drag_parameter_convention"] = (
            "The fixed-drag HDF5 b_drag_m2_per_kg parameter uses the molecular "
            "free-flow convention and multiplies rho |v| v directly, so b_drag = C_D A / m."
        )

        catalogue = h.create_group("catalog")
        for key in catalogue_fieldnames():
            if key in string_columns:
                catalogue[key] = np.asarray([str(row.get(key, "")) for row in rows], dtype=string_dtype)
            elif key in bool_columns:
                catalogue[key] = as_bool_column(rows, key)
            else:
                catalogue[key] = as_float_column(rows, key)

        summary_group = h.create_group("summary")
        for model, model_summary in summary["models"].items():
            group = summary_group.create_group(str(model))
            for key, value in model_summary.items():
                if value is None:
                    group.attrs[key] = np.nan
                else:
                    group.attrs[key] = value


def write_summary(path: str, summary: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant-velocity-h5", default=DEFAULT_CONST_VELOCITY_H5)
    parser.add_argument("--ballistic-h5", default=None, help="Fixed-drag HDF5 product. Defaults to newest matching product.")
    parser.add_argument("--ceplecha-h5", default=None, help="Ceplecha HDF5 product. Defaults to newest matching product.")
    parser.add_argument("--output-h5", default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--confidence", type=float, default=0.95, help="One-sided confidence level for limits.")
    parser.add_argument(
        "--delta-chi2-threshold",
        type=float,
        default=2.71,
        help="Minimum one-parameter improvement over constant velocity for a measured physical scale.",
    )
    parser.add_argument("--density-kg-m3", type=float, default=3000.0)
    parser.add_argument("--drag-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--max-log-std",
        type=float,
        default=0.5,
        help="Maximum acceptable 1-sigma log10 parameter uncertainty for a measured value.",
    )
    parser.add_argument(
        "--edge-tolerance-dex",
        type=float,
        default=0.05,
        help="Distance from the no-drag parameter bound treated as effectively pegged.",
    )
    parser.add_argument(
        "--min-informative-diameter-um",
        type=float,
        default=5.0,
        help="Lower diameter limits below this value are marked uninformative and left blank in the catalogue.",
    )
    parser.add_argument(
        "--max-informative-diameter-um",
        type=float,
        default=500.0,
        help="Lower diameter limits above this value are marked uninformative as likely covariance/pathology cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ballistic_h5 is None:
        args.ballistic_h5 = newest_existing(DEFAULT_BALLISTIC_GLOB)
    if args.ceplecha_h5 is None:
        args.ceplecha_h5 = newest_existing(DEFAULT_CEPLECHA_GLOB)

    rows: list[dict[str, object]] = []
    with (
        h5py.File(args.constant_velocity_h5, "r") as cv_h5,
        h5py.File(args.ballistic_h5, "r") as ballistic_h5,
        h5py.File(args.ceplecha_h5, "r") as ceplecha_h5,
    ):
        rows.extend(
            ballistic_rows(
                cv_h5,
                ballistic_h5,
                confidence=args.confidence,
                delta_chi2_threshold=args.delta_chi2_threshold,
                density_kg_m3=args.density_kg_m3,
                drag_coefficient=args.drag_coefficient,
                max_log_std=args.max_log_std,
                edge_tolerance_dex=args.edge_tolerance_dex,
                min_informative_diameter_um=args.min_informative_diameter_um,
                max_informative_diameter_um=args.max_informative_diameter_um,
            )
        )
        rows.extend(
            ceplecha_rows(
                cv_h5,
                ceplecha_h5,
                confidence=args.confidence,
                delta_chi2_threshold=args.delta_chi2_threshold,
                max_log_std=args.max_log_std,
                edge_tolerance_dex=args.edge_tolerance_dex,
                min_informative_diameter_um=args.min_informative_diameter_um,
                max_informative_diameter_um=args.max_informative_diameter_um,
            )
        )

    summary = make_summary(rows, args)
    write_h5(args.output_h5, rows, summary)
    write_csv(args.output_csv, rows)
    write_summary(args.output_json, summary)
    print(f"wrote {args.output_h5}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
