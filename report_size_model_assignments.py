"""Write an HDF5 catalogue of size-model assignments for each tri-static event.

The output records, per event, whether the current products support a
fixed-drag/free-flow size estimate, a shrinking-radius Ceplecha size estimate,
or only a one-sided lower-bound/censored interpretation.  It is a reporting
layer only: it does not refit events.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np


DEFAULT_BALLISTIC_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
DEFAULT_CEPLECHA_GLOB = "results/all_tristatic_ceplecha_snr_weighted_*.h5"
DEFAULT_LIMIT_H5 = "results/drag_size_limit_catalog.h5"
DEFAULT_OUTPUT_H5 = "results/size_model_assignment_catalog.h5"
DEFAULT_OUTPUT_JSON = "results/size_model_assignment_summary.json"
DEFAULT_OUTPUT_REPORT = None
DEFAULT_OUTPUT_TEX = "/Users/jvi019/src/sanya_tristatic_paper/memos/memo21_size_model_assignments.tex"


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


def event_ids_from_dataset(h5: h5py.File, path: str = "event_id") -> list[str]:
    return [decode_string(value) for value in h5[path][:]]


def read_limit_catalog(path: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with h5py.File(path, "r") as h:
        c = h["catalog"]
        event_id = event_ids_from_dataset(h, "catalog/event_id")
        model = [decode_string(value) for value in c["model"][:]]
        classification = [decode_string(value) for value in c["classification"][:]]
        reason = [decode_string(value) for value in c["reason"][:]]
        informative = np.asarray(c["limit_is_informative"][:], dtype=bool)
        reported_d_um = np.asarray(c["reported_lower_limit_diameter_um"][:], dtype=float)
        measured_d_um = np.asarray(c["measured_diameter_um"][:], dtype=float)
        parameter_value = np.asarray(c["parameter_value"][:], dtype=float)
        log10_parameter_std = np.asarray(c["log10_parameter_std"][:], dtype=float)

        for i, event in enumerate(event_id):
            entry = rows.setdefault(event, {})
            prefix = "fixed" if model[i] == "fixed_drag" else "ceplecha"
            entry[f"{prefix}_classification"] = classification[i]
            entry[f"{prefix}_reason"] = reason[i]
            entry[f"{prefix}_limit_is_informative"] = bool(informative[i])
            entry[f"{prefix}_reported_lower_limit_diameter_um"] = float(reported_d_um[i])
            entry[f"{prefix}_measured_diameter_um"] = float(measured_d_um[i])
            entry[f"{prefix}_parameter_value"] = float(parameter_value[i])
            entry[f"{prefix}_log10_parameter_std"] = float(log10_parameter_std[i])
    return rows


def choose_assignment(row: dict[str, object]) -> str:
    fixed_class = row.get("fixed_classification", "")
    cepl_class = row.get("ceplecha_classification", "")
    fixed_info = bool(row.get("fixed_limit_is_informative", False))
    cepl_info = bool(row.get("ceplecha_limit_is_informative", False))

    if cepl_class == "measured":
        return "shrinking_radius"
    if fixed_class == "measured":
        return "fixed_acceleration"
    if cepl_info or fixed_info:
        return "lower_bound"
    if cepl_class == "limit" or fixed_class == "limit":
        return "censored_uninformative"
    return "not_fit"


def finite_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def write_h5(path: str, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        for key, value in summary["inputs"].items():
            h.attrs[key] = value
        h.attrs["assignment_priority"] = (
            "shrinking_radius measured; else fixed_acceleration measured; else informative lower_bound; "
            "else censored_uninformative; else not_fit"
        )

        c = h.create_group("catalog")
        string_columns = [
            "event_id",
            "assigned_model",
            "fixed_classification",
            "fixed_reason",
            "ceplecha_classification",
            "ceplecha_reason",
        ]
        bool_columns = [
            "has_fixed_acceleration_fit",
            "has_shrinking_radius_fit",
            "has_lower_bound",
            "fixed_limit_is_informative",
            "ceplecha_limit_is_informative",
        ]
        float_columns = [
            "fixed_measured_diameter_um",
            "ceplecha_measured_diameter_um",
            "fixed_reported_lower_limit_diameter_um",
            "ceplecha_reported_lower_limit_diameter_um",
            "fixed_parameter_value",
            "ceplecha_parameter_value",
            "fixed_log10_parameter_std",
            "ceplecha_log10_parameter_std",
        ]

        for key in string_columns:
            c[key] = np.asarray([str(row.get(key, "")) for row in rows], dtype=string_dtype)
        for key in bool_columns:
            c[key] = np.asarray([bool(row.get(key, False)) for row in rows], dtype=bool)
        for key in float_columns:
            c[key] = np.asarray([finite_or_nan(row.get(key, np.nan)) for row in rows], dtype=np.float64)

        s = h.create_group("summary")
        for key, value in summary["counts"].items():
            s.attrs[key] = value


def format_float(value: object, digits: int = 2) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(out):
        return ""
    return f"{out:.{digits}f}"


def write_report(path: str, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    """Write a compact human-readable report derived from the HDF5 catalogue."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["assigned_model"]), []).append(row)

    order = ["shrinking_radius", "fixed_acceleration", "lower_bound", "censored_uninformative", "not_fit"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Size Model Assignment Report\n\n")
        f.write("Generated by `report_size_model_assignments.py` from:\n\n")
        for key, value in summary["inputs"].items():
            f.write(f"- `{key}`: `{value}`\n")
        f.write("\n")
        f.write("Assignment priority: shrinking-radius measured; else fixed-acceleration measured; "
                "else informative lower bound; else censored/uninformative; else not fit.\n\n")
        f.write("## Counts\n\n")
        for key, value in summary["counts"].items():
            f.write(f"- `{key}`: {value}\n")
        f.write("\n")

        for assigned_model in order:
            model_rows = grouped.get(assigned_model, [])
            if not model_rows:
                continue
            f.write(f"## {assigned_model}\n\n")
            f.write("| event_id | fixed class | Ceplecha class | fixed d (um) | Ceplecha d0 (um) | lower d (um) | reason |\n")
            f.write("|---|---:|---:|---:|---:|---:|---|\n")
            for row in model_rows:
                fixed_lower = format_float(row.get("fixed_reported_lower_limit_diameter_um"))
                cepl_lower = format_float(row.get("ceplecha_reported_lower_limit_diameter_um"))
                lower = fixed_lower or cepl_lower
                reason = str(row.get("ceplecha_reason") or row.get("fixed_reason") or "")
                f.write(
                    "| "
                    f"`{row['event_id']}` | "
                    f"{row.get('fixed_classification', '')} | "
                    f"{row.get('ceplecha_classification', '')} | "
                    f"{format_float(row.get('fixed_measured_diameter_um'))} | "
                    f"{format_float(row.get('ceplecha_measured_diameter_um'))} | "
                    f"{lower} | "
                    f"{reason} |\n"
                )
            f.write("\n")


def tex_mono(value: object) -> str:
    return r"\texttt{\detokenize{" + str(value) + "}}"


def tex_cell(value: object) -> str:
    text = str(value)
    if text == "" or text.lower() == "nan":
        return "--"
    return tex_mono(text)


def write_latex_memo(path: str, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    """Write a standalone LaTeX memo from the model-assignment catalogue."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["assigned_model"]), []).append(row)

    order = ["shrinking_radius", "fixed_acceleration", "lower_bound", "censored_uninformative", "not_fit"]
    lower_bound_rows = grouped.get("lower_bound", [])

    with open(path, "w", encoding="utf-8") as f:
        f.write(r"""\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{siunitx}
\usepackage[hidelinks]{hyperref}

\title{Memo 21: Size-Model Assignment Catalogue}
\author{Sanya tri-static head-echo analysis notes}
\date{\today}

\begin{document}
\maketitle

\section*{Purpose}

This memo records the event-level size-model assignment for the Sanya
tri-static meteor fits.  It is a reporting product, not a new fit.  The memo
is generated directly from the HDF5 fit products and the HDF5 size-limit
catalogue so that the classification can be rerun when the underlying fits are
updated.

\section{Input Products}

The input products are:
\begin{itemize}
""")
        for key, value in summary["inputs"].items():
            f.write(f"  \\item {tex_mono(key)}: {tex_mono(value)}\n")
        f.write(r"""\end{itemize}

The source program for this memo is
\begin{quote}
\texttt{/Users/jvi019/src/lfm\_meteor/report\_size\_model\_assignments.py}.
\end{quote}
The primary machine-readable output is
\begin{quote}
\texttt{/Users/jvi019/src/lfm\_meteor/results/size\_model\_assignment\_catalog.h5}.
\end{quote}

\section{Assignment Rule}

Each event is assigned using a single deterministic priority order.  If the
shrinking-radius Ceplecha fit gives a measured size, the event is assigned to
\texttt{shrinking\_radius}.  If not, but the fixed-drag/free-flow fit gives a
measured size, it is assigned to \texttt{fixed\_acceleration}.  If neither
model gives a measured size but at least one model gives an informative
one-sided size constraint, the event is assigned to \texttt{lower\_bound}.
Events for which the fitted drag or radius parameter is too weakly constrained
to give a useful lower limit are assigned to
\texttt{censored\_uninformative}.  This last category is physically important:
the data are then effectively falling back toward a constant-velocity
trajectory model, so the size scale should not be reported as a precise
measurement.

\section{Summary}

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
Quantity & Count \\
\midrule
""")
        for key, value in summary["counts"].items():
            f.write(f"{tex_mono(key)} & {value} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\caption{Summary counts for the size-model assignment catalogue.}
\label{tab:memo21-size-model-counts}
\end{table}

""")
        if lower_bound_rows:
            f.write("The informative lower-bound-only events are ")
            f.write(", ".join(tex_mono(row["event_id"]) for row in lower_bound_rows))
            f.write(".  ")
            f.write("For the present run, the fixed-drag/free-flow model reports ")
            first = lower_bound_rows[0]
            f.write(
                f"a lower diameter limit of \\SI{{{format_float(first.get('fixed_reported_lower_limit_diameter_um'), 1)}}}{{\\micro\\metre}} "
                f"for {tex_mono(first['event_id'])}."
            )
            f.write("\n\n")

        f.write(r"""\section{Event Catalogue}

\small
\begin{longtable}{p{0.33\linewidth}llllll}
\caption{Event-level size-model assignments.  Diameters are reported in
\si{\micro\metre}.  The fixed-drag diameter uses the free-molecular
\(C_D A/m\) convention.  The Ceplecha column is the fitted initial diameter
\(d_0\).}\\
\toprule
Event & Assignment & Fixed & Ceplecha & Fixed \(d\) & Ceplecha \(d_0\) & Lower \(d\) \\
\midrule
\endfirsthead
\caption[]{Event-level size-model assignments, continued.}\\
\toprule
Event & Assignment & Fixed & Ceplecha & Fixed \(d\) & Ceplecha \(d_0\) & Lower \(d\) \\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{Continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
""")
        for assigned_model in order:
            for row in grouped.get(assigned_model, []):
                fixed_lower = format_float(row.get("fixed_reported_lower_limit_diameter_um"))
                cepl_lower = format_float(row.get("ceplecha_reported_lower_limit_diameter_um"))
                lower = fixed_lower or cepl_lower or "--"
                f.write(
                    f"{tex_mono(row['event_id'])} & "
                    f"{tex_cell(row.get('assigned_model', ''))} & "
                    f"{tex_cell(row.get('fixed_classification', ''))} & "
                    f"{tex_cell(row.get('ceplecha_classification', ''))} & "
                    f"{format_float(row.get('fixed_measured_diameter_um')) or '--'} & "
                    f"{format_float(row.get('ceplecha_measured_diameter_um')) or '--'} & "
                    f"{lower} \\\\\n"
                )
        f.write(r"""\end{longtable}
\normalsize

\end{document}
""")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ballistic-h5", default=DEFAULT_BALLISTIC_H5)
    p.add_argument("--ceplecha-h5", default=None, help="Defaults to newest matching Ceplecha product.")
    p.add_argument("--limit-h5", default=DEFAULT_LIMIT_H5)
    p.add_argument("--output-h5", default=DEFAULT_OUTPUT_H5)
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--output-report", default=DEFAULT_OUTPUT_REPORT, help="Optional Markdown report path.")
    p.add_argument("--output-tex", default=DEFAULT_OUTPUT_TEX)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ceplecha_h5 is None:
        args.ceplecha_h5 = newest_existing(DEFAULT_CEPLECHA_GLOB)

    limit_rows = read_limit_catalog(args.limit_h5)
    with h5py.File(args.ballistic_h5, "r") as h:
        fixed_events = set(event_ids_from_dataset(h))
    with h5py.File(args.ceplecha_h5, "r") as h:
        ceplecha_events = set(event_ids_from_dataset(h))

    event_ids = sorted(fixed_events | ceplecha_events | set(limit_rows))
    rows = []
    for event_id in event_ids:
        source = dict(limit_rows.get(event_id, {}))
        fixed_class = source.get("fixed_classification", "")
        cepl_class = source.get("ceplecha_classification", "")
        row = {
            "event_id": event_id,
            **source,
            "has_fixed_acceleration_fit": fixed_class == "measured",
            "has_shrinking_radius_fit": cepl_class == "measured",
            "has_lower_bound": bool(source.get("fixed_limit_is_informative", False))
            or bool(source.get("ceplecha_limit_is_informative", False)),
        }
        row["assigned_model"] = choose_assignment(row)
        rows.append(row)

    assigned = np.asarray([row["assigned_model"] for row in rows], dtype=object)
    counts = {f"n_{name}": int(np.count_nonzero(assigned == name)) for name in sorted(set(assigned))}
    counts["n_events"] = int(len(rows))
    counts["n_fixed_measured"] = int(np.count_nonzero([row["has_fixed_acceleration_fit"] for row in rows]))
    counts["n_ceplecha_measured"] = int(np.count_nonzero([row["has_shrinking_radius_fit"] for row in rows]))
    counts["n_informative_lower_bound"] = int(np.count_nonzero([row["has_lower_bound"] for row in rows]))

    summary = {
        "inputs": {
            "ballistic_h5": args.ballistic_h5,
            "ceplecha_h5": args.ceplecha_h5,
            "limit_h5": args.limit_h5,
        },
        "counts": counts,
    }
    write_h5(args.output_h5, rows, summary)
    if args.output_report:
        write_report(args.output_report, rows, summary)
    write_latex_memo(args.output_tex, rows, summary)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.output_h5}")
    if args.output_report:
        print(f"wrote {args.output_report}")
    print(f"wrote {args.output_tex}")
    print(f"wrote {args.output_json}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
