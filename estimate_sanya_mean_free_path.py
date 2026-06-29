#!/usr/bin/env python3
"""Estimate MSIS mean free paths and Knudsen numbers for Sanya meteors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


FALCON9_DIR = Path("/Users/jvi019/src/falcon9")
PAPER_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper")

COMMON_VOLUME_LAT_DEG = 18.567821
COMMON_VOLUME_LON_DEG = 109.683719
REFERENCE_TIME_UTC = np.datetime64("2024-04-22T18:00:00")
ALTITUDES_KM = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

MEDIAN_DIAMETER_UM = 28.0
UPPER_IQR_DIAMETER_UM = 57.0


def import_falcon9_mfp():
    sys.path.insert(0, str(FALCON9_DIR))
    import mean_free_path  # noqa: PLC0415

    return mean_free_path


def fmt_sci(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2f}$\times10^{{{exponent}}}$"


def make_table() -> str:
    mfp = import_falcon9_mfp()
    lambda_m = mfp.mean_free_path_m(
        time_dt64=REFERENCE_TIME_UTC,
        lat_deg=COMMON_VOLUME_LAT_DEG,
        lon_deg=COMMON_VOLUME_LON_DEG,
        alt_km=ALTITUDES_KM,
    )
    kn_median = lambda_m / (MEDIAN_DIAMETER_UM * 1e-6)
    kn_upper = lambda_m / (UPPER_IQR_DIAMETER_UM * 1e-6)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Altitude (km) & \(\lambda\) (m) & \(Kn(d=28~\mu\mathrm{m})\) & \(Kn(d=57~\mu\mathrm{m})\) \\",
        r"\midrule",
    ]
    for alt, lam, kn_med, kn_up in zip(ALTITUDES_KM, lambda_m, kn_median, kn_upper):
        lines.append(
            f"{alt:.0f} & {lam:.3g} & {fmt_sci(kn_med)} & {fmt_sci(kn_up)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_DIR / "tables" / "sanya_mean_free_path_knudsen.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = make_table()
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(table)


if __name__ == "__main__":
    main()
