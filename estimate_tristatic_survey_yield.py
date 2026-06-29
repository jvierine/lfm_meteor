#!/usr/bin/env python3
"""Estimate Sanya tri-static meteor survey yield from the April 2024 run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PAPER_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper")
COMMON_VOLUME_100KM_YIELD_FACTOR = 2.0


@dataclass(frozen=True)
class ObservedYield:
    start_utc: str = "2024-04-22 12:01:26.659"
    end_utc: str = "2024-04-22 23:49:53.159"
    tristatic_candidates: int = 238
    successful_fits: int = 167

    @property
    def hours(self) -> float:
        start = datetime.fromisoformat(self.start_utc)
        end = datetime.fromisoformat(self.end_utc)
        return (end - start).total_seconds() / 3600.0

    @property
    def candidate_rate_per_hour(self) -> float:
        return self.tristatic_candidates / self.hours

    @property
    def fit_rate_per_hour(self) -> float:
        return self.successful_fits / self.hours

    @property
    def minutes_per_candidate(self) -> float:
        return 60.0 / self.candidate_rate_per_hour

    @property
    def minutes_per_fit(self) -> float:
        return 60.0 / self.fit_rate_per_hour


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def make_table(observed: ObservedYield) -> str:
    year_hours = 365.25 * 24.0
    lines = [
        r"\begin{tabularx}{\linewidth}{Xcccc}",
        r"\toprule",
        r"Case & Meteor duty & Fitted h\(^{-1}\) & Fitted yr\(^{-1}\) & 50\% operations \\",
        r"\midrule",
        r"Observed meteor mode & 100\% & "
        + fmt(observed.fit_rate_per_hour, 1)
        + r" & \multicolumn{2}{c}{measured interval only} \\",
    ]

    for dwell_min, period_min in [(1.0, 60.0)]:
        duty = dwell_min / period_min
        elapsed_rate = observed.fit_rate_per_hour * duty
        annual = elapsed_rate * year_hours
        annual_half = 0.5 * annual
        lines.append(
            (
                f"{dwell_min:.0f} min per {period_min:.0f} min"
                + " & "
                + fmt(100.0 * duty, 1)
                + r"\% & "
                + fmt(elapsed_rate, 2)
                + " & "
                + fmt(annual / 1000.0, 1)
                + r"\(\times10^3\) & "
                + fmt(annual_half / 1000.0, 1)
                + r"\(\times10^3\) \\"
            )
        )
        improved_elapsed_rate = elapsed_rate * COMMON_VOLUME_100KM_YIELD_FACTOR
        improved_annual = annual * COMMON_VOLUME_100KM_YIELD_FACTOR
        improved_annual_half = annual_half * COMMON_VOLUME_100KM_YIELD_FACTOR
        lines.append(
            (
                "Same, 100 km common volume"
                + " & "
                + fmt(100.0 * duty, 1)
                + r"\% & "
                + fmt(improved_elapsed_rate, 2)
                + " & "
                + fmt(improved_annual / 1000.0, 1)
                + r"\(\times10^3\) & "
                + fmt(improved_annual_half / 1000.0, 1)
                + r"\(\times10^3\) \\"
            )
        )

    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Sanya tri-static survey yields and write the LaTeX table "
            "used by the article."
        )
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=DEFAULT_PAPER_DIR,
        help="Path to the Sanya tri-static paper repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex table path. Defaults to PAPER_DIR/tables/tristatic_survey_yield_estimate.tex.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.paper_dir / "tables" / "tristatic_survey_yield_estimate.tex"

    observed = ObservedYield()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_table(observed), encoding="utf-8")

    print(f"Wrote {output}")
    print(f"Observed interval: {observed.hours:.6f} h")
    print(
        "Tri-static candidates: "
        f"{observed.candidate_rate_per_hour:.2f}/h, "
        f"one per {observed.minutes_per_candidate:.2f} min"
    )
    print(
        "Successful fitted tri-static meteors: "
        f"{observed.fit_rate_per_hour:.2f}/h, "
        f"one per {observed.minutes_per_fit:.2f} min"
    )


if __name__ == "__main__":
    main()
