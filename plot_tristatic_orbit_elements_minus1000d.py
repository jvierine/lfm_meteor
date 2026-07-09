#!/usr/bin/env python3
"""Plot propagated Sanya tri-static orbit elements with bootstrap intervals."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


INPUT_H5 = Path("results/tristatic_orbits_minus1000d_jopek_williams_20260703.h5")
OUTPUT_PDF = Path("/Users/jvi019/src/sanya_tristatic_paper/figures/tristatic_orbit_elements_minus1000d.pdf")
OUTPUT_PNG = OUTPUT_PDF.with_suffix(".png")
MAX_Q_INTERVAL_AU = 20.0
JUPITER_A_AU = 5.2044
GAUSS_GRAV_K = 0.01720209895
CLASS_COLORS = {
    "cometary": "C1",
    "asteroidal": "C2",
}
CLASS_LABELS = {
    "cometary": "Cometary",
    "asteroidal": "Asteroidal",
}


def finite_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return (
        np.nanmedian(values, axis=1),
        np.nanpercentile(values, 2.5, axis=1),
        np.nanpercentile(values, 97.5, axis=1),
    )


def jopek_williams_criteria(a_au: np.ndarray, e: np.ndarray, inc_deg: np.ndarray, period_years: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        k_value = np.log(a_au * (1.0 + e) / (1.0 - e)) - 1.0
        q_aphelion = a_au * (1.0 + e)
        energy = -(GAUSS_GRAV_K**2) / (2.0 * a_au)
        tj_term = a_au * (1.0 - e * e)
        tj = 1.0 / a_au + 2.0 * JUPITER_A_AU ** (-1.5) * np.sqrt(np.maximum(tj_term, 0.0)) * np.cos(
            np.deg2rad(inc_deg)
        )

    high_i = inc_deg > 75.0
    hyperbolic = e >= 1.0
    return {
        "TJMinusI": np.logical_or.reduce((tj < 0.58, high_i, hyperbolic)),
        "KMinusI": np.logical_or.reduce((k_value > 0.0, high_i, hyperbolic)),
        "QMinusI": np.logical_or.reduce((q_aphelion > 4.6, high_i, hyperbolic)),
        "PMinusI": np.logical_or.reduce((period_years * e > 2.5, high_i, hyperbolic)),
        "EpsilonMinusI": np.logical_or.reduce((energy > -5.28e-5, high_i, hyperbolic)),
    }


def consensus_class(a_au: np.ndarray, e: np.ndarray, inc_deg: np.ndarray, period_years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cometary_probability = np.full(a_au.shape[0], np.nan, dtype=np.float64)
    label = np.full(a_au.shape[0], "asteroidal", dtype=object)
    for idx in range(a_au.shape[0]):
        finite = np.isfinite(a_au[idx]) & np.isfinite(e[idx]) & np.isfinite(inc_deg[idx]) & np.isfinite(period_years[idx])
        if not np.any(finite):
            continue
        criteria = jopek_williams_criteria(a_au[idx, finite], e[idx, finite], inc_deg[idx, finite], period_years[idx, finite])
        votes = np.vstack([np.asarray(values, dtype=np.float64) for values in criteria.values()])
        cometary_probability[idx] = float(np.nanmean(votes))
        label[idx] = "cometary" if cometary_probability[idx] >= 0.5 else "asteroidal"
    return label, cometary_probability


def errorbar_panel(ax, x, xlo, xhi, y, ylo, yhi, classes, *, xlabel, ylabel, xscale=None, show_legend=False):
    good = np.isfinite(x) & np.isfinite(xlo) & np.isfinite(xhi) & np.isfinite(y) & np.isfinite(ylo) & np.isfinite(yhi)
    if xscale == "log":
        good &= (x > 0.0) & (xlo > 0.0) & (xhi > 0.0)
    for class_name in ("cometary", "asteroidal"):
        keep = good & (classes == class_name)
        if not np.any(keep):
            continue
        xerr = np.vstack([np.maximum(x[keep] - xlo[keep], 0.0), np.maximum(xhi[keep] - x[keep], 0.0)])
        yerr = np.vstack([np.maximum(y[keep] - ylo[keep], 0.0), np.maximum(yhi[keep] - y[keep], 0.0)])
        ax.errorbar(
            x[keep],
            y[keep],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=3.0,
            lw=0.45,
            elinewidth=0.45,
            capsize=0,
            color=CLASS_COLORS[class_name],
            ecolor=(0.1, 0.1, 0.1, 0.14),
            markerfacecolor=CLASS_COLORS[class_name],
            markeredgewidth=0,
            alpha=0.82,
            zorder=3 if class_name == "cometary" else 2,
            label=CLASS_LABELS[class_name] if show_legend else None,
        )
    if xscale:
        ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", lw=0.6)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(loc="best", frameon=True, framealpha=0.88)


def main() -> None:
    with h5py.File(INPUT_H5, "r") as h:
        event_id = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in h["event_id"][()]])
        a, alo, ahi = finite_interval(h["a_au"][()])
        e, elo, ehi = finite_interval(h["e"][()])
        q, qlo, qhi = finite_interval(h["q_au"][()])
        Q, Qlo, Qhi = finite_interval(h["Q_au"][()])
        inc, inclo, inchi = finite_interval(h["i_deg"][()])
        period = np.asarray(h["period_years"][()], dtype=np.float64)
        eps, epslo, epshi = finite_interval(h["specific_energy_j_kg"][()] / 1.0e6)
        classes, cometary_probability = consensus_class(
            np.asarray(h["a_au"][()], dtype=np.float64),
            np.asarray(h["e"][()], dtype=np.float64),
            np.asarray(h["i_deg"][()], dtype=np.float64),
            period,
        )

    q_interval = Qhi - Qlo
    plot_keep = np.isfinite(q_interval) & (q_interval <= MAX_Q_INTERVAL_AU)
    omitted = event_id[~plot_keep]
    print(f"omitting {omitted.size} events with 95% Q interval > {MAX_Q_INTERVAL_AU:g} AU")
    for ev in omitted:
        print(f"  {ev}")
    a, alo, ahi = a[plot_keep], alo[plot_keep], ahi[plot_keep]
    e, elo, ehi = e[plot_keep], elo[plot_keep], ehi[plot_keep]
    q, qlo, qhi = q[plot_keep], qlo[plot_keep], qhi[plot_keep]
    Q, Qlo, Qhi = Q[plot_keep], Qlo[plot_keep], Qhi[plot_keep]
    inc, inclo, inchi = inc[plot_keep], inclo[plot_keep], inchi[plot_keep]
    eps, epslo, epshi = eps[plot_keep], epslo[plot_keep], epshi[plot_keep]
    classes = classes[plot_keep]
    cometary_probability = cometary_probability[plot_keep]
    print(
        "consensus classes: "
        f"cometary={np.count_nonzero(classes == 'cometary')} "
        f"asteroidal={np.count_nonzero(classes == 'asteroidal')}"
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.7), constrained_layout=True)

    errorbar_panel(
        axes[0, 0],
        a,
        alo,
        ahi,
        e,
        elo,
        ehi,
        xlabel=r"Semimajor axis $a$ (AU)",
        ylabel=r"Eccentricity $e$",
        xscale="log",
        classes=classes,
        show_legend=True,
    )
    errorbar_panel(
        axes[0, 1],
        q,
        qlo,
        qhi,
        inc,
        inclo,
        inchi,
        xlabel=r"Perihelion distance $q$ (AU)",
        ylabel=r"Inclination $i$ (deg)",
        classes=classes,
    )
    errorbar_panel(
        axes[1, 0],
        Q,
        Qlo,
        Qhi,
        inc,
        inclo,
        inchi,
        xlabel=r"Aphelion distance $Q$ (AU)",
        ylabel=r"Inclination $i$ (deg)",
        xscale="log",
        classes=classes,
    )
    errorbar_panel(
        axes[1, 1],
        eps,
        epslo,
        epshi,
        inc,
        inclo,
        inchi,
        xlabel=r"Specific orbital energy $\varepsilon$ (km$^2$ s$^{-2}$)",
        ylabel=r"Inclination $i$ (deg)",
        classes=classes,
    )

    labels = ["a)", "b)", "c)", "d)"]
    for ax, label in zip(axes.flat, labels):
        ax.text(
            0.02,
            0.96,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
            zorder=10,
        )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=220)
    print(OUTPUT_PDF)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
