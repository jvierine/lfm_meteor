#!/usr/bin/env python3
"""Plot quiet and high-noise/RFI raw-voltage passband spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


SITE_ORDER = ("Sanya", "Danzhou", "Wenchang")
SITE_COLORS = {
    "Sanya": "#1f77b4",
    "Danzhou": "#2ca02c",
    "Wenchang": "#d62728",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-h5",
        default="/Users/jvi019/src/lfm_meteor/results/sanya_passband_spectra_quiet_rfi.h5",
        help="Passband spectra HDF5 product.",
    )
    p.add_argument(
        "--output-dir",
        default="/Users/jvi019/src/sanya_tristatic_paper/memos/figures",
        help="Directory for PDF and PNG outputs.",
    )
    p.add_argument("--basename", default="memo20_passband_spectra_quiet_rfi")
    return p.parse_args()


def spectrum_db_relative(power: np.ndarray, reference: float) -> np.ndarray:
    floor = np.finfo(np.float64).tiny
    return 10.0 * np.log10(np.maximum(power.astype(np.float64), floor) / reference)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as h:
        freq_mhz = h["frequency_hz"][:] / 1.0e6

        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "savefig.dpi": 300,
            }
        )
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)

        for ax, site in zip(axes, SITE_ORDER, strict=True):
            quiet_group = h[f"spectra/{site}/quiet"]
            rfi_group = h[f"spectra/{site}/rfi"]
            quiet = quiet_group["power_spectrum"][:].astype(np.float64)
            rfi = rfi_group["power_spectrum"][:].astype(np.float64)
            reference = float(np.nanmedian(quiet))
            color = SITE_COLORS[site]

            quiet_label = f"Quiet, {quiet_group.attrs['n_raw_pulses']} pulses"
            rfi_label = f"RFI/noisy, {rfi_group.attrs['n_raw_pulses']} pulses"
            ax.plot(freq_mhz, spectrum_db_relative(quiet, reference), color="0.15", lw=1.0, label=quiet_label)
            ax.plot(freq_mhz, spectrum_db_relative(rfi, reference), color=color, lw=1.0, alpha=0.9, label=rfi_label)
            ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
            ax.grid(True, color="0.88", lw=0.6)
            ax.set_axisbelow(True)
            ax.set_ylabel("Rel. power (dB)")
            ax.set_title(site, loc="left", fontweight="bold")
            ax.legend(loc="upper right", frameon=False)

        axes[-1].set_xlabel("Baseband frequency (MHz)")
        axes[-1].set_xlim(float(np.nanmin(freq_mhz)), float(np.nanmax(freq_mhz)))

    pdf = outdir / f"{args.basename}.pdf"
    png = outdir / f"{args.basename}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
