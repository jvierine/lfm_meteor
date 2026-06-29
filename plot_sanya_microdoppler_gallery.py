#!/usr/bin/env python3
"""Make paginated Sanya monostatic micro-Doppler gallery figures.

Each row is one meteor event, not one pulse.  The row image is made from all
usable pulses in the event-cut HDF5 file.
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_tristatic_microdoppler_fft import (
    decode,
    draw_microdoppler_panel,
    microdoppler_image,
)


SCRIPT_VERSION = "v20260619a"
DEFAULT_EVENT_GLOB = os.path.join("results", "tristatic_head_echoes", "sanya", "sanya_*.h5")
DEFAULT_OUTPUT_DIR = os.path.join("results", f"sanya_microdoppler_gallery_{SCRIPT_VERSION}")


def load_sanya_event(path: str) -> dict:
    with h5py.File(path, "r") as h:
        return {
            "path": path,
            "site": decode(h["site"][()]).lower() if "site" in h else "sanya",
            "event_id": decode(h["event_id"][()]) if "event_id" in h else os.path.splitext(os.path.basename(path))[0],
            "times_ns": np.asarray(h["times_ns"][:], dtype=np.int64),
            "raw": np.asarray(h["raw"][:], dtype=np.complex64),
            "range_gate": np.asarray(h["range_gate"][:], dtype=np.float64),
            "snr_peak_db": np.asarray(h["snr_peak_db"][:], dtype=np.float64),
            "sr_mhz": float(h["sr_mhz"][()]),
            "bw_mhz": float(h["bw_mhz"][()]),
            "pulse_length_us": float(h["pulse_length_us"][()]) if "pulse_length_us" in h else 199.0,
            "source_file": decode(h["source_file"][()]) if "source_file" in h else "",
        }


def event_sort_key(path: str) -> tuple[int, str]:
    try:
        with h5py.File(path, "r") as h:
            if "times_ns" in h and len(h["times_ns"]) > 0:
                return int(h["times_ns"][0]), path
    except OSError:
        pass
    return 0, path


def event_label(data: dict) -> str:
    first_us = int(np.asarray(data["times_ns"], dtype=np.int64)[0] // 1000)
    return f"{data['event_id']}  first pulse: {first_us} unix us"


def draw_empty_panel(ax, data: dict, args: argparse.Namespace) -> None:
    t0_ns = int(np.nanmin(data["times_ns"]))
    t1_ns = int(np.nanmax(data["times_ns"]))
    duration_s = max((t1_ns - t0_ns) / 1e9, 0.005)
    ax.set_xlim(0.0, duration_s)
    ax.set_ylim(-0.5 * args.width_khz, 0.5 * args.width_khz)
    ax.set_ylabel("kHz")
    ax.text(0.5, 0.5, "No usable spectra", ha="center", va="center", transform=ax.transAxes)


def write_page(page_index: int, page_events: list[dict], output_dir: str, args: argparse.Namespace) -> dict:
    n_rows = len(page_events)
    fig_height = max(2.0 * n_rows, 2.4)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8.3, fig_height),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    mesh = None
    page_rows = []
    for ax, data in zip(axes, page_events):
        result = microdoppler_image(data, args)
        t0_ns = int(np.nanmin(data["times_ns"]))
        t1_ns = int(np.nanmax(data["times_ns"]))
        if result is None:
            draw_empty_panel(ax, data, args)
            n_valid = 0
        else:
            mesh = draw_microdoppler_panel(
                ax,
                result,
                data,
                np.asarray(data["times_ns"], dtype=np.int64),
                t0_ns,
                t1_ns,
                args,
                show_xlabel=True,
            )
            n_valid = int(result["n_valid"])
        ax.set_title(event_label(data), fontsize=9.5, loc="left", pad=3.0)
        page_rows.append(
            {
                "event_id": data["event_id"],
                "source_h5": data["path"],
                "source_file": data["source_file"],
                "first_pulse_unix_us": int(data["times_ns"][0] // 1000),
                "n_pulses": int(len(data["times_ns"])),
                "n_valid": int(n_valid),
            }
        )

    fig.suptitle(
        f"Sanya monostatic micro-Doppler gallery, page {page_index + 1}",
        fontsize=12,
    )
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, pad=0.012)
        cbar.set_label("Relative power (dB)")

    os.makedirs(output_dir, exist_ok=True)
    stem = f"sanya_microdoppler_gallery_page_{page_index + 1:04d}"
    png = os.path.join(output_dir, f"{stem}.png")
    pdf = os.path.join(output_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png, "pdf": pdf, "rows": page_rows}


def write_manifest(pages: list[dict], output_dir: str, args: argparse.Namespace) -> str:
    path = os.path.join(output_dir, "sanya_microdoppler_gallery_manifest.h5")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    rows = []
    for page_index, page in enumerate(pages):
        for row_index, row in enumerate(page["rows"]):
            rows.append(
                {
                    **row,
                    "page_index": page_index,
                    "row_index": row_index,
                    "png": page["png"],
                    "pdf": page["pdf"],
                }
            )

    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["event_glob"] = args.event_glob
        h.attrs["shard_index"] = -1 if args.shard_index is None else int(args.shard_index)
        h.attrs["num_shards"] = -1 if args.num_shards is None else int(args.num_shards)
        h.attrs["events_per_page"] = int(args.events_per_page)
        h.attrs["zero_pad_factor"] = int(args.zero_pad_factor)
        h.attrs["gate_upsample_factor"] = int(args.gate_upsample_factor)
        h.attrs["width_khz"] = float(args.width_khz)
        h.attrs["db_floor"] = float(args.db_floor)
        h.attrs["snr_min_db"] = float(args.snr_min_db)
        h.attrs["row_definition"] = "one Sanya event-cut HDF5 file, corresponding to one meteor"
        h.create_dataset("event_id", data=np.asarray([r["event_id"] for r in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("source_h5", data=np.asarray([r["source_h5"] for r in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("source_file", data=np.asarray([r["source_file"] for r in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("png", data=np.asarray([r["png"] for r in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("pdf", data=np.asarray([r["pdf"] for r in rows], dtype=object), dtype=string_dtype)
        for key in ("page_index", "row_index", "first_pulse_unix_us", "n_pulses", "n_valid"):
            h[key] = np.asarray([r[key] for r in rows], dtype=np.int64)
    return path


def run(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(args.event_glob), key=event_sort_key)
    if args.num_shards is not None:
        if args.shard_index is None:
            raise ValueError("--shard-index is required when --num-shards is used")
        if args.num_shards < 1:
            raise ValueError("--num-shards must be at least 1")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
        paths = paths[args.shard_index :: args.num_shards]
    if args.max_events is not None:
        paths = paths[: args.max_events]
    if not paths:
        raise FileNotFoundError(f"No Sanya event HDF5 files matched {args.event_glob!r}")

    pages = []
    for start in range(0, len(paths), args.events_per_page):
        page_paths = paths[start : start + args.events_per_page]
        page_events = [load_sanya_event(path) for path in page_paths]
        page = write_page(len(pages), page_events, args.output_dir, args)
        pages.append(page)
        print(f"page {len(pages):04d}: {page['png']}")

    manifest = write_manifest(pages, args.output_dir, args)
    print(f"n_events={len(paths)}")
    print(f"n_pages={len(pages)}")
    print(f"output_dir={os.path.abspath(args.output_dir)}")
    print(f"manifest={os.path.abspath(manifest)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-glob", default=DEFAULT_EVENT_GLOB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--events-per-page", type=int, default=5)
    parser.add_argument("--zero-pad-factor", type=int, default=64)
    parser.add_argument("--gate-upsample-factor", type=int, default=32)
    parser.add_argument("--width-khz", type=float, default=200.0)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--snr-min-db", type=float, default=-np.inf)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()
    if args.events_per_page < 1:
        raise ValueError("--events-per-page must be at least 1")
    run(args)


if __name__ == "__main__":
    main()
