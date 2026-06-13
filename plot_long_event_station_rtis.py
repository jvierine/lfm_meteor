import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "results"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
SITE_LABELS = {
    "sanya": "Sanya TX",
    "danzhou": "Danzhou RX",
    "wenchang": "Wenchang RX",
}
SITE_ORDER = ["sanya", "danzhou", "wenchang"]


def event_slug(event, rank):
    return f"rank{rank:02d}_{event['event_id']}"


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def local_event_path(path, site, repo_root):
    """Resolve old absolute Linux paths to the local staged tri-static copies."""
    path = decode(path)
    old_prefix = "/home/j/src/lfm_meteor/"
    if path.startswith(old_prefix):
        path = os.path.join(repo_root, path[len(old_prefix) :])
    if os.path.exists(path):
        return path
    staged = os.path.join(repo_root, RESULTS_DIR, "tristatic_head_echoes", site, os.path.basename(path))
    if os.path.exists(staged):
        return staged
    raise FileNotFoundError(path)


def rank_events(results_h5):
    with h5py.File(results_h5, "r") as h:
        duration_s = h["summary_duration_s"][()]
        n_points = h["summary_n_points"][()]
        event_ids = [decode(value) for value in h["summary_event_id"][()]]
    score = duration_s + 1e-6 * n_points
    order = np.argsort(-score)
    return [
        {
            "index": int(idx),
            "event_id": event_ids[idx],
            "duration_s": float(duration_s[idx]),
            "n_points": int(n_points[idx]),
        }
        for idx in order
    ]


def event_paths(index_h5, event_index, repo_root):
    paths = {}
    with h5py.File(index_h5, "r") as h:
        for site in SITE_ORDER:
            paths[site] = local_event_path(h[f"{site}_event_h5"][event_index], site, repo_root)
    return paths


def load_rti(path):
    with h5py.File(path, "r") as h:
        echoes = h["echoes"][()]
        ranges_km = h["ranges_km_axis"][()]
        times_ns = h["times_ns"][()].astype(np.int64)
        relative_time_s = h["relative_time_s"][()]
        detected_range_km = h["range_km"][()]
        snr_peak_db = h["snr_peak_db"][()]
    power_db = 10.0 * np.log10(np.maximum(np.abs(echoes.T) ** 2.0, 1e-12))
    power_db = power_db - np.nanmedian(power_db)
    return {
        "power_db": power_db,
        "ranges_km": ranges_km,
        "times_ns": times_ns,
        "relative_time_s": relative_time_s,
        "detected_range_km": detected_range_km,
        "snr_peak_db": snr_peak_db,
    }


def plot_panel(ax, data, site, t0_ns, vmin, vmax):
    t_s = (data["times_ns"].astype(np.float64) - float(t0_ns)) / 1e9
    mesh = ax.pcolormesh(
        t_s,
        data["ranges_km"],
        data["power_db"],
        shading="auto",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    ax.plot(t_s, data["detected_range_km"], ".", color="white", ms=2.7, alpha=0.95)
    center = float(np.nanmedian(data["detected_range_km"]))
    spread = float(np.nanmax(data["detected_range_km"]) - np.nanmin(data["detected_range_km"]))
    half_width = max(4.0, 0.5 * spread + 2.0)
    ax.set_ylim(center - half_width, center + half_width)
    ax.set_ylabel("Range (km)")
    ax.set_title(f"{SITE_LABELS[site]}  max SNR={np.nanmax(data['snr_peak_db']):.1f} dB")
    ax.grid(False)
    return mesh


def save_single_station(output_base, event, site, data, t0_ns, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6.7, 4.1), constrained_layout=True)
    mesh = plot_panel(ax, data, site, t0_ns, vmin, vmax)
    ax.set_xlabel("Time since first station detection (s)")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Matched-filter power (dB)")
    fig.suptitle(f"{event['event_id']}  duration={event['duration_s']:.3f} s", fontsize=11)
    fig.savefig(f"{output_base}_{site}.png", dpi=260)
    fig.savefig(f"{output_base}_{site}.pdf")
    plt.close(fig)


def plot_event(event, paths, output_dir, rank, legacy_names=False):
    os.makedirs(output_dir, exist_ok=True)
    data = {site: load_rti(path) for site, path in paths.items()}
    t0_ns = min(int(station["times_ns"].min()) for station in data.values())
    vmax = max(float(np.nanpercentile(station["power_db"], 99.7)) for station in data.values())
    vmax = min(max(vmax, 20.0), 65.0)
    vmin = 0.0

    output_name = "tristatic_long_event_rti" if legacy_names else f"tristatic_long_event_rti_{event_slug(event, rank)}"
    output_base = os.path.join(output_dir, output_name)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True, constrained_layout=True)
    mesh = None
    for ax, site in zip(axes, SITE_ORDER):
        mesh = plot_panel(ax, data[site], site, t0_ns, vmin, vmax)
    axes[-1].set_xlabel("Time since first station detection (s)")
    fig.suptitle(
        f"Long tri-static head-echo event {event['event_id']} "
        f"({event['n_points']} matched points, {event['duration_s']:.3f} s)",
        fontsize=12,
    )
    cb = fig.colorbar(mesh, ax=axes, pad=0.018, shrink=0.88)
    cb.set_label("Matched-filter power (dB)")
    fig.savefig(f"{output_base}.png", dpi=260)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)

    for site in SITE_ORDER:
        save_single_station(output_base, event, site, data[site], t0_ns, vmin, vmax)

    return output_base


def parse_args():
    parser = argparse.ArgumentParser(description="Plot three station RTIs for a long tri-static Sanya meteor event.")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=ARTICLE_FIGURE_DIR)
    parser.add_argument("--rank", type=int, default=1, help="1-based rank by fitted tri-static duration.")
    parser.add_argument("--count", type=int, default=1, help="Number of ranked events to plot, starting at --rank.")
    parser.add_argument("--event-index", type=int, default=None, help="Explicit row in tristatic_event_index.h5.")
    parser.add_argument("--legacy-names", action="store_true", help="Use the original non-event-specific output filenames.")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.getcwd()
    results_h5 = os.path.join(args.results_dir, "tristatic_results.h5")
    index_h5 = os.path.join(args.results_dir, "tristatic_event_index.h5")
    ranked = rank_events(results_h5)
    if args.event_index is not None:
        event = next(item for item in ranked if item["index"] == args.event_index)
        events = [(args.rank, event)]
    else:
        start = max(0, args.rank - 1)
        stop = min(len(ranked), start + max(1, args.count))
        events = [(rank + 1, ranked[rank]) for rank in range(start, stop)]

    for rank, event in events:
        paths = event_paths(index_h5, event["index"], repo_root)
        output_base = plot_event(event, paths, args.output_dir, rank, legacy_names=args.legacy_names)
        print(f"event: {event['event_id']} rank={rank} index={event['index']} duration={event['duration_s']:.3f}s points={event['n_points']}")
        for site in SITE_ORDER:
            print(f"{site}: {paths[site]}")
        print(f"wrote: {output_base}.png")
        print(f"wrote: {output_base}.pdf")
        for site in SITE_ORDER:
            print(f"wrote: {output_base}_{site}.png")
            print(f"wrote: {output_base}_{site}.pdf")


if __name__ == "__main__":
    main()
