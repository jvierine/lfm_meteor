import os
import shutil
from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
import numpy as np


C = 299792458.0
SOURCE_TIMEZONE_OFFSET_HOURS = 8
SOURCE_TIMEZONE_OFFSET_NS = int(SOURCE_TIMEZONE_OFFSET_HOURS * 3600 * 1e9)
RESULTS_DIR = os.path.expanduser("~/src/lfm_meteor/results")
INDEX_PATH = os.path.join(RESULTS_DIR, "head_echoes", "head_echo_index.h5")
TRISTATIC_DIR = os.path.join(RESULTS_DIR, "tristatic_head_echoes")
TRISTATIC_INDEX = os.path.join(RESULTS_DIR, "tristatic_event_index.h5")
PLOT_PNG = os.path.join(RESULTS_DIR, "tristatic_detection_overview.png")
PLOT_PDF = os.path.join(RESULTS_DIR, "tristatic_detection_overview.pdf")


@dataclass
class Event:
    event_id: str
    site: str
    dt0_ns: int
    dt1_ns: int
    median_range_km: float
    event_h5: str

    @property
    def delay_us(self):
        return 2.0 * self.median_range_km * 1e3 / C * 1e6


def decode_strings(arr):
    out = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        elif hasattr(v, "decode"):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def load_events():
    with h5py.File(INDEX_PATH, "r") as h:
        event_ids = decode_strings(h["event_id"][()])
        sites = decode_strings(h["site"][()])
        dt0 = h["dt0_ns"][()].astype(np.int64)
        dt1 = h["dt1_ns"][()].astype(np.int64)
        if str(h.attrs.get("times_ns_time_scale", "")).upper() != "UTC":
            dt0 = dt0 - SOURCE_TIMEZONE_OFFSET_NS
            dt1 = dt1 - SOURCE_TIMEZONE_OFFSET_NS
        median_range_km = h["median_range_km"][()].astype(np.float64)
        event_h5 = decode_strings(h["event_h5"][()])

    events = []
    for i in range(len(event_ids)):
        events.append(
            Event(
                event_id=event_ids[i],
                site=sites[i],
                dt0_ns=int(dt0[i]),
                dt1_ns=int(dt1[i]),
                median_range_km=float(median_range_km[i]),
                event_h5=event_h5[i],
            )
        )
    return events


def overlap_ns(a, b):
    return max(0, min(a.dt1_ns, b.dt1_ns) - max(a.dt0_ns, b.dt0_ns))


def best_overlap(event, candidates):
    best = None
    best_ns = 0
    for candidate in candidates:
        ns = overlap_ns(event, candidate)
        if ns > best_ns:
            best_ns = ns
            best = candidate
    return best


def pair_tristatic(events):
    sanya = [e for e in events if e.site == "sanya"]
    danzhou = [e for e in events if e.site == "danzhou"]
    wenchang = [e for e in events if e.site == "wenchang"]

    triplets = []
    for san in sanya:
        dan = best_overlap(san, danzhou)
        wen = best_overlap(san, wenchang)
        if dan is None or wen is None:
            continue
        if overlap_ns(san, dan) == 0 or overlap_ns(san, wen) == 0:
            continue
        triplets.append((san, dan, wen))
    return triplets


def make_plot(events, triplets):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    site_colors = {"sanya": "#1f77b4", "danzhou": "#ff7f0e", "wenchang": "#2ca02c"}

    for site in ["sanya", "danzhou", "wenchang"]:
        subset = [e for e in events if e.site == site]
        times = np.array([e.dt0_ns for e in subset], dtype=np.float64) / 1e9
        delays = np.array([e.delay_us for e in subset], dtype=np.float64)
        ax.scatter(
            times,
            delays,
            s=18,
            color=site_colors[site],
            alpha=0.85,
            linewidths=0,
            label=f"{site.capitalize()} ({len(subset)})",
        )

    tri_times = np.array([triplet[0].dt0_ns for triplet in triplets], dtype=np.float64) / 1e9
    tri_delays = np.array([triplet[0].delay_us for triplet in triplets], dtype=np.float64)
    ax.scatter(
        tri_times,
        tri_delays,
        s=26,
        color="#d62728",
        alpha=0.95,
        linewidths=0,
        label=f"Tristatic ({len(triplets)})",
    )

    ax.set_xlabel("Unix time (s)")
    ax.set_ylabel("Group delay ($\\mu$s)")
    ax.set_title("Meteor detections in the Sanya tri-static experiment")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=220)
    fig.savefig(PLOT_PDF)
    plt.close(fig)


def stage_tristatic_files(triplets):
    for site in ["sanya", "danzhou", "wenchang"]:
        os.makedirs(os.path.join(TRISTATIC_DIR, site), exist_ok=True)

    unique_events = {}
    for san, dan, wen in triplets:
        for event in [san, dan, wen]:
            unique_events[event.event_h5] = event

    for event in unique_events.values():
        dest = os.path.join(TRISTATIC_DIR, event.site, os.path.basename(event.event_h5))
        if not os.path.exists(dest):
            shutil.copy2(event.event_h5, dest)


def write_tristatic_index(triplets):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(TRISTATIC_INDEX, "w") as h:
        h["sanya_event_id"] = np.asarray([t[0].event_id for t in triplets], dtype=string_dtype)
        h["danzhou_event_id"] = np.asarray([t[1].event_id for t in triplets], dtype=string_dtype)
        h["wenchang_event_id"] = np.asarray([t[2].event_id for t in triplets], dtype=string_dtype)
        h["sanya_event_h5"] = np.asarray([t[0].event_h5 for t in triplets], dtype=string_dtype)
        h["danzhou_event_h5"] = np.asarray([t[1].event_h5 for t in triplets], dtype=string_dtype)
        h["wenchang_event_h5"] = np.asarray([t[2].event_h5 for t in triplets], dtype=string_dtype)
        h["sanya_dt0_ns"] = np.asarray([t[0].dt0_ns for t in triplets], dtype=np.int64)
        h["danzhou_dt0_ns"] = np.asarray([t[1].dt0_ns for t in triplets], dtype=np.int64)
        h["wenchang_dt0_ns"] = np.asarray([t[2].dt0_ns for t in triplets], dtype=np.int64)
        h["sanya_delay_us"] = np.asarray([t[0].delay_us for t in triplets], dtype=np.float64)
        h["danzhou_delay_us"] = np.asarray([t[1].delay_us for t in triplets], dtype=np.float64)
        h["wenchang_delay_us"] = np.asarray([t[2].delay_us for t in triplets], dtype=np.float64)
        h.attrs["times_ns_time_scale"] = "UTC"
        h.attrs["source_time_zone"] = "Beijing local time (UTC+8)"
        h.attrs["source_time_correction"] = "legacy event index times are corrected by subtracting 8 hours"


def main():
    events = load_events()
    triplets = pair_tristatic(events)
    make_plot(events, triplets)
    stage_tristatic_files(triplets)
    write_tristatic_index(triplets)
    print(f"Total events: {len(events)}")
    print(f"Tristatic triplets: {len(triplets)}")
    print(PLOT_PNG)
    print(TRISTATIC_INDEX)
    print(TRISTATIC_DIR)


if __name__ == "__main__":
    main()
