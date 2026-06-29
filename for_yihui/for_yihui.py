"""Very small rank-02 tri-static example.

What this script does:

1. Read three MATLAB v7.3 raw voltage files.
2. Matched filter each pulse with the LFM code.
3. Pick the rank-02 echo event and extract peak delays.
4. Triangulate each echo in ITRF/ECEF.
5. Plot latitude/longitude map and two height-section panels.

Run from this folder:

    python for_yihui.py
"""

import json
import os
import bz2
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def progress(message):
    print(f"[for_yihui] {message}", flush=True)


progress("starting Python imports")

import h5py
import jcoord
import numpy as np

from rangedelay import predict_station_total_path

progress("finished Python imports")


C = 299792458.0
C_KM_S = C / 1e3

# Nominal Sanya first range gate used as the monostatic r0 in the remote
# bistatic path convention:
#   total_path_km = 2*r0 + c*gate/fs + delay*c
DEFAULT_MONOSTATIC_R0_KM = 69.9

DATA = {
    "sanya": "data/sanya_rank02.mat",
    "danzhou": "data/danzhou_rank02.mat",
    "wenchang": "data/wenchang_rank02.mat",
}

TARGET_EVENT_START_UTC_NS = {
    "sanya": 1713821283054349899,
    "danzhou": 1713821283119349957,
    "wenchang": 1713821283129349947,
}

SITE_LLH = {
    "sanya": (18.3492, 109.6222, 50.0),
    "danzhou": (19.5281, 109.1322, 99.9),
    "wenchang": (19.5982, 110.7908, 24.9),
}

SITE_ECEF = {k: np.asarray(jcoord.geodetic2ecef(*v), dtype=float) for k, v in SITE_LLH.items()}
TX = SITE_ECEF["sanya"]
ORDER = ("sanya", "danzhou", "wenchang")
OUT = "results"


@contextmanager
def open_matlab_file(path):
    """Open either a plain HDF5 .mat file or a .mat.bz2 copy.

    The compressed files are useful for sending the example around. HDF5 needs
    a seekable file, so the .bz2 payload is expanded once to the matching .mat
    file and reused on later runs. Both .mat and .mat.bz2 are ignored by git.
    """
    if os.path.exists(path):
        progress(f"opening {path}")
        with h5py.File(path, "r") as h:
            yield h
        return

    compressed_path = path + ".bz2"
    if not os.path.exists(compressed_path):
        raise FileNotFoundError(f"Missing {path} or {compressed_path}")

    progress(f"decompressing {compressed_path} to {path}")
    with bz2.open(compressed_path, "rb") as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    progress(f"opening {path}")
    with h5py.File(path, "r") as h:
        yield h


def gate_to_total_path_m(site, gate, site_data):
    """Convert matched-filter gate to total tx-target-rx path length.

    Sanya is monostatic, so its MATLAB first range gate is a one-way range.
    Danzhou/Wenchang use the nominal bistatic delay predicted from the Sanya
    transmit beam and the remote receive beam:

        total_path_km = 2*r0 + c*gate/fs + delay*c

    where r0 is the Sanya monostatic first range gate and delay is predicted
    by rangedelay.py from the common-volume beam geometry.
    """
    gate = np.asarray(gate, dtype=np.float64)
    sample_rate_mhz = float(site_data["sample_rate_mhz"])
    if site == "sanya":
        one_way_range_km = site_data["first_range_gate_km"] + gate * C / (2.0 * sample_rate_mhz * 1e6) / 1e3
        return 2.0 * one_way_range_km * 1e3
    if site == "danzhou":
        total_path_km = (
            2.0 * site_data["monostatic_r0_km"]
            + C_KM_S * gate / (sample_rate_mhz * 1e6)
            + C_KM_S * site_data["rangedelay_delay_s"]
        )
        return total_path_km * 1e3
    if site == "wenchang":
        total_path_km = (
            2.0 * site_data["monostatic_r0_km"]
            + C_KM_S * gate / (sample_rate_mhz * 1e6)
            + C_KM_S * site_data["rangedelay_delay_s"]
        )
        return total_path_km * 1e3
    raise ValueError(f"Unknown site {site}")


def lfm_code(sample_rate_mhz, bandwidth_mhz, pulse_us):
    """Generate the same baseband down-chirp convention used by the receiver."""
    n = int(round(sample_rate_mhz * pulse_us))
    t = np.arange(n) / (sample_rate_mhz * 1e6)
    bandwidth_hz = bandwidth_mhz * 1e6
    sweep = bandwidth_hz / (pulse_us * 1e-6) / 2.0
    return np.exp(1j * 2.0 * np.pi * (t * bandwidth_hz / 2.0 - sweep * t**2))


def matlab_time_to_utc_ns(time, i):
    year = int(time[0, i] + 2000)
    month = int(time[1, i])
    day = int(time[2, i])
    hour = int(time[3, i])
    minute = int(time[4, i])
    second = float(time[5, i])
    whole = int(np.floor(second))
    frac_ns = int(round((second - whole) * 1e9))
    beijing = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{whole:02d}", "ns")
    utc = beijing + np.timedelta64(frac_ns, "ns") - np.timedelta64(8, "h")
    return int((utc - np.datetime64("1970-01-01T00:00:00", "ns")).astype("int64"))


def read_matlab(site):
    path = DATA[site]
    progress(f"reading {site} MATLAB data")
    with open_matlab_file(path) as h:
        raw_struct = h["data_raw"][()]
        para = np.ravel(h["para"][()])
        time = h["time"][()]
        pulse_offset = int(h["pulse_offset"][()]) if "pulse_offset" in h else 0

    # MATLAB v7.3 stores the complex voltage as an HDF5 compound array with
    # separate real and imaginary fields.
    raw = np.asarray(raw_struct["real"] + 1j * raw_struct["imag"], dtype=np.complex64)
    sample_rate_mhz = float(para[14])

    # The MATLAB files store radar settings in para.  The entries used here are:
    # para[6]/para[7] = beam azimuth/elevation in degrees,
    # para[12]/para[13] = first/last range gate in km,
    # para[14]/para[15] = sample rate and LFM bandwidth in MHz.
    # Keep the range-gate metadata visible. Sanya uses first_range_gate_km in
    # gate_to_total_path_m(); the remote sites use calibrated delay constants.
    out = {
        "raw": raw,
        "time_ns": np.asarray([matlab_time_to_utc_ns(time, i) for i in range(raw.shape[1])], dtype=np.int64),
        "az": float(para[6]),
        "el": float(para[7]),
        "pulse_us": float(para[9]),
        "first_range_gate_km": float(para[12]),
        "last_range_gate_km": float(para[13]),
        "sample_rate_mhz": sample_rate_mhz,
        "bandwidth_mhz": float(para[15]),
        "first_sample_delay_us": None,
        "pulse_offset": pulse_offset,
    }
    progress(
        f"{site}: raw shape {raw.shape[0]} range gates x {raw.shape[1]} pulses; "
        f"az/el {out['az']:.3f}/{out['el']:.3f} deg; pulse_offset {pulse_offset}"
    )
    return out


def matched_filter(site, data, threshold=6.0):
    progress(f"{site}: matched filtering {data['raw'].shape[1]} pulses")
    code = lfm_code(data["sample_rate_mhz"], data["bandwidth_mhz"], data["pulse_us"])
    detections = []
    rti_columns_db = []
    for pulse in range(data["raw"].shape[1]):
        # Plain convolution matched filter: no Doppler prediction, no second
        # interpolated pass, and no fitted trajectory information.
        z = np.convolve(data["raw"][:, pulse], np.conj(code), mode="same")
        amp = np.abs(z)
        rti_columns_db.append(20.0 * np.log10(np.maximum(amp, 1e-12)))
        noise = np.median(amp)
        peak = np.max(amp)
        if noise <= 0.0 or peak / noise < threshold:
            continue
        gate = int(np.argmax(amp))

        # Convert the peak gate into the same total path coordinate used by the
        # production triangulation scripts.
        total_path_m = gate_to_total_path_m(site, gate, data)
        delay_us = total_path_m / C * 1e6
        detections.append(
            {
                "pulse": pulse,
                "original_pulse": pulse + data["pulse_offset"],
                "time_ns": int(data["time_ns"][pulse]),
                "gate": gate,
                "delay_us": float(delay_us),
                "total_path_m": float(total_path_m),
                "snr": float(peak / noise),
            }
        )
    data["matched_filter_rti_db"] = np.asarray(rti_columns_db, dtype=np.float32).T
    progress(f"{site}: found {len(detections)} detections above SNR threshold {threshold:g}")
    return detections


def split_events(detections, max_gap_pulses=10):
    events = []
    event = []
    last = None
    for d in detections:
        if last is None or d["pulse"] - last <= max_gap_pulses:
            event.append(d)
        else:
            events.append(event)
            event = [d]
        last = d["pulse"]
    if event:
        events.append(event)
    return events


def choose_rank02_event(site, detections):
    events = split_events(detections)
    target = TARGET_EVENT_START_UTC_NS[site]
    event = min(events, key=lambda e: abs(e[0]["time_ns"] - target))
    progress(
        f"{site}: selected rank-02 event with {len(event)} pulses "
        f"(local pulses {event[0]['pulse']}..{event[-1]['pulse']})"
    )
    return event


def print_range_gate_report(events):
    print("\nRange gates for selected rank-02 event", flush=True)
    print("---------------------------------------", flush=True)
    for site in ORDER:
        gates = np.asarray([d["gate"] for d in events[site]], dtype=int)
        pulses = np.asarray([d["pulse"] for d in events[site]], dtype=int)
        original_pulses = np.asarray([d["original_pulse"] for d in events[site]], dtype=int)
        print(
            f"{site:8s}: gates {gates.min()}..{gates.max()} "
            f"(median {np.median(gates):.1f}); "
            f"local pulses {pulses.min()}..{pulses.max()}; "
            f"original pulses {original_pulses.min()}..{original_pulses.max()}",
            flush=True,
        )


def nearest(event, time_ns):
    times = np.asarray([d["time_ns"] for d in event])
    return event[int(np.argmin(np.abs(times - time_ns)))]


def match_three_sites(events):
    progress("matching Sanya, Danzhou, and Wenchang pulses by UTC time")
    measurements = []
    times = []
    used = []
    for s in events["sanya"]:
        # Pair each Sanya pulse with the nearest Danzhou/Wenchang pulse in UTC.
        # This keeps the example simple and avoids any trajectory fitting.
        d = nearest(events["danzhou"], s["time_ns"])
        w = nearest(events["wenchang"], s["time_ns"])
        if abs(d["time_ns"] - s["time_ns"]) > 7.5e6:
            continue
        if abs(w["time_ns"] - s["time_ns"]) > 7.5e6:
            continue
        measurements.append([s["total_path_m"], d["total_path_m"], w["total_path_m"]])
        times.append(int(round((s["time_ns"] + d["time_ns"] + w["time_ns"]) / 3.0)))
        used.append([s, d, w])
    progress(f"matched {len(measurements)} tri-static pulse triplets")
    return np.asarray(measurements), np.asarray(times, dtype=np.int64), used


def predicted_paths(x):
    """Total Sanya-transmit to target to receiver path for each station."""
    return np.asarray(
        [
            np.linalg.norm(x - TX) + np.linalg.norm(x - SITE_ECEF["sanya"]),
            np.linalg.norm(x - TX) + np.linalg.norm(x - SITE_ECEF["danzhou"]),
            np.linalg.norm(x - TX) + np.linalg.norm(x - SITE_ECEF["wenchang"]),
        ]
    )


def triangulate(measurements):
    progress("importing scipy optimizer")
    import scipy.optimize as opt

    progress(f"triangulating {len(measurements)} ECEF points")
    # Initial guess: somewhere along the Sanya beam near the median monostatic range.
    range_m = np.median(measurements[:, 0]) / 2.0
    lat, lon, alt = jcoord.az_el_r2geodetic(*SITE_LLH["sanya"], 15.0, 75.0, range_m)
    x0 = np.asarray(jcoord.geodetic2ecef(lat, lon, alt), dtype=float)

    points = []
    fit_success = []
    fit_cost = []
    for m in measurements:
        # Solve the three path-length equations directly in ECEF/ITRF.  There is
        # no fitted line here; each pulse gets its own independent point.
        fit = opt.least_squares(lambda x: predicted_paths(x) - m, x0, method="lm")
        x0 = fit.x
        points.append(x0)
        fit_success.append(bool(fit.success))
        fit_cost.append(float(fit.cost))
    points = np.asarray(points)
    llh = np.asarray([jcoord.ecef2geodetic(*p) for p in points])
    progress("triangulation finished")
    residuals_m = np.asarray([predicted_paths(p) - m for p, m in zip(points, measurements)])
    fit_info = {
        "success_count": int(np.sum(fit_success)),
        "total_count": int(len(fit_success)),
        "mean_least_squares_cost_m2": float(np.mean(fit_cost)),
    }
    return points, llh[:, 0], llh[:, 1], llh[:, 2] / 1e3, residuals_m, fit_info


def print_fit_report(height, residuals_m, fit_info, data):
    residual_norm_m = np.linalg.norm(residuals_m, axis=1)
    all_link_rms_m = np.sqrt(np.mean(residuals_m**2))
    print("\nFit diagnostics", flush=True)
    print("---------------", flush=True)
    print(f"Tri-static points: {len(height)}", flush=True)
    print(f"Least-squares converged: {fit_info['success_count']} / {fit_info['total_count']}", flush=True)
    print(
        f"Height: min {np.min(height):.3f} km, median {np.median(height):.3f} km, "
        f"max {np.max(height):.3f} km",
        flush=True,
    )
    print(f"Triangulation equation RMS over all links: {all_link_rms_m:.3f} m", flush=True)
    print(
        f"Path residual norm: RMS {np.sqrt(np.mean(residual_norm_m**2)):.3f} m, "
        f"median {np.median(residual_norm_m):.3f} m, max {np.max(residual_norm_m):.3f} m",
        flush=True,
    )
    print(
        "Path residual norm percentiles: "
        f"p10 {np.percentile(residual_norm_m, 10):.3f} m, "
        f"p50 {np.percentile(residual_norm_m, 50):.3f} m, "
        f"p90 {np.percentile(residual_norm_m, 90):.3f} m",
        flush=True,
    )
    for i, site in enumerate(ORDER):
        station_residual = residuals_m[:, i]
        print(
            f"{site:8s}: residual bias {np.mean(station_residual):+.6f} m, "
            f"RMS {np.sqrt(np.mean(station_residual**2)):.3f} m",
            flush=True,
        )


def beam_line(site, data, max_height_km=150):
    """Beam centerline from the station ground point up to 150 km altitude."""
    ranges_m = np.linspace(0.0, 500e3, 1000)
    lat = []
    lon = []
    height = []
    site_lat, site_lon, site_alt = SITE_LLH[site]
    for r in ranges_m:
        p = jcoord.az_el_r2geodetic(site_lat, site_lon, site_alt, data[site]["az"], data[site]["el"], r)
        lat.append(p[0])
        lon.append(p[1])
        height.append(p[2] / 1e3)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    height = np.asarray(height)
    keep = height <= max_height_km
    return lat[keep], lon[keep], height[keep]


def make_plots(data, lat, lon, height):
    progress("importing matplotlib")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)

    progress(f"writing {os.path.join(OUT, 'rank02_map.png')}")
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    sc = ax.scatter(lon, lat, c=height, s=20, cmap="viridis")
    for site, (site_lat, site_lon, _site_alt) in SITE_LLH.items():
        ax.plot(site_lon, site_lat, "^", ms=8, label=site)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Rank-02 simple triangulation result")
    ax.grid(True, alpha=0.3)
    ax.legend()
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Height (km)")
    fig.savefig(os.path.join(OUT, "rank02_map.png"), dpi=220)
    plt.close(fig)

    progress(f"writing {os.path.join(OUT, 'rank02_height_sections.png')}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, x, xname in [(axes[0], lon, "Longitude"), (axes[1], lat, "Latitude")]:
        for site, color in [("sanya", "black"), ("danzhou", "#d95f02"), ("wenchang", "#1b9e77")]:
            b_lat, b_lon, b_h = beam_line(site, data)
            b_x = b_lon if xname == "Longitude" else b_lat
            ax.plot(b_x, b_h, color=color, lw=1.5, label=f"{site} beam")
        ax.scatter(x, height, s=18, color="#377eb8", label="echo points", zorder=3)
        ax.set_xlabel(f"{xname} (deg)")
        ax.set_ylabel("Height (km)")
        ax.set_title(f"{xname} vs height: simple delay-origin test")
        ax.set_ylim(0, 150)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "rank02_height_sections.png"), dpi=220)
    plt.close(fig)


def make_rti_plots(data, events):
    progress("importing matplotlib for RTI plots")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    for site in ORDER:
        rti_db = data[site]["matched_filter_rti_db"]
        # Robust color limits keep one bright meteor echo from washing out the
        # background. These are display limits only; the peak gates are still
        # taken from the full matched-filter amplitude.
        vmin, vmax = np.percentile(rti_db, [5, 99.7])
        pulses = np.asarray([d["pulse"] for d in events[site]], dtype=int)
        gates = np.asarray([d["gate"] for d in events[site]], dtype=int)

        fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
        im = ax.imshow(
            rti_db,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            extent=[-0.5, rti_db.shape[1] - 0.5, -0.5, rti_db.shape[0] - 0.5],
        )
        ax.plot(pulses, gates, "r.", ms=4, alpha=0.2, label="selected peak gates")
        ax.set_xlabel("Local pulse number in compact file")
        ax.set_ylabel("Range gate")
        ax.set_title(f"{site.capitalize()} LFM matched-filter RTI")
        ax.legend(loc="upper right", fontsize=8)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Matched-filter amplitude (dB)")
        out_path = os.path.join(OUT, f"rank02_rti_{site}.png")
        progress(f"writing {out_path}")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)


def main():
    progress("starting rank-02 example workflow")
    data = {site: read_matlab(site) for site in ORDER}
    monostatic_r0_km = data["sanya"].get("first_range_gate_km", DEFAULT_MONOSTATIC_R0_KM)
    for site_data in data.values():
        site_data["monostatic_r0_km"] = monostatic_r0_km
    remote_predictions = {
        site: predict_station_total_path(site.capitalize(), monostatic_r0_km, 0.0, data[site]["sample_rate_mhz"] * 1e6)
        for site in ("danzhou", "wenchang")
    }
    for site, prediction in remote_predictions.items():
        data[site]["rangedelay_delay_s"] = prediction.delay_s
    progress(
        "range conversion: Sanya uses MATLAB r0 as one-way range; "
        "Danzhou/Wenchang use rangedelay.py beam-geometry delays "
        f"{remote_predictions['danzhou'].delay_s * 1e6:.3f}/"
        f"{remote_predictions['wenchang'].delay_s * 1e6:.3f} us"
    )
    detections = {site: matched_filter(site, data[site]) for site in ORDER}
    events = {site: choose_rank02_event(site, detections[site]) for site in ORDER}
    print_range_gate_report(events)
    measurements, times, used = match_three_sites(events)
    _points, lat, lon, height, residuals_m, fit_info = triangulate(measurements)
    make_plots(data, lat, lon, height)
    make_rti_plots(data, events)

    summary = {
        "n_points": int(len(height)),
        "height_min_km": float(np.min(height)),
        "height_median_km": float(np.median(height)),
        "height_max_km": float(np.max(height)),
        "processing_note": (
            "This is the deliberately simple handoff workflow: one LFM matched filter, "
            "peak-gate extraction, and direct tri-static triangulation. It uses the "
            "rangedelay.py beam-geometry prediction for Danzhou/Wenchang path offsets."
        ),
        "station_metadata": {
            site: {
                "az_deg_from_matlab": data[site]["az"],
                "el_deg_from_matlab": data[site]["el"],
                "first_range_gate_km_from_matlab": data[site]["first_range_gate_km"],
                "last_range_gate_km_from_matlab": data[site]["last_range_gate_km"],
                "monostatic_r0_km_used": monostatic_r0_km,
                "rangedelay_predicted_delay_us": None
                if site == "sanya"
                else remote_predictions[site].delay_s * 1e6,
                "rangedelay_rg_delay_km": None
                if site == "sanya"
                else remote_predictions[site].rg_delay,
                "gate0_total_path_km": None
                if site == "sanya"
                else remote_predictions[site].total_path_km,
                "pulse_offset_in_original_matlab_file": data[site]["pulse_offset"],
            }
            for site in ORDER
        },
        "outputs": [
            os.path.join(OUT, "rank02_map.png"),
            os.path.join(OUT, "rank02_height_sections.png"),
            os.path.join(OUT, "rank02_rti_sanya.png"),
            os.path.join(OUT, "rank02_rti_danzhou.png"),
            os.path.join(OUT, "rank02_rti_wenchang.png"),
        ],
    }
    with open(os.path.join(OUT, "rank02_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    progress(f"wrote {os.path.join(OUT, 'rank02_summary.json')}")
    print_fit_report(height, residuals_m, fit_info, data)


if __name__ == "__main__":
    main()
