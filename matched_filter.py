import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as n


REFERENCE_CHIRP_RATE_SCALE = 1.0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

epoch = n.datetime64("1970-01-01T00:00:00", "ns")
C = 299792458.0
SOURCE_TIMEZONE_NAME = "Beijing local time"
SOURCE_TIMEZONE_OFFSET_HOURS = 8
SOURCE_TIMEZONE_OFFSET = n.timedelta64(SOURCE_TIMEZONE_OFFSET_HOURS, "h")
DEFAULT_RESULTS_DIR = os.path.expanduser("~/src/lfm_meteor/results")
DEFAULT_DATA_ROOT = "/mnt/data/juha/SANYA/Juha/20240422"
SITE_DIRS = {
    "sanya": "Sanya",
    "wenchang": "Wenchang",
    "danzhou": "Danzhou",
}
SITE_FIRST_SAMPLE_DELAY_US = {
    "sanya": 466.32,
    "danzhou": 438.426,
    "wenchang": 430.906,
}


def delay_us_to_range_km(delay_us):
    return 0.5 * float(delay_us) * 1e-6 * C / 1e3


def site_first_sample_r0_km(site):
    return delay_us_to_range_km(SITE_FIRST_SAMPLE_DELAY_US[site])


def lfm(l=200, sr=4, bw=4e6, chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE):
    tidx = n.arange(l * sr) / (sr * 1e6)
    om = bw * 1e6 / float(l) / 2.0 * float(chirp_rate_scale)
    return n.array(n.exp(1j * 2 * n.pi * (tidx * bw / 2 - om * tidx**2.0)), dtype=n.complex64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched-filter the Sanya raw data, detect head echoes, and write RTIs for all events."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--snr-threshold", type=float, default=6.0)
    parser.add_argument("--min-echoes", type=int, default=6)
    parser.add_argument("--gap-ipps", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--site", choices=["all", "sanya", "wenchang", "danzhou"], default="all")
    return parser.parse_args()


def dt_from_time_array(tm, i):
    """Return the raw MATLAB timestamp, documented as Beijing local time (UTC+8).

    The MATLAB files contain three important variables:
    para: experiment configuration, including azimuth, elevation, pulse width,
        IPP, gate start/end, sampling rate, and bandwidth.
    time: Beijing local time (UTC+8), shaped [7, N] as
        [year_since_2000, month, day, hour, minute, second, code].
    data_raw: raw IQ voltage data, indexed by range sample and pulse time.
    """
    base_dt = n.datetime64(
        f"{int((tm[0, i] + 2000)):04d}-{int(tm[1, i]):02d}-{int(tm[2, i]):02d}T"
        f"{int(tm[3, i]):02d}:{int(tm[4, i]):02d}"
    )
    whole_sec = int(n.floor(tm[5, i]))
    frac_ns = int(n.round(1e9 * (tm[5, i] - whole_sec)))
    return base_dt + n.timedelta64(whole_sec, "s") + n.timedelta64(frac_ns, "ns")


def event_id(site, dt0_ns):
    return f"{site}_{int(dt0_ns)}"


def write_event(output_dir, site, dt0, time_ns, beijing_local_time_ns, ranges_km, echoes, raw, rgs, meta, source_file):
    dt0_ns = int(time_ns[0])
    ev_id = event_id(site, dt0_ns)
    ev_dir = os.path.join(output_dir, "head_echoes", site)
    os.makedirs(ev_dir, exist_ok=True)

    dB = 10.0 * n.log10(n.maximum(n.abs(echoes.T) ** 2.0, 1e-12))
    dB = dB - n.nanmedian(dB)
    t_rel = time_ns.astype(n.float64) / 1e9 - time_ns[0] / 1e9
    center_range = ranges_km[int(n.median(rgs))]

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(t_rel, ranges_km, dB, vmin=0, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Power (dB)")
    ax.plot(t_rel, ranges_km[n.asarray(rgs, dtype=n.int32)], "w.", ms=3)
    ax.set_ylim([center_range - 10, center_range + 10])
    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("Range (km)")
    ax.set_title(f"{site} {str(dt0)} {SOURCE_TIMEZONE_NAME}")
    fig.tight_layout()
    png_path = os.path.join(ev_dir, f"{ev_id}.png")
    fig.savefig(png_path)
    plt.close(fig)

    h5_path = os.path.join(ev_dir, f"{ev_id}.h5")
    with h5py.File(h5_path, "w") as ho:
        ho["echoes"] = echoes
        ho["raw"] = raw
        ho["times_ns"] = time_ns
        ho["beijing_local_time_ns"] = beijing_local_time_ns
        ho["relative_time_s"] = t_rel
        ho.attrs["times_ns_time_scale"] = "UTC"
        ho.attrs["source_time_zone"] = f"{SOURCE_TIMEZONE_NAME} (UTC+{SOURCE_TIMEZONE_OFFSET_HOURS})"
        ho.attrs["source_time_correction"] = "times_ns = raw MATLAB time - 8 hours"
        ho.attrs["source_timezone_offset_hours"] = SOURCE_TIMEZONE_OFFSET_HOURS
        ho.attrs["raw_matlab_variable_para"] = (
            "Experiment configuration: azimuth, elevation, LFM pulse width, IPP, "
            "gate start/end, sampling rate, and bandwidth."
        )
        ho.attrs["raw_matlab_variable_time"] = (
            "Beijing local time (UTC+8), [year_since_2000, month, day, hour, minute, second, code]."
        )
        ho.attrs["raw_matlab_variable_data_raw"] = "Raw IQ voltage data indexed by range sample and pulse time."
        ho["r0"] = meta["r0_km"]
        ho["r1"] = meta["r1_km"]
        ho["site"] = site
        ho["az"] = meta["az_deg"]
        ho["el"] = meta["el_deg"]
        ho["range_gate"] = n.asarray(rgs, dtype=n.int32)
        ho["range_km"] = ranges_km[n.asarray(rgs, dtype=n.int32)]
        ho["ranges_km_axis"] = ranges_km
        ho["snr_peak_db"] = n.max(dB, axis=0)
        ho["sr_mhz"] = meta["sr_mhz"]
        ho["bw_mhz"] = meta["bw_mhz"]
        ho["ipp_us"] = meta["ipp_us"]
        ho["pulse_length_us"] = meta["pulse_length_us"]
        ho["source_file"] = n.bytes_(source_file)
        ho["event_id"] = n.bytes_(ev_id)
        ho["rti_png"] = n.bytes_(png_path)

    return {
        "event_id": ev_id,
        "site": site,
        "dt0_ns": dt0_ns,
        "dt1_ns": int(time_ns[-1]),
        "dt0_beijing_local_ns": int(beijing_local_time_ns[0]),
        "dt1_beijing_local_ns": int(beijing_local_time_ns[-1]),
        "n_echoes": int(len(rgs)),
        "median_range_km": float(center_range),
        "az_deg": float(meta["az_deg"]),
        "el_deg": float(meta["el_deg"]),
        "sr_mhz": float(meta["sr_mhz"]),
        "bw_mhz": float(meta["bw_mhz"]),
        "ipp_us": float(meta["ipp_us"]),
        "pulse_length_us": float(meta["pulse_length_us"]),
        "source_file": source_file,
        "event_h5": h5_path,
        "event_png": png_path,
    }


def flush_event(output_dir, site, dts, echoes, raw, rgs, ranges_km, meta, source_file, min_echoes):
    if len(echoes) < min_echoes:
        return None

    echoes = n.asarray(echoes)
    raw = n.asarray(raw)
    beijing_local_time_ns = (n.asarray(dts) - epoch).astype("int64")
    time_ns = ((n.asarray(dts) - SOURCE_TIMEZONE_OFFSET) - epoch).astype("int64")
    return write_event(
        output_dir=output_dir,
        site=site,
        dt0=dts[0],
        time_ns=time_ns,
        beijing_local_time_ns=beijing_local_time_ns,
        ranges_km=ranges_km,
        echoes=echoes,
        raw=raw,
        rgs=rgs,
        meta=meta,
        source_file=source_file,
    )


def read_site_file(path, site, output_dir, snr_threshold, min_echoes, gap_ipps):
    with h5py.File(path, "r") as h:
        zz = h["data_raw"][()]
        p = h["para"][()]
        tm = h["time"][()]

    az = float(p[6])
    el = float(p[7])
    pulse_length_us = float(p[9])
    ipp_us = float(p[11])
    raw_r0_km = float(p[12])
    r1_km = float(p[13])
    sr_mhz = float(p[14])
    bw_mhz = float(p[15])
    code = lfm(l=int(round(pulse_length_us)), sr=int(round(sr_mhz)), bw=bw_mhz * 1e6)

    z = n.array(zz["real"] + zz["imag"] * 1j, dtype=n.complex64)
    dr_km = C / (sr_mhz * 1e6) / 2.0 / 1e3
    r0_km = site_first_sample_r0_km(site)
    ranges_km = r0_km + dr_km * n.arange(z.shape[0], dtype=n.float64)
    zd = n.empty_like(z)

    detections = []
    echoes = []
    raw = []
    dts = []
    rgs = []
    prev = -gap_ipps - 1
    meta = {
        "az_deg": az,
        "el_deg": el,
        "pulse_length_us": pulse_length_us,
        "ipp_us": ipp_us,
        "r0_km": r0_km,
        "raw_r0_km": raw_r0_km,
        "r1_km": r1_km,
        "sr_mhz": sr_mhz,
        "bw_mhz": bw_mhz,
    }

    for i in range(z.shape[1]):
        zd[:, i] = n.convolve(z[:, i], n.conj(code), mode="same")
        noise = n.median(n.abs(zd[:, i]))
        peak = n.max(n.abs(zd[:, i]))
        if noise <= 0 or peak / noise <= snr_threshold:
            continue

        rgmax = int(n.argmax(n.abs(zd[:, i])))
        if i - prev > gap_ipps:
            detection = flush_event(
                output_dir=output_dir,
                site=site,
                dts=dts,
                echoes=echoes,
                raw=raw,
                rgs=rgs,
                ranges_km=ranges_km,
                meta=meta,
                source_file=path,
                min_echoes=min_echoes,
            )
            if detection is not None:
                detections.append(detection)
            echoes = []
            raw = []
            dts = []
            rgs = []

        dt = dt_from_time_array(tm, i)
        dts.append(dt)
        echoes.append(zd[:, i])
        raw.append(z[:, i])
        rgs.append(rgmax)
        prev = i

    detection = flush_event(
        output_dir=output_dir,
        site=site,
        dts=dts,
        echoes=echoes,
        raw=raw,
        rgs=rgs,
        ranges_km=ranges_km,
        meta=meta,
        source_file=path,
        min_echoes=min_echoes,
    )
    if detection is not None:
        detections.append(detection)
    return detections


def write_index(results_dir, detections):
    index_path = os.path.join(results_dir, "head_echoes", "head_echo_index.h5")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(index_path, "w") as h:
        h["event_id"] = n.asarray([d["event_id"] for d in detections], dtype=string_dtype)
        h["site"] = n.asarray([d["site"] for d in detections], dtype=string_dtype)
        h["dt0_ns"] = n.asarray([d["dt0_ns"] for d in detections], dtype=n.int64)
        h["dt1_ns"] = n.asarray([d["dt1_ns"] for d in detections], dtype=n.int64)
        h["dt0_beijing_local_ns"] = n.asarray([d["dt0_beijing_local_ns"] for d in detections], dtype=n.int64)
        h["dt1_beijing_local_ns"] = n.asarray([d["dt1_beijing_local_ns"] for d in detections], dtype=n.int64)
        h["n_echoes"] = n.asarray([d["n_echoes"] for d in detections], dtype=n.int32)
        h["median_range_km"] = n.asarray([d["median_range_km"] for d in detections], dtype=n.float64)
        h["az_deg"] = n.asarray([d["az_deg"] for d in detections], dtype=n.float64)
        h["el_deg"] = n.asarray([d["el_deg"] for d in detections], dtype=n.float64)
        h["sr_mhz"] = n.asarray([d["sr_mhz"] for d in detections], dtype=n.float64)
        h["bw_mhz"] = n.asarray([d["bw_mhz"] for d in detections], dtype=n.float64)
        h["ipp_us"] = n.asarray([d["ipp_us"] for d in detections], dtype=n.float64)
        h["pulse_length_us"] = n.asarray([d["pulse_length_us"] for d in detections], dtype=n.float64)
        h["source_file"] = n.asarray([d["source_file"] for d in detections], dtype=string_dtype)
        h["event_h5"] = n.asarray([d["event_h5"] for d in detections], dtype=string_dtype)
        h["event_png"] = n.asarray([d["event_png"] for d in detections], dtype=string_dtype)
        h.attrs["times_ns_time_scale"] = "UTC"
        h.attrs["source_time_zone"] = f"{SOURCE_TIMEZONE_NAME} (UTC+{SOURCE_TIMEZONE_OFFSET_HOURS})"
        h.attrs["source_time_correction"] = "dt0_ns/dt1_ns = raw MATLAB time - 8 hours"
    return index_path


def get_site_files(data_root, site, max_files):
    sites = [site] if site != "all" else list(SITE_DIRS.keys())
    site_files = []
    for site_name in sites:
        pattern = os.path.join(data_root, SITE_DIRS[site_name], "*.mat")
        files = sorted(glob.glob(pattern))
        if max_files > 0:
            files = files[:max_files]
        site_files.extend([(site_name, path) for path in files])
    site_files.sort(key=lambda item: item[1])
    return site_files


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.results_dir, "head_echoes"), exist_ok=True)

    site_files = get_site_files(args.data_root, args.site, args.max_files)
    if not site_files:
        raise SystemExit(f"No input files found under {args.data_root}")

    local_detections = []
    for idx in range(rank, len(site_files), size):
        site, path = site_files[idx]
        print(f"[rank {rank}] processing {site} {path}")
        local_detections.extend(
            read_site_file(
                path=path,
                site=site,
                output_dir=args.results_dir,
                snr_threshold=args.snr_threshold,
                min_echoes=args.min_echoes,
                gap_ipps=args.gap_ipps,
            )
        )

    gathered = comm.gather(local_detections, root=0)
    if rank == 0:
        detections = [d for group in gathered for d in group]
        detections.sort(key=lambda item: (item["dt0_ns"], item["site"]))
        index_path = write_index(args.results_dir, detections)
        print(f"Wrote {len(detections)} head-echo RTIs")
        print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
