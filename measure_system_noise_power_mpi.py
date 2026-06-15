#!/usr/bin/env python3
"""Measure per-pulse raw-voltage noise power for the Sanya 4 MHz experiment.

This MPI program is intended to run on revontuli:

    mpiexec -n 48 python3 measure_system_noise_power_mpi.py

For every MATLAB v7.3 raw-voltage file, it computes the average raw-voltage
power for each pulse,

    P_noise[pulse] = mean_range(real(data_raw)^2 + imag(data_raw)^2).

The output is HDF5 only.  Per-rank shard files are written first, then rank 0
merges them into one compact HDF5 product.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI


SITE_ORDER = ("Sanya", "Danzhou", "Wenchang")
SITE_ID = {site: idx for idx, site in enumerate(SITE_ORDER)}
SOURCE_TIMEZONE_OFFSET_NS = np.int64(8 * 3600 * 1_000_000_000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/mnt/data/juha/SANYA/Juha/20240422")
    p.add_argument("--output-dir", default="/mnt/data/juha/sanya/system_noise_power_4mhz")
    p.add_argument("--output-h5", default="sanya_4mhz_system_noise_power_per_pulse.h5")
    p.add_argument("--chunk-pulses", type=int, default=512)
    p.add_argument("--max-files", type=int, default=0, help="Process only the first N discovered files for smoke tests.")
    p.add_argument("--keep-shards", action="store_true")
    return p.parse_args()


def discover_files(data_root: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for site in SITE_ORDER:
        for path in sorted((Path(data_root) / site).glob("*.mat")):
            out.append((site, str(path)))
    return out


def datetime64_ns_from_matlab_time(tm: np.ndarray, idx: int) -> np.int64:
    year = int(tm[0, idx] + 2000)
    month = int(tm[1, idx])
    day = int(tm[2, idx])
    hour = int(tm[3, idx])
    minute = int(tm[4, idx])
    sec_float = float(tm[5, idx])
    sec = int(np.floor(sec_float))
    frac_ns = int(round((sec_float - sec) * 1e9))
    dt = (
        np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}", "ns")
        + np.timedelta64(sec, "s")
        + np.timedelta64(frac_ns, "ns")
    )
    return np.int64((dt - np.datetime64("1970-01-01T00:00:00", "ns")).astype("timedelta64[ns]").astype(np.int64))


def matlab_time_ns(tm: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.empty(stop - start, dtype=np.int64)
    for out_idx, pulse_idx in enumerate(range(start, stop)):
        local[out_idx] = datetime64_ns_from_matlab_time(tm, pulse_idx)
    return local, local - SOURCE_TIMEZONE_OFFSET_NS


def process_file(site: str, path: str, chunk_pulses: int) -> tuple[dict, dict | None]:
    with h5py.File(path, "r") as h:
        raw = h["data_raw"]
        time = h["time"]
        n_range, n_pulse = raw.shape
        noise = np.empty(n_pulse, dtype=np.float64)
        local_ns = np.empty(n_pulse, dtype=np.int64)
        utc_ns = np.empty(n_pulse, dtype=np.int64)
        for start in range(0, n_pulse, chunk_pulses):
            stop = min(n_pulse, start + chunk_pulses)
            block = raw[:, start:stop]
            real = block["real"].astype(np.float64, copy=False)
            imag = block["imag"].astype(np.float64, copy=False)
            noise[start:stop] = np.mean(real * real + imag * imag, axis=0, dtype=np.float64)
            local_ns[start:stop], utc_ns[start:stop] = matlab_time_ns(time, start, stop)

    file_row = {
        "site": site,
        "source_file": path,
        "n_range_samples": int(n_range),
        "n_pulses": int(n_pulse),
        "time_utc_start_ns": int(utc_ns[0]) if n_pulse else -1,
        "time_utc_end_ns": int(utc_ns[-1]) if n_pulse else -1,
        "mean_noise_power": float(np.nanmean(noise)),
        "median_noise_power": float(np.nanmedian(noise)),
        "std_noise_power": float(np.nanstd(noise)),
    }
    pulse_row = {
        "station_id": np.full(n_pulse, SITE_ID[site], dtype=np.uint8),
        "pulse_index": np.arange(n_pulse, dtype=np.uint32),
        "time_beijing_local_ns": local_ns,
        "time_utc_ns": utc_ns,
        "noise_power_mean_raw_voltage": noise,
    }
    return file_row, pulse_row


def write_shard(path: str, file_rows: list[dict], pulse_rows: list[dict], errors: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Per-pulse system noise power shard: mean(real(raw)^2 + imag(raw)^2) over range samples."
        h.attrs["units"] = "raw ADC voltage-squared units"
        h.attrs["n_files"] = len(file_rows)
        h.attrs["n_errors"] = len(errors)
        files = h.create_group("files")
        files["site"] = np.asarray([row["site"] for row in file_rows], dtype=string_dtype)
        files["source_file"] = np.asarray([row["source_file"] for row in file_rows], dtype=string_dtype)
        for key in (
            "n_range_samples",
            "n_pulses",
            "time_utc_start_ns",
            "time_utc_end_ns",
            "mean_noise_power",
            "median_noise_power",
            "std_noise_power",
        ):
            files[key] = np.asarray([row[key] for row in file_rows])

        n_total = int(sum(len(row["noise_power_mean_raw_voltage"]) for row in pulse_rows))
        pulses = h.create_group("pulses")
        pulses["file_index"] = np.empty(n_total, dtype=np.uint32)
        pulses["station_id"] = np.empty(n_total, dtype=np.uint8)
        pulses["pulse_index"] = np.empty(n_total, dtype=np.uint32)
        pulses["time_beijing_local_ns"] = np.empty(n_total, dtype=np.int64)
        pulses["time_utc_ns"] = np.empty(n_total, dtype=np.int64)
        pulses["noise_power_mean_raw_voltage"] = np.empty(n_total, dtype=np.float64)
        cursor = 0
        for file_idx, row in enumerate(pulse_rows):
            n = len(row["noise_power_mean_raw_voltage"])
            sl = slice(cursor, cursor + n)
            pulses["file_index"][sl] = file_idx
            for key in ("station_id", "pulse_index", "time_beijing_local_ns", "time_utc_ns", "noise_power_mean_raw_voltage"):
                pulses[key][sl] = row[key]
            cursor += n

        errors_group = h.create_group("errors")
        errors_group["site"] = np.asarray([row["site"] for row in errors], dtype=string_dtype)
        errors_group["source_file"] = np.asarray([row["source_file"] for row in errors], dtype=string_dtype)
        errors_group["message"] = np.asarray([row["message"] for row in errors], dtype=string_dtype)


def merge_shards(output_path: str, shard_paths: list[str], args: argparse.Namespace, n_ranks: int) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    file_counts = []
    pulse_counts = []
    error_counts = []
    for path in shard_paths:
        with h5py.File(path, "r") as h:
            file_counts.append(len(h["files/source_file"]))
            pulse_counts.append(len(h["pulses/noise_power_mean_raw_voltage"]))
            error_counts.append(len(h["errors/source_file"]))

    n_files = int(sum(file_counts))
    n_pulses = int(sum(pulse_counts))
    n_errors = int(sum(error_counts))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as out:
        out.attrs["description"] = "Sanya 4 MHz per-pulse system noise power from raw voltage files."
        out.attrs["definition"] = "noise_power_mean_raw_voltage = mean(real(data_raw)^2 + imag(data_raw)^2) over all recorded range samples for one pulse."
        out.attrs["units"] = "raw ADC voltage-squared units"
        out.attrs["source_data_root"] = args.data_root
        out.attrs["mpi_ranks"] = n_ranks
        out.attrs["chunk_pulses"] = args.chunk_pulses
        out.attrs["source_time_zone"] = "Beijing local time (UTC+8)"
        out.attrs["time_utc_ns"] = "time_beijing_local_ns - 8 hours"

        out["site_names"] = np.asarray(SITE_ORDER, dtype=string_dtype)
        files = out.create_group("files")
        files.create_dataset("site", shape=(n_files,), dtype=string_dtype)
        files.create_dataset("source_file", shape=(n_files,), dtype=string_dtype)
        for key, dtype in (
            ("n_range_samples", np.int64),
            ("n_pulses", np.int64),
            ("time_utc_start_ns", np.int64),
            ("time_utc_end_ns", np.int64),
            ("mean_noise_power", np.float64),
            ("median_noise_power", np.float64),
            ("std_noise_power", np.float64),
        ):
            files[key] = np.empty(n_files, dtype=dtype)

        pulses = out.create_group("pulses")
        pulses["file_index"] = np.empty(n_pulses, dtype=np.uint32)
        pulses["station_id"] = np.empty(n_pulses, dtype=np.uint8)
        pulses["pulse_index"] = np.empty(n_pulses, dtype=np.uint32)
        pulses["time_beijing_local_ns"] = np.empty(n_pulses, dtype=np.int64)
        pulses["time_utc_ns"] = np.empty(n_pulses, dtype=np.int64)
        pulses["noise_power_mean_raw_voltage"] = np.empty(n_pulses, dtype=np.float64)

        errors = out.create_group("errors")
        errors.create_dataset("site", shape=(n_errors,), dtype=string_dtype)
        errors.create_dataset("source_file", shape=(n_errors,), dtype=string_dtype)
        errors.create_dataset("message", shape=(n_errors,), dtype=string_dtype)

        file_cursor = 0
        pulse_cursor = 0
        error_cursor = 0
        for path in shard_paths:
            with h5py.File(path, "r") as h:
                nf = len(h["files/source_file"])
                npulse = len(h["pulses/noise_power_mean_raw_voltage"])
                ne = len(h["errors/source_file"])
                fsl = slice(file_cursor, file_cursor + nf)
                psl = slice(pulse_cursor, pulse_cursor + npulse)
                esl = slice(error_cursor, error_cursor + ne)

                for key in h["files"].keys():
                    files[key][fsl] = h["files"][key][:]
                for key in h["pulses"].keys():
                    values = h["pulses"][key][:]
                    if key == "file_index":
                        values = values + file_cursor
                    pulses[key][psl] = values
                for key in h["errors"].keys():
                    errors[key][esl] = h["errors"][key][:]
                file_cursor += nf
                pulse_cursor += npulse
                error_cursor += ne


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    files = discover_files(args.data_root)
    if args.max_files > 0:
        files = files[: args.max_files]
    shard_dir = os.path.join(args.output_dir, "shards")
    shard_path = os.path.join(shard_dir, f"rank_{rank:04d}.h5")
    assigned = files[rank::size]
    file_rows: list[dict] = []
    pulse_rows: list[dict] = []
    errors: list[dict] = []

    for local_idx, (site, path) in enumerate(assigned, start=1):
        try:
            file_row, pulse_row = process_file(site, path, args.chunk_pulses)
            file_rows.append(file_row)
            pulse_rows.append(pulse_row)
            print(f"rank {rank:03d}/{size:03d} {local_idx:04d}/{len(assigned):04d} ok {site} {os.path.basename(path)}", flush=True)
        except Exception as exc:
            errors.append({"site": site, "source_file": path, "message": repr(exc) + "\n" + traceback.format_exc()})
            print(f"rank {rank:03d}/{size:03d} ERROR {site} {path}: {exc!r}", flush=True)

    write_shard(shard_path, file_rows, pulse_rows, errors)
    comm.Barrier()

    if rank == 0:
        shard_paths = [os.path.join(shard_dir, f"rank_{rr:04d}.h5") for rr in range(size)]
        output_path = os.path.join(args.output_dir, args.output_h5)
        merge_shards(output_path, shard_paths, args, size)
        if not args.keep_shards:
            for path in shard_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
