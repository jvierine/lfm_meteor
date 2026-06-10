import glob
import os

import h5py
import numpy as np


TRISTATIC_PATTERN = os.path.join("results", "tristatic_head_echoes", "*", "*.h5")
TRISTATIC_INDEX = os.path.join("results", "tristatic_event_index.h5")
SOURCE_TIMEZONE_OFFSET_HOURS = 8
SOURCE_TIMEZONE_OFFSET_NS = int(SOURCE_TIMEZONE_OFFSET_HOURS * 3600 * 1e9)


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def patch_event_file(path):
    with h5py.File(path, "r+") as h:
        if str(h.attrs.get("times_ns_time_scale", "")).upper() == "UTC":
            return False

        legacy_times_ns = h["times_ns"][()].astype(np.int64)
        utc_times_ns = legacy_times_ns - SOURCE_TIMEZONE_OFFSET_NS

        if "beijing_local_time_ns" in h:
            del h["beijing_local_time_ns"]
        h["beijing_local_time_ns"] = legacy_times_ns

        del h["times_ns"]
        h["times_ns"] = utc_times_ns

        relative_time_s = utc_times_ns.astype(np.float64) / 1e9 - utc_times_ns[0].astype(np.float64) / 1e9
        if "relative_time_s" in h:
            del h["relative_time_s"]
        h["relative_time_s"] = relative_time_s

        site = decode_scalar(h["site"][()]) if "site" in h else os.path.basename(os.path.dirname(path))
        event_id = f"{site}_{int(utc_times_ns[0])}"
        if "event_id" in h:
            del h["event_id"]
        h["event_id"] = np.bytes_(event_id)

        h.attrs["times_ns_time_scale"] = "UTC"
        h.attrs["source_time_zone"] = f"Beijing local time (UTC+{SOURCE_TIMEZONE_OFFSET_HOURS})"
        h.attrs["source_time_correction"] = "times_ns = legacy raw MATLAB local time - 8 hours"
        h.attrs["source_timezone_offset_hours"] = SOURCE_TIMEZONE_OFFSET_HOURS
        h.attrs["legacy_times_ns_dataset"] = "beijing_local_time_ns"
        h.attrs["raw_matlab_variable_time"] = (
            "Beijing local time (UTC+8), [year_since_2000, month, day, hour, minute, second, code]."
        )
        h.attrs["raw_matlab_variable_para"] = (
            "Experiment configuration: azimuth, elevation, LFM pulse width, IPP, "
            "gate start/end, sampling rate, and bandwidth."
        )
        h.attrs["raw_matlab_variable_data_raw"] = "Raw IQ voltage data indexed by range sample and pulse time."
    return True


def patch_index(path):
    if not os.path.exists(path):
        return False
    with h5py.File(path, "r+") as h:
        if str(h.attrs.get("times_ns_time_scale", "")).upper() == "UTC":
            return False
        for key in ["sanya_dt0_ns", "danzhou_dt0_ns", "wenchang_dt0_ns"]:
            if key in h:
                legacy = h[key][()].astype(np.int64)
                del h[key]
                h[key] = legacy - SOURCE_TIMEZONE_OFFSET_NS
                h[f"{key}_beijing_local"] = legacy
        h.attrs["times_ns_time_scale"] = "UTC"
        h.attrs["source_time_zone"] = f"Beijing local time (UTC+{SOURCE_TIMEZONE_OFFSET_HOURS})"
        h.attrs["source_time_correction"] = "dt0 datasets = legacy raw MATLAB local time - 8 hours"
        h.attrs["source_timezone_offset_hours"] = SOURCE_TIMEZONE_OFFSET_HOURS
    return True


def main():
    paths = sorted(glob.glob(TRISTATIC_PATTERN))
    changed = 0
    for path in paths:
        if patch_event_file(path):
            changed += 1
    index_changed = patch_index(TRISTATIC_INDEX)
    print(f"event files checked: {len(paths)}")
    print(f"event files patched: {changed}")
    print(f"index patched: {index_changed}")


if __name__ == "__main__":
    main()
