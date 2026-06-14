#!/usr/bin/env python3
"""Validate raw-voltage cut files by reproducing stored range gates.

The validation analytically regenerates the LFM transmit pulse from the scalar
waveform parameters stored in each cut file, matched-filters raw_voltage, and
checks that the maximum within the science range window recovers each detection
range gate.  By default it validates the events listed in the tri-static index,
because those are the events used for the trajectory solution in the paper.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cut-root", default="/mnt/data/juha/sanya/replication_data/raw_voltage_cuts")
    p.add_argument("--tristatic-index", default="results/tristatic_event_index.h5")
    p.add_argument("--output-csv", default="/mnt/data/juha/sanya/replication_data/raw_voltage_cut_validation.csv")
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--tolerance-gates", type=int, default=0)
    return p.parse_args()


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def tristatic_event_ids(path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with h5py.File(path, "r") as h:
        for site, key in (
            ("sanya", "sanya_event_id"),
            ("danzhou", "danzhou_event_id"),
            ("wenchang", "wenchang_event_id"),
        ):
            rows.extend((site, decode(value)) for value in h[key][:])
    return rows


def lfm_pulse(length_us: float = 199.0, sample_rate_mhz: float = 4.0, bandwidth_mhz: float = 4.0) -> np.ndarray:
    n = int(round(float(length_us) * float(sample_rate_mhz)))
    t = np.arange(n, dtype=np.float64) / (float(sample_rate_mhz) * 1e6)
    bandwidth_hz = float(bandwidth_mhz) * 1e6
    omega = bandwidth_hz * 1e6 / float(length_us) / 2.0
    phase = 2.0 * np.pi * (t * bandwidth_hz / 2.0 - omega * t**2)
    return np.exp(1j * phase).astype(np.complex64)


def validate_file(path: str, tolerance_gates: int) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as h:
        raw_voltage = h["raw_voltage"][:].astype(np.complex64)
        code = lfm_pulse(
            float(h["lfm_pulse_length_us"][()]),
            float(h["lfm_sample_rate_mhz"][()]),
            float(h["lfm_bandwidth_mhz"][()]),
        )
        global_gates = h["range_gate_index"][:].astype(np.int64)
        science_gates = h["science_range_gate_index"][:].astype(np.int64)
        sci0 = int(science_gates[0])
        sci1 = int(science_gates[-1]) + 1
        sci_local = np.flatnonzero((global_gates >= sci0) & (global_gates < sci1))
        det = h["detections"]
        pulse_index = det["pulse_index"][:].astype(np.int64)
        expected_gate = det["range_gate_index"][:].astype(np.int64)
        selected = det["selected_for_cut"][:].astype(bool)
        event_id = str(h.attrs["event_id"])
        site = str(h.attrs["site"])

        for i in np.flatnonzero(selected):
            pidx = int(pulse_index[i])
            if pidx < 0 or pidx >= raw_voltage.shape[1]:
                rows.append(
                    {
                        "event_id": event_id,
                        "site": site,
                        "detection_index": int(i),
                        "expected_gate": int(expected_gate[i]),
                        "reproduced_gate": -1,
                        "gate_error": 999999,
                        "ok": False,
                        "reason": "pulse_outside_cut",
                    }
                )
                continue
            matched = np.convolve(raw_voltage[:, pidx], np.conj(code), mode="same")
            local_peak = int(sci_local[int(np.argmax(np.abs(matched[sci_local])))] if len(sci_local) else np.argmax(np.abs(matched)))
            reproduced_gate = int(global_gates[local_peak])
            gate_error = reproduced_gate - int(expected_gate[i])
            rows.append(
                {
                    "event_id": event_id,
                    "site": site,
                    "detection_index": int(i),
                    "expected_gate": int(expected_gate[i]),
                    "reproduced_gate": reproduced_gate,
                    "gate_error": int(gate_error),
                    "ok": bool(abs(gate_error) <= tolerance_gates),
                    "reason": "ok" if abs(gate_error) <= tolerance_gates else "gate_mismatch",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    event_refs = tristatic_event_ids(args.tristatic_index)
    if args.max_events > 0:
        event_refs = event_refs[: args.max_events]

    rows = []
    missing = []
    for site, event_id in event_refs:
        path = os.path.join(args.cut_root, site, f"{event_id}.h5")
        if not os.path.exists(path):
            missing.append((site, event_id))
            continue
        rows.extend(validate_file(path, args.tolerance_gates))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["event_id", "site", "detection_index", "expected_gate", "reproduced_gate", "gate_error", "ok", "reason"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for row in rows if row["ok"])
    n_bad = len(rows) - n_ok
    print(f"validated detections: {len(rows)}")
    print(f"ok: {n_ok}")
    print(f"bad: {n_bad}")
    print(f"missing tri-static event files: {len(missing)}")
    if missing[:10]:
        print("first missing:", missing[:10])
    print(args.output_csv)
    if n_bad or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
