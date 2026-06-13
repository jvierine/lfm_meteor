import argparse
import os

import h5py
import numpy as np

import plot_example_ballistic_fit_2x2 as example


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260611c.h5"
OUTPUT_DIR = "results/all_ballistic_fit_2x2_v20260611c"


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def plot_event_to_dir(h, idx, event_id, output_dir):
    old_output_dir = example.OUTPUT_DIR
    try:
        example.OUTPUT_DIR = output_dir
        return example.plot_event(h, idx, event_id)
    finally:
        example.OUTPUT_DIR = old_output_dir


def main():
    parser = argparse.ArgumentParser(description="Create 2x2 weighted-ballistic fit diagnostics for all events.")
    parser.add_argument("--input-h5", default=INPUT_H5)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--png-only", action="store_true", help="Remove PDFs after writing them to save space.")
    parser.add_argument("--limit", type=int, default=None, help="Only plot the first N events, for smoke testing.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    example.INPUT_H5 = args.input_h5

    with h5py.File(args.input_h5, "r") as h:
        event_ids = decode_strings(h["event_id"][:])
        if args.limit is not None:
            event_ids = event_ids[: args.limit]
        for idx, event_id in enumerate(event_ids):
            png, pdf = plot_event_to_dir(h, idx, event_id, args.output_dir)
            if args.png_only and os.path.exists(pdf):
                os.remove(pdf)
            print(f"{idx + 1:04d}/{len(event_ids):04d} {event_id} -> {png}", flush=True)

    print(f"wrote {len(event_ids)} 2x2 fit plots to {args.output_dir}")


if __name__ == "__main__":
    main()
