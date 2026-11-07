"""Packaging safety tests; no scientific exporters are executed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import package_submission as p


class PackagingTests(unittest.TestCase):
    def test_containment_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = root / "real.h5"
            f.touch()
            (root / "link.h5").symlink_to(f)
            p.safe_file(f, root)
            with self.assertRaises(ValueError):
                p.safe_file(root / "link.h5", root)
            with self.assertRaises(ValueError):
                p.safe_file(f, root / "other")

    def test_external_hdf5_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "bad.h5"
            with h5py.File(f, "w") as h:
                h["external"] = h5py.ExternalLink("secret.h5", "/")
            with self.assertRaises(ValueError):
                p.check_hdf5(f)

    def test_build_verify_and_corruption(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = root / "raw_voltage_cuts/sanya/event.h5"
            f.parent.mkdir(parents=True)
            with h5py.File(f, "w") as h:
                h.attrs.update(format_name="sanya_head_echo_raw_voltage_cut", site="sanya",
                               event_id="event", selection_reason="test", force_tristatic_include=True)
                h["raw_voltage"] = np.zeros((5, 2), dtype="complex64")
                h["unix_sample"] = np.array([100, 100000100], dtype="uint64")
                h["range_gate_index"] = np.arange(5)
                h["science_range_gate_index"] = [2]
                h["detections/selected_for_cut"] = [True]
                h["detections/pulse_index"] = [0]
                for k in ("lfm_pulse_length_us", "lfm_sample_rate_mhz", "lfm_bandwidth_mhz"):
                    h[k] = 4.
            args = argparse.Namespace(root=root, output=root / "out", repo=root, component="cuts")
            with patch.object(p, "source_snapshot", return_value=("test", [])), patch.object(p, "git", return_value=b""):
                p.build(args)
            p.verify(args.output)
            with self.assertRaises(FileExistsError):
                p.build(args)
            with (args.output / "L0/sanya/event.h5").open("ab") as out:
                out.write(b"corruption")
            with self.assertRaises(ValueError):
                p.verify(args.output)


if __name__ == "__main__":
    unittest.main()
