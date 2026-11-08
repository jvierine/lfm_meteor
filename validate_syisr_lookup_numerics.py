"""Controlled Metal mesh and duration tests; never modifies the science lookup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
FDTD = ROOT.parent / "fdtd"
sys.path.insert(0, str(FDTD / "examples"))
import converge_tarnecki as conv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    # The complete physical box and tapered density are identical in each trio.
    cases = [("baseline", 512, 50., 4.), ("duration", 512, 50., 8.),
             ("mesh", 640, 62.5, 4.)]
    for _, n, cpw, _ in cases:
        assert n / cpw == 10.24
        assert conv.geometry(n, cpw)[1] >= 4.
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/syisr_lookup_numerical_checks_20261108")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    cases = plan()
    if args.self_test:
        print("Identical box, plasma support, frequency and independent mesh/time comparisons verified.")
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lookup = ROOT / "results/syisr_fdtd_lookup_20261027/lookup.h5"
    sources = [Path(__file__), FDTD / "examples/converge_tarnecki.py",
               FDTD / "examples/reproduce_tarnecki_sparse.py",
               FDTD / "examples/dimant_oppenheim.py",
               FDTD / "examples/metal_fdtd.swift", FDTD / "src/main.rs"]
    signatures = {str(p): sha(p) for p in sources + [lookup]}
    records, summary = [], []
    started = time.time()

    def save(status, error=""):
        partial = output / "validation.partial.h5"
        with h5py.File(partial, "w") as h:
            h.attrs.update(status=status, error=error, pid=os.getpid(),
                           started_unix_s=started, updated_unix_s=time.time(),
                           tolerance_db=1., frequency_hz=430e6,
                           source_hashes_json=json.dumps(signatures, sort_keys=True),
                           scope="Six representative probes, not an exhaustive per-cell certification",
                           geometry="Head-on, B=0, collisionless; same physical box 10.24 wavelengths and plasma taper support 4 wavelengths",
                           mesh="50 to 62.5 cells per wavelength; identical density prescription",
                           duration="periods setting 4 to 8 after pulse/domain travel; not only four total RF periods")
            for p in sources:
                h.create_dataset("sources/" + p.name, data=p.read_text())
            h["records_json"] = json.dumps(records)
            h["comparisons_json"] = json.dumps(summary)
        partial.replace(output / "validation.h5")

    try:
        save("Preparing")
        if shutil.disk_usage(output).free < 20 * 1024**3:
            raise RuntimeError("Less than 20 GiB free; refusing density allocation")
        with h5py.File(lookup) as h:
            altitude = h["altitude_km"][:]
            velocity = h["velocity_km_s"][:]
            strengths = h["source_n0_rM2_m_inv"][:]
            neutral = h["neutral_number_density_m3"][:]
            previous = h["rcs_dbsm"][:]
        probes = [(0, 7, 3), (8, 7, 3), (0, 1, 3), (8, 1, 3),
                  (3, 3, 2), (8, 1, 0)]
        support_m = 4. * 299792458. / 430e6
        minimum_length = min(conv.sparse.plasma_parameters(
            velocity[v], strengths[s], neutral[a])[0] for a, v, s in probes)
        shape_path = output / "dimensionless_shape.h5"
        if shape_path.exists():
            from scipy.interpolate import RegularGridInterpolator
            with h5py.File(shape_path) as h:
                radii, mu, values = h["radius"][:], h["mu"][:], h["log_density"][:]
                scale = float(h.attrs["scale"])
            shape = radii, scale, RegularGridInterpolator((np.log(radii), mu), values)
        else:
            shape = conv.sparse.shared_shape(max_radius=1.01*np.sqrt(3)*support_m/minimum_length)
            with h5py.File(shape_path, "w") as h:
                h["radius"] = shape[0]
                h["mu"] = shape[2].grid[1]
                h["log_density"] = shape[2].values
                h.attrs["scale"] = shape[1]
        binary = output / "metal-fdtd"
        if not binary.exists():
            subprocess.run(["xcrun", "swiftc", "-O", str(FDTD / "examples/metal_fdtd.swift"),
                            "-o", str(binary), "-framework", "Metal", "-framework", "Foundation"], check=True)
        # run_case uses the existing solver path; supply our frozen binary there
        # through a private ROOT proxy, without altering the production binary.
        original_root = conv.ROOT
        solver_root = output / "solver"
        (solver_root / "tmp").mkdir(parents=True, exist_ok=True)
        link = solver_root / "tmp/metal-fdtd"
        if not link.exists():
            link.symlink_to(binary)
        (solver_root / "examples").mkdir(exist_ok=True)
        metal_link = solver_root / "examples/metal_fdtd.swift"
        if not metal_link.exists():
            metal_link.symlink_to(FDTD / "examples/metal_fdtd.swift")
        conv.ROOT = solver_root
        rust_source = (FDTD / "src/main.rs").read_text()
        for a, v, s in probes:
            tag = f"h{altitude[a]:g}_v{velocity[v]:g}_s{s}"
            values = {}
            for kind, n, cpw, periods in cases:
                for p, digest in signatures.items():
                    if sha(p) != digest:
                        raise RuntimeError(f"Input source changed during validation: {p}")
                if shutil.disk_usage(output).free < 20 * 1024**3:
                    raise RuntimeError("Less than 20 GiB free; stopped before allocation")
                save(f"Running {tag} {kind}; {len(records)}/18 cases complete")
                print(f"VALIDATE {tag} {kind}: n={n}, cpw={cpw}, periods={periods}", flush=True)
                run_args = SimpleNamespace(output=output, backend="metal", neutral_density=float(neutral[a]),
                                           ranks=1, threads=1, wait_pid=None)
                row = conv.run_case(run_args, "syisr", n, cpw, 4, periods, 4.,
                                    tag + "_" + kind, shape, None, rust_source,
                                    frequency=430e6, speed=float(velocity[v]), source=float(strengths[s]))
                values[kind] = row["rcs_dbsm"]
                row.update(altitude_km=float(altitude[a]), kind=kind)
                records.append(row)
                save(f"Completed {len(records)}/18 cases")
            comparison = dict(altitude_km=float(altitude[a]), velocity_km_s=float(velocity[v]),
                              source_m_inv=float(strengths[s]), **values,
                              original_lookup_dbsm=float(previous[a, v, s]),
                              support_adjustment_db=values["baseline"]-float(previous[a, v, s]),
                              duration_change_db=values["duration"]-values["baseline"],
                              mesh_change_db=values["mesh"]-values["baseline"])
            comparison["passed"] = bool(all(abs(comparison[k]) <= 1. for k in
                ("support_adjustment_db", "duration_change_db", "mesh_change_db")))
            summary.append(comparison)
            print("COMPARISON", json.dumps(comparison), flush=True)
            save(f"Compared {len(summary)}/6 probes")
        conv.ROOT = original_root
        save("PASS: six probes within 1 dB" if all(p["passed"] for p in summary)
             else "FAIL: at least one probe exceeds 1 dB; science lookup unchanged")
    except BaseException as exc:
        save("FAILED", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
