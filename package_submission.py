"""Stage an UNFROZEN, local-only Sanya submission component; never upload.

Only explicitly selected event products and committed source blobs are read.
Provenance, coverage gaps and checksums are stored in HDF5 metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import h5py
import numpy as np

PRODUCTS = (
    "input.h5", "fit.h5", "physical_states.h5", "plot.h5", "beam_rcs.h5",
    "rcs_mass.h5", "dynamic_mass_posterior.h5", "excluded_measurements.h5",
    "path_measurement_uncertainty.h5", "complete.h5",
    "cache_validation.h5", "posterior_cache_validation.h5",
    "reduced_raw_validation.h5", "reduced_statistics_validation.h5",
)
REQUIRED = PRODUCTS[:7]
STR = h5py.string_dtype("utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(p):
    s = p.stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def safe_file(p, root):
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Not a regular contained input: {p}")


def check_hdf5(path):
    # Reject references that would silently depend on files outside the archive.
    with h5py.File(path) as h:
        visited = set()

        def walk(g):
            address = h5py.h5o.get_info(g.id).addr
            if address in visited:
                return
            visited.add(address)
            for key in g:
                link = g.get(key, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValueError(f"Nonlocal HDF5 link: {path}:{g.name}/{key}")
                obj = g[key]
                if isinstance(obj, h5py.Group):
                    walk(obj)
                elif obj.is_virtual or obj.external:
                    raise ValueError(f"External HDF5 storage: {path}:{obj.name}")
        walk(h)


def validate_cut(p):
    with h5py.File(p) as h:
        if h.attrs.get("format_name") != "sanya_head_echo_raw_voltage_cut":
            raise ValueError(f"Not a memo16 detection cut: {p}")
        for key in ("raw_voltage", "unix_sample", "range_gate_index",
                    "science_range_gate_index", "detections/selected_for_cut",
                    "lfm_pulse_length_us", "lfm_sample_rate_mhz", "lfm_bandwidth_mhz"):
            if key not in h:
                raise ValueError(f"Missing cut field {key}: {p}")
        v, t, g = h["raw_voltage"], h["unix_sample"][:], h["range_gate_index"][:]
        if v.shape != (len(g), len(t)) or v.dtype != np.dtype("complex64"):
            raise ValueError(f"Bad voltage layout: {p}")
        if not len(t) or not np.all(np.diff(t.astype(np.int64)) > 0):
            raise ValueError(f"Bad cut time axis: {p}")
        selected = h["detections/selected_for_cut"][:].astype(bool)
        if not selected.any() or (int(t[-1]) - int(t[0])) / 1e9 > 120:
            raise ValueError(f"Not a bounded detection cut: {p}")
        indices = h["detections/pulse_index"][:][selected]
        if np.any(indices < 0) or np.any(indices >= len(t)):
            raise ValueError(f"Detection outside cut: {p}")


def select(args):
    root = args.root.resolve()
    files, gaps = {}, []

    def add(path, dest):
        safe_file(path, root)
        if dest in files and files[dest] != path:
            raise ValueError(f"Duplicate destination: {dest}")
        files[dest] = path

    if args.component == "code":
        return {}, ["Code license: no project-wide declaration found; owner decision pending"]
    if args.component == "yield":
        path = root / "results/submission_yield_20261107.h5"
        with h5py.File(path) as h:
            if h.attrs["selection_sha256"] != sha(root / "results/tristatic_shrinking_orbits_current_20261107/orbits.h5"):
                raise ValueError("Yield product references a superseded orbit selection")
        add(path, "yield/submission_yield_20261107.h5")
        return files, ["Planning extrapolation, not seasonal yield; original limitation metadata retained"]
    if args.component == "l1":
        folder = root / args.l1_dir
        for name in ("detection_rates.h5", "inferred_states.h5", "checksums.h5"):
            add(folder / name, "L1_velocity/" + name)
        return files, ["Independent Doppler velocity explicitly unavailable; measurement-derived apparent track rates are provided",
                       "Per-detection velocity vectors are inferred best-fit states, not independently measured vectors",
                       "Full vector covariance is not supplied; selected ensemble sources and published scalar intervals are retained"]
    if args.component == "uncertainty":
        with h5py.File(args.provenance_audit) as h:
            g = h["audit"]
            for i in range(len(g["path"])):
                if g["role"].asstr()[i] != "uncertainty":
                    continue
                path = Path(g["path"].asstr()[i])
                if g["status"].asstr()[i] != "match" or sha(path) != g["expected_sha256"].asstr()[i]:
                    raise ValueError(f"Unverified uncertainty source: {path}")
                event = g["event"].asstr()[i]
                add(path, f"uncertainty/{event}/{path.name}")
        return files, ["Full saved uncertainty sources; conditional and robustness ensembles have distinct interpretations"]
    if args.component == "orbits":
        orbit = root / args.orbit_product
        for name in ("orbits.h5", "verification.h5", "delivery_validation.h5", "refresh_manifest.h5",
                     "remote_source_audit.h5", "AUDIT.txt", "classification.tex"):
            add(orbit / name, "orbits/" + name)
        with h5py.File(orbit / "orbits.h5") as h:
            for event, g in h["source_audit"].items():
                for key, label in (("snapshot", "states"), ("propagated_source", "propagated")):
                    path = Path(g.attrs[key])
                    expected = g.attrs["snapshot_sha256" if key == "snapshot" else "propagated_sha256"]
                    if sha(path) != expected:
                        raise ValueError(f"Orbit source hash changed: {path}")
                    add(path, f"orbits/{label}/{event}.h5")
        gaps += ["Orbit audit: all 232 remote inputs/fits/current plots and 649 selected source hashes verified",
                 "Orbit audit: 40 stale local plot selections did not imply superseded saved orbit ensembles; see corrected AUDIT.txt",
                 "Orbit audit: 232 ensembles, 229 plotted; tri0035 rejected, tri0145 replaced",
                 "Orbit audit: all saved samples bound, not proof all physical trajectories are bound"]
        return files, gaps
    if args.component == "cuts":
        for site in ("sanya", "danzhou", "wenchang"):
            paths = sorted((root / "raw_voltage_cuts" / site).glob("*.h5"))
            if not paths:
                gaps.append(f"L0: no cuts for {site}")
            for p in paths:
                validate_cut(p)
                add(p, f"L0/{site}/{p.name}")
        p = root / "raw_voltage_cut_index.h5"
        if p.exists():
            add(p, "L0/raw_voltage_cut_index.h5")
        else:
            gaps.append("L0: cut index absent")
        gaps += ["L0: historical matched-filter validation not rerun by packaging",
                 "L1: memo16 detections have range/time/SNR, no measured Doppler velocity",
                 "L0/L1: join to final selected catalogue awaits cross-component audit"]
    else:
        catalog = root / args.catalog
        folders = sorted(p for p in catalog.glob("tri_*") if p.is_dir())
        if not folders:
            raise ValueError("No catalogue events")
        for folder in folders:
            event = folder.name
            for name in PRODUCTS:
                p = folder / name
                if p.exists():
                    add(p, f"L1/events/{event}/{name}")
                elif name in REQUIRED:
                    gaps.append(f"{event}: missing {name}")
            if (folder / "physical_states.h5").exists():
                with h5py.File(folder / "physical_states.h5") as h:
                    if "velocity_ecef_m_s" not in h:
                        gaps.append(f"{event}: state velocity vector absent (scalar speed retained)")
            if (folder / "input.h5").exists():
                with h5py.File(folder / "input.h5") as h:
                    original = Path(str(h.attrs.get("original_input", "")))
                    if original.is_file():
                        add(original, f"support/original_inputs/{event}.h5")
                    else:
                        gaps.append(f"{event}: original unexcluded input absent")
            if (folder / "rcs_mass.h5").exists():
                with h5py.File(folder / "rcs_mass.h5") as h:
                    lookup = Path(str(h.attrs.get("lookup_file", "")))
                    if lookup.is_file():
                        add(lookup, "lookup/" + lookup.parent.name + "/" + lookup.name)
                    else:
                        gaps.append(f"{event}: standalone RCS lookup absent")
        for p in sorted((catalog / "flags_snapshot").glob("*.h5")):
            add(p, f"exclusions/{p.name}")
        if not (catalog / "flags_snapshot").is_dir():
            gaps.append("Catalogue: review flags snapshot absent")
        gaps += ["Freeze: mass, orbit, selection and uncertainty owners must approve exact hashes",
                 "Reproduction: full trajectory chains and likelihood caches omitted from compact core",
                 "Reproduction: committed source is a baseline, not proof of the historical producer revision",
                 "Reproduction: original_input/source_event_files paths need archive path resolution",
                 "Reproduction: Rustplotlib external path dependency not vendored",
                 "L1: detected Doppler velocity and full state covariance require owner export"]
    if not files:
        raise ValueError("Empty selection")
    return files, sorted(set(gaps))


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def source_snapshot(repo, out):
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    names = git(repo, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    selected = []
    for name in names:
        p = Path(name)
        allowed = ((len(p.parts) == 1 and p.suffix == ".py") or
                   (p.parts[0] in ("tristatic_native", "rustplotlib_figures") and
                    p.suffix in (".rs", ".toml", ".lock")) or
                   name == "SUBMISSION_PACKAGING.md")
        if not allowed:
            continue
        blob = git(repo, "show", f"{commit}:{name}")
        if re.search(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}", blob):
            raise ValueError(f"Possible credential in source: {name}")
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        selected.append(dest)
    return commit, selected


def write_rows(h, name, rows):
    group = h.create_group(name)
    if rows:
        for key in rows[0]:
            values = [row[key] for row in rows]
            group.create_dataset(key, data=values, dtype=STR if isinstance(values[0], str) else None)


def orbit_request(root, output):
    rows = []
    with h5py.File(root / "results/tristatic_shrinking_orbits_audited_20261107/orbits.h5") as h:
        for event, g in h["source_audit"].items():
            with h5py.File(g.attrs["snapshot"]) as s:
                base = "/home/juha/src/lfm_meteor/results/tristatic_cached_shrinking_20261029/" + event
                for role, path, digest in [("fit", base + "/fit.h5", s.attrs["fit_sha256"]),
                                           ("input", base + "/input.h5", s.attrs["input_sha256"])]:
                    rows.append(dict(event=event, role=role, path=path, expected_sha256=digest))
                for path, digest in zip(json.loads(s.attrs["source_files"]), json.loads(s.attrs["source_sha256"])):
                    rows.append(dict(event=event, role="uncertainty", path=path, expected_sha256=digest))
    with h5py.File(output, "x") as h:
        write_rows(h, "requests", rows)
        h.attrs["orbit_product_sha256"] = sha(root / "results/tristatic_shrinking_orbits_audited_20261107/orbits.h5")


def remote_audit(root, request, output):
    rows = []
    with h5py.File(request) as h:
        g = h["requests"]
        for i in range(len(g["path"])):
            row = {k:g[k].asstr()[i] for k in g}
            path = Path(row["path"])
            if path.suffix != ".h5" or not path.resolve().is_relative_to((root / "results").resolve()):
                raise ValueError(f"Audit path outside scientific results: {path}")
            row.update(actual_sha256="", bytes=0, status="missing")
            if path.exists():
                safe_file(path, root)
                before = signature(path)
                row.update(actual_sha256=sha(path), bytes=path.stat().st_size)
                row["status"] = "match" if row["actual_sha256"] == row["expected_sha256"] else "mismatch"
                if before != signature(path):
                    row["status"] = "changed_during_read"
            rows.append(row)
    with h5py.File(output, "x") as h:
        write_rows(h, "audit", rows)
        h.attrs.update(request_sha256=sha(request), host=platform.node(), root=str(root),
                       meaning="Current remote file identity against saved propagation provenance; no refit")
    from collections import Counter
    print(json.dumps(dict(Counter(row["role"] + ":" + row["status"] for row in rows))))


def verify(out, archives_only=False):
    with h5py.File(out / "manifest.h5") as h:
        paths = h["files/path"].asstr()[:]
        digests = h["files/sha256"].asstr()[:]
        for name, digest in ([] if archives_only else zip(paths, digests)):
            path = out / name
            safe_file(path, out)
            if sha(path) != digest:
                raise ValueError(f"Checksum mismatch: {path}")
    with h5py.File(out / "checksums.h5") as h:
        archives = list(h["archives/path"].asstr()[:])
        for name in ["manifest.h5"] + archives:
            if sha(out / name) != h.attrs[name]:
                raise ValueError(f"Component mismatch: {name}")
    expected = dict(zip(paths, digests))
    seen = set()
    for archive in archives:
        with tarfile.open(out / archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                digest = hashlib.sha256()
                with tar.extractfile(member) as f:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected[member.name]:
                    raise ValueError(f"Archive checksum mismatch: {member.name}")
                seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Incomplete tar archives")
    print(json.dumps(dict(verified=str(out), files=len(paths), archives={n:(out/n).stat().st_size for n in archives})))


def build(args):
    files, gaps = select(args)
    initial = {name: signature(p) for name, p in files.items()}
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    out.mkdir(parents=True, mode=0o700)
    rows = []
    for name, src in sorted(files.items()):
        if src.suffix == ".h5":
            check_hdf5(src)
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        digest = sha(dst)
        if signature(src) != initial[name] or sha(src) != digest:
            raise RuntimeError(f"Source changed during snapshot: {src}")
        rows.append(dict(path=name, source_path=str(src), bytes=dst.stat().st_size,
                         sha256=digest, source_mtime_ns=initial[name][1]))
    if getattr(args, "data_only", False):
        commit, source_files = git(args.repo, "rev-parse", "HEAD").decode().strip(), []
    else:
        commit, source_files = source_snapshot(args.repo, out)
    for p in source_files:
        rows.append(dict(path=p.relative_to(out).as_posix(), source_path="git:" + commit,
                         bytes=p.stat().st_size, sha256=sha(p), source_mtime_ns=0))
    # Copy only the detections group: never duplicate raw recording arrays into L1.
    if args.component == "cuts":
        p = out / "L1" / "detections.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as h:
            h.attrs["velocity_status"] = "Not supplied by original cut exporter; no synthetic velocity substituted"
            for name, src in sorted(files.items()):
                if not name.startswith(("L0/sanya/", "L0/danzhou/", "L0/wenchang/")):
                    continue
                with h5py.File(out / name) as cut:
                    g = h.create_group(Path(name).stem)
                    cut.copy("detections", g)
                    g.attrs["L0_path"] = name
                    for k in ("site", "event_id", "selection_reason", "force_tristatic_include"):
                        g.attrs[k] = cut.attrs[k]
        rows.append(dict(path="L1/detections.h5", source_path="derived:copied L0 detections groups",
                         bytes=p.stat().st_size, sha256=sha(p), source_mtime_ns=0))
    current, _ = select(args)
    if current != files or any(signature(p) != initial[n] for n, p in files.items()):
        raise RuntimeError("Selection changed; discard this incomplete snapshot and retry")
    rows.sort(key=lambda r: r["path"])
    with h5py.File(out / "manifest.h5", "w") as h:
        h.attrs.update(schema="sanya-submission-component-v1", status="UNFROZEN_INITIAL_CORE",
                       component=args.component, host=platform.node(), source_commit=commit,
                       source_root=str(args.root.resolve()), python=platform.python_version(),
                       packaging_program_sha256=sha(Path(__file__)),
                       h5py=h5py.__version__, numpy=np.__version__,
                       source_worktree_status=git(args.repo, "status", "--short", "--untracked-files=no").decode(),
                       upload_policy="No upload or publication performed or implemented",
                       reserved_doi="10.5281/zenodo.24037714",
                       doi_status="User-confirmed reserved DOI, unpublished reviewer draft",
                       scientific_status="Conditional results; retain original limitations and exclusions")
        write_rows(h, "files", rows)
        h.create_dataset("missing_or_pending", data=gaps, dtype=STR)
    # Stable ordering, ownership and timestamps; source bytes are unchanged.
    import gzip
    partitions = {}
    for row in rows:
        top = row["path"].split("/")[0]
        key = {"yield":"yield", "L1_velocity":"l1-velocity", "uncertainty":"uncertainty", "orbits":"orbits", "L0":"raw-cuts", "L1":"detections" if args.component == "cuts" else "fits",
               "support":"fits", "exclusions":"fits", "lookup":"lookup", "source":"code"}[top]
        partitions.setdefault(key + ".tar.gz", []).append(row["path"])
    for archive, names in sorted(partitions.items()):
        with open(out / archive, "xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=4) as gz:
                with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for name in sorted(names):
                        p = out / name
                        info = tarfile.TarInfo(name)
                        info.size = p.stat().st_size
                        info.mode = 0o644
                        with p.open("rb") as f:
                            tar.addfile(info, f)
    with h5py.File(out / "checksums.h5", "w") as h:
        write_rows(h, "archives", [{"path":n} for n in sorted(partitions)])
        for name in ["manifest.h5"] + sorted(partitions):
            h.attrs[name] = sha(out / name)
            h.attrs[name + "_bytes"] = (out / name).stat().st_size
    verify(out)
    print(json.dumps(dict(source_bytes=sum(r["bytes"] for r in rows), pending=len(gaps), source_commit=commit)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--component", choices=("cuts", "catalog", "orbits", "code", "uncertainty", "l1", "yield"))
    p.add_argument("--root", type=Path)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--catalog", default="results/tristatic_cached_shrinking_20261029")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--archives-only", action="store_true")
    p.add_argument("--make-orbit-request", action="store_true")
    p.add_argument("--audit-request", type=Path)
    p.add_argument("--provenance-audit", type=Path)
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--orbit-product", default="results/tristatic_shrinking_orbits_current_20261107")
    p.add_argument("--l1-dir", default="results/submission_staging/20261107/l1-velocity-supplement")
    args = p.parse_args()
    if args.make_orbit_request:
        orbit_request(args.root, args.output)
    elif args.audit_request:
        remote_audit(args.root, args.audit_request, args.output)
    elif args.verify:
        verify(args.output.resolve(), args.archives_only)
    elif args.root is None or args.component is None:
        p.error("--root and --component required for packaging")
    else:
        build(args)


if __name__ == "__main__":
    main()
