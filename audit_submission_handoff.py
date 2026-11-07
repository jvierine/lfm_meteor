"""Inventory generated upload files and audit cut/fit joins without changing data."""
import argparse
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import h5py

from package_submission import sha, write_rows


def audit(root, output, code_component="code-release", readme="README.txt"):
    inventory = []
    orbit_component = "orbits-refreshed" if (root / "orbits-refreshed").exists() else "orbits-current"
    for component in ("cuts-core", "catalog-core", orbit_component, "uncertainty-current", "l1-complete-archive", "yield-current", "mass-derived-final", "mass-evidence-final", "kinematics-final", code_component):
        folder = root / component
        if not folder.exists():
            continue
        with h5py.File(folder / "checksums.h5") as h:
            names = ["manifest.h5"] + list(h["archives/path"].asstr()[:])
            for name in names:
                path = folder / name
                digest = sha(path)
                if digest != h.attrs[name]:
                    raise ValueError(f"Transfer mismatch: {path}")
                inventory.append(dict(path=f"{component}/{name}", bytes=path.stat().st_size,
                                      sha256=digest, status="superseded_baseline_not_for_upload" if name == "code.tar.gz" else "verified"))
        path = folder / "checksums.h5"
        inventory.append(dict(path=f"{component}/checksums.h5", bytes=path.stat().st_size,
                              sha256=sha(path), status="checksum_record"))
    for name in (readme, "orbit-audit-request.h5", "orbit-remote-provenance.h5", "current-selected-uncertainty-audit.h5"):
        path = root / name
        if not path.exists():
            continue
        inventory.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path), status="metadata"))
    with h5py.File(root / "cuts-core/manifest.h5") as h:
        cut_paths = [s for s in h["files/path"].asstr()[:] if s.startswith("L0/") and s.count("/") == 2]
        cuts = {Path(s).name for s in cut_paths}
    with h5py.File(root / "catalog-core/manifest.h5") as h:
        hashes = dict(zip(h["files/path"].asstr()[:], h["files/sha256"].asstr()[:]))
    rows = {}
    with tarfile.open(root / "catalog-core/fits.tar.gz", "r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if len(parts) != 4 or parts[:2] != ("L1", "events"):
                continue
            event, name = parts[2:]
            if name not in ("input.h5", "physical_states.h5", "beam_rcs.h5", "rcs_mass.h5", "plot.h5"):
                continue
            row = rows.setdefault(event, dict(event=event, cuts_required=0, cuts_present=0,
                missing_cuts="", state_velocity_vector=False, r0=False, m0=False,
                beam_fit_hash="absent", rcs_fit_hash="absent", rcs_beam_hash="absent"))
            with h5py.File(io.BytesIO(tar.extractfile(member).read())) as h:
                if name == "input.h5":
                    required = [Path(s).name for s in str(h.attrs.get("source_event_files", "")).split(";") if s]
                    missing = [s for s in required if s not in cuts]
                    row.update(cuts_required=len(required), cuts_present=len(required)-len(missing), missing_cuts=";".join(missing))
                elif name == "physical_states.h5":
                    row["state_velocity_vector"] = "velocity_ecef_m_s" in h
                elif name == "plot.h5":
                    row["r0"] = "initial_radius_m_quantiles" in h
                    row["m0"] = "initial_mass_kg_quantiles" in h
                else:
                    prefix = f"L1/events/{event}/"
                    key = "beam_fit_hash" if name == "beam_rcs.h5" else "rcs_fit_hash"
                    row[key] = "match" if h.attrs.get("fit_sha256") == hashes.get(prefix + "fit.h5") else "mismatch"
                    if name == "rcs_mass.h5":
                        row["rcs_beam_hash"] = "match" if h.attrs.get("beam_sha256") == hashes.get(prefix + "beam_rcs.h5") else "mismatch"
    supplement = root / "l1-complete/inferred_states.h5"
    if supplement.exists():
        with h5py.File(root / "l1-complete/checksums.h5") as h:
            assert sha(supplement) == h.attrs["inferred_states.h5"]
        with h5py.File(supplement) as h:
            for event, row in rows.items():
                row["core_state_velocity_vector"] = row["state_velocity_vector"]
                row["state_velocity_vector"] = "velocity_ecef_m_s" in h[event]
                row["vector_source"] = "l1-complete-archive supplement"
                row["independent_doppler_available"] = False
    summary = dict(cut_files=len(cut_paths), cut_sites=dict(Counter(Path(p).parts[1] for p in cut_paths)),
                   fitted_events=len(rows), required_cut_associations=sum(r["cuts_required"] for r in rows.values()),
                   present_cut_associations=sum(r["cuts_present"] for r in rows.values()),
                   missing_state_velocity_vectors=sum(not r["state_velocity_vector"] for r in rows.values()),
                   missing_r0_intervals=sum(not r["r0"] for r in rows.values()),
                   missing_m0_intervals=sum(not r["m0"] for r in rows.values()))
    for key in ("beam_fit_hash", "rcs_fit_hash", "rcs_beam_hash"):
        summary[key] = dict(Counter(r[key] for r in rows.values()))
    with h5py.File(output, "x") as h:
        h.attrs.update(status="FROZEN_LOCAL_RELEASE", summary_json=json.dumps(summary),
                       reserved_doi="10.5281/zenodo.24037714", publication="unpublished",
                       scope="Cut associations and recorded fit/beam provenance only; no inference validation",
                       code_choice=f"{code_component}/code-final.tar.gz; earlier code archives are superseded",
                       upload_status="Not verified here; parent reports remaining uploads pending Mac unlock")
        write_rows(h, "files", inventory)
        write_rows(h, "event_coverage", list(rows.values()))
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--code-component", default="code-release")
    parser.add_argument("--readme", default="README.txt")
    args = parser.parse_args()
    audit(args.root, args.output, args.code_component, args.readme)
