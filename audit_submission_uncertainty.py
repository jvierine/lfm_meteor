"""Check archive membership against CURRENT plot-selected uncertainty sources."""
import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from package_submission import sha, signature, write_rows


def audit(root, staging, output):
    with h5py.File(staging / "catalog-core/manifest.h5") as h:
        core = dict(zip(h["files/path"].asstr()[:], h["files/sha256"].asstr()[:]))
    with h5py.File(staging / "uncertainty-current/manifest.h5") as h:
        archived = dict(zip(h["files/path"].asstr()[:], h["files/sha256"].asstr()[:]))
    with h5py.File(staging / "orbit-audit-request.h5") as h:
        accepted = set(h["requests/event"].asstr()[:])
    rows, sources = [], []
    for folder in sorted(root.glob("tri_*")):
        if not folder.is_dir():
            continue
        event = folder.name
        watched = {folder / n:signature(folder / n) for n in ("plot.h5", "dynamic_mass_posterior.h5")}
        with h5py.File(folder / "plot.h5") as p, h5py.File(folder / "dynamic_mass_posterior.h5") as m:
            method = str(p.attrs.get("uncertainty_method", "conditional physical-model posterior"))
            mass_method = str(m.attrs.get("uncertainty_method", "conditional physical-model posterior"))
            assert method == mass_method, (event, "plot/mass method mismatch")
            if "random-deletion" in method:
                paths = [folder / "random_deletion.h5"]
                with h5py.File(paths[0]) as h:
                    draws = h["params"][:][np.isclose(h["drop_fraction"][:], .35)]
            elif "bootstrap" in method:
                paths = [folder / "block_bootstrap_states.h5"]
                with h5py.File(paths[0]) as h:
                    draws = h["params"][:]
            else:
                paths = [folder / f"chain_{i}.h5" for i in range(4)]
                parts = []
                for path in paths:
                    with h5py.File(path) as h:
                        parts.append(h["samples"][:])
                n = min(map(len, parts))
                draws = np.asarray([part[:n] for part in parts]).reshape(-1, 7)
            radius = 10**draws[:, 6]
            mass = 4*np.pi/3*3000*radius**3
            for name, data, mass_field in (("initial_radius_m_quantiles", radius, "radius_m"),
                                          ("initial_mass_kg_quantiles", mass, "mass_kg")):
                np.testing.assert_allclose(np.quantile(data, [.025, .5, .975]), p[name][:], rtol=.01, atol=0)
                np.testing.assert_allclose(np.quantile(m[mass_field][:], [.025, .5, .975]), p[name][:], rtol=1e-10, atol=0)
            row = dict(event=event, method=method, accepted=event in accepted,
                       selected_draw_count=len(draws), selected_source_count=len(paths))
            for name in ("plot.h5", "dynamic_mass_posterior.h5"):
                digest = sha(folder / name)
                assert digest == core[f"L1/events/{event}/{name}"], (event, name, "stale core")
                row[name + "_sha256"] = digest
            rows.append(row)
            for path in paths:
                before = signature(path)
                digest = sha(path)
                assert before == signature(path), path
                status = "not_in_accepted_orbit_sample"
                if event in accepted:
                    assert digest == archived[f"uncertainty/{event}/{path.name}"], (event, path, "wrong selected source")
                    status = "match"
                sources.append(dict(event=event, path=str(path), sha256=digest, bytes=path.stat().st_size, status=status))
        assert all(signature(path) == sig for path, sig in watched.items()), event
    with h5py.File(output, "x") as h:
        h.attrs.update(status="PASS", selection_authority=str(root),
                       definition="CURRENT plot.h5 method, selected draws, plot/mass quantiles and archived source hashes",
                       quantile_rtol=.01, publication="unpublished; orbit propagation refresh pending")
        write_rows(h, "events", rows)
        write_rows(h, "selected_sources", sources)
    print("PASS", len(rows), "fits;", len(accepted), "accepted events;", Counter(r["method"] for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("staging", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.root, a.staging, a.output)
