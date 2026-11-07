"""Freeze committed code plus explicitly inventoried dependency/working overlays.

No Git configuration, credentials, home directory, build trees or generated
scientific data are copied. Local imports are followed even when untracked.
"""
import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import h5py

from package_submission import git, safe_file, sha, signature, write_rows

EXTRA = (
    "refresh_current_orbit_ensembles.py", "install_current_orbit_products.py",
    "render_native_voltage_event.py", "run_syisr_fdtd_lookup.py",
    "freeze_submission_code.py",
)
SECRET = re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|https?://[^\s/]+:[^\s/]+@|zenodo[^\s]*[?&](?:token|access_token)=[A-Za-z0-9_-]{16,}")


def tree(repo):
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    entries = {}
    for line in git(repo, "ls-tree", "-r", commit).decode().splitlines():
        meta, path = line.split("\t", 1)
        entries[path] = meta.split()[2]
    return commit, entries


def freeze(repo, output, expected_commit):
    if output.exists():
        raise FileExistsError(output)
    commit, tracked = tree(repo)
    assert commit == git(repo, "rev-parse", expected_commit).decode().strip(), "Producer commit changed; confirm the new freeze explicitly"
    chosen = {}
    for name in tracked:
        p = Path(name)
        if ((len(p.parts) == 1 and p.suffix == ".py") or
            (p.parts[0] in ("tristatic_native", "rustplotlib_figures") and p.suffix in (".rs", ".toml", ".lock")) or
            name == "SUBMISSION_PACKAGING.md" or p.name in ("LICENSE", "COPYING")):
            chosen[repo / name] = (repo, "lfm_meteor", commit, tracked)
    for name in EXTRA:
        chosen[repo / name] = (repo, "lfm_meteor", commit, tracked)
    # Follow direct local Python imports, including ignored/untracked helpers.
    queue = list(chosen)
    modules = set()
    while queue:
        path = queue.pop()
        if path.suffix != ".py":
            continue
        syntax = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(syntax):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                modules.add(name.split(".")[0])
                for candidate in (repo / (name.replace(".", "/") + ".py"), repo / name.replace(".", "/") / "__init__.py"):
                    if candidate.is_file() and candidate not in chosen:
                        chosen[candidate] = (repo, "lfm_meteor", commit, tracked)
                        queue.append(candidate)
    # Preserve sibling layout required by the Cargo path dependency and FDTD driver.
    for label in ("rustplotlib", "fdtd"):
        dep = repo.parent / label
        rev, files = tree(dep)
        for name in files:
            p = Path(name)
            if p.suffix in (".rs", ".toml", ".lock", ".py", ".swift", ".metal") or p.name in ("LICENSE", "COPYING"):
                if p.parts[0] not in ("target", "tmp", "results", ".git"):
                    chosen[dep / name] = (dep, label, rev, files)
        if label == "fdtd":
            for path in (dep / "examples").glob("*.py"):
                chosen[path] = (dep, label, rev, files)
    for name in ("converge_tarnecki.py", "reproduce_tarnecki_sparse.py", "metal_fdtd.swift"):
        assert repo.parent / "fdtd/examples" / name in chosen
    output.mkdir(parents=True, mode=0o700)
    rows = []
    for path, (base, label, rev, tracked_files) in sorted(chosen.items()):
        safe_file(path, base)
        before = signature(path)
        data = path.read_bytes()
        assert signature(path) == before, path
        if SECRET.search(data):
            raise ValueError(f"Possible credential in selected source: {path}")
        if path.suffix == ".py":
            ast.parse(data, filename=str(path))
        rel = path.relative_to(base).as_posix()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        status = "committed" if tracked_files.get(rel) == blob else ("working_overlay" if rel in tracked_files else "untracked_dependency")
        name = f"source/{label}/{rel}"
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(dict(path=name, source_path=str(path), bytes=len(data), sha256=sha(target),
                         repository_commit=rev, source_status=status, git_blob_at_commit=tracked_files.get(rel, "")))
    # Verify all resolvable local imports have been carried into the snapshot.
    for path in (output / "source/lfm_meteor").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                relative = Path(name.replace(".", "/") + ".py")
                if (repo / relative).exists():
                    assert (output / "source/lfm_meteor" / relative).exists(), relative
    environment = output / "source/environment.h5"
    with h5py.File(environment, "x") as h:
        h.attrs.update(python=sys.version, platform=sys.platform,
                       note="Observed local base environment; install external packages separately; no environment variables or credentials captured")
        write_rows(h, "distributions", sorted([dict(name=d.metadata["Name"], version=d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]], key=lambda r:r["name"].lower()))
        h.create_dataset("import_roots", data=sorted(modules), dtype=h5py.string_dtype())
    rows.append(dict(path="source/environment.h5", source_path="generated:package metadata only", bytes=environment.stat().st_size,
                     sha256=sha(environment), repository_commit="", source_status="environment", git_blob_at_commit=""))
    with h5py.File(output / "manifest.h5", "x") as h:
        h.attrs.update(status="FROZEN_CODE_SNAPSHOT", base_producer_commit=commit,
                       paper_commit="cfa2f89", reserved_doi="10.5281/zenodo.24037714",
                       source_policy="Working bytes for tracked files plus local-import and explicit untracked helpers; all overlays individually hashed",
                       code_license="No project-wide lfm_meteor/FDTD declaration found; Rustplotlib MIT preserved",
                       limitations="External Python packages, Cargo registry dependencies, system HDF5/TeX/MPI and optional Metal toolchain not vendored; paths may require relocation")
        write_rows(h, "files", rows)
    archive = output / "code-final.tar.gz"
    with archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as tar:
            for name in sorted([r["path"] for r in rows]):
                path = output / name
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open("rb") as f:
                    tar.addfile(info, f)
    with h5py.File(output / "checksums.h5", "x") as h:
        write_rows(h, "archives", [dict(path=archive.name)])
        for path in (archive, output / "manifest.h5"):
            h.attrs[path.name] = sha(path)
            h.attrs[path.name + "_bytes"] = path.stat().st_size
    from package_submission import verify
    verify(output)
    print(json.dumps(dict(base_commit=commit, files=len(rows), overlays=[r["path"] for r in rows if r["source_status"] != "committed"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--expected-commit", required=True)
    a = p.parse_args()
    freeze(a.repo.resolve(), a.output.resolve(), a.expected_commit)
