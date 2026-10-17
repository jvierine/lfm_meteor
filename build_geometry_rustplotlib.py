"""Build the alternative geometry figure natively, without replacing Figure 1."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pymupdf

from prepare_native_geometry_data import main as prepare

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / "sanya_tristatic_paper"
OUTPUT = ROOT / "results/paper_rustplotlib_native"
NAME = "sanya_system_geometry"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = PAPER / "figures/sanya_tri_static_meteor_head_echo_trajectories.png"
    original_hash = sha(original)
    prepare()
    env = dict(os.environ, HDF5_DIR="/opt/anaconda3")
    env.pop("DYLD_LIBRARY_PATH", None)
    subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "rustplotlib_figures", env=env, check=True)
    data = ROOT / "results/native_paper_data/system_geometry.h5"
    subprocess.run([str(ROOT / "rustplotlib_figures/target/release/native-paper-figures"),
                    "geometry", str(data), str(OUTPUT / NAME)],
                   env=dict(env, DYLD_LIBRARY_PATH="/opt/anaconda3/lib"), check=True)
    with pymupdf.open(OUTPUT / f"{NAME}.pdf") as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1800 / page.rect.width, 1800 / page.rect.width), alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        assert np.mean(np.min(rgb, axis=2) < 200) > 0.005
        # Native LaTeX glyphs are vector outlines, not extractable PDF text.
        assert "<image" not in (OUTPUT / f"{NAME}.svg").read_text(), "Unexpected raster fallback"
        review = OUTPUT / "pdf_review"
        review.mkdir(exist_ok=True)
        pix.save(review / f"{NAME}.png")
    with h5py.File(OUTPUT / f"{NAME}_validation.h5", "w") as record:
        record.attrs["source_program"] = "rustplotlib_figures/src/native/geometry.rs"
        record.attrs["original_figure_sha256"] = original_hash
        record.attrs["scientific_geometry_sha256"] = sha(data)
        record.attrs["rust_builder_sha256"] = sha(ROOT / "rustplotlib_figures/src/native/geometry.rs")
        record.attrs["projection_source_sha256"] = sha(ROOT.parent / "rustplotlib/src/perspective.rs")
        for ext in ("pdf", "svg", "png", "include.tex"):
            source = OUTPUT / f"{NAME}.{ext}"
            record.attrs[f"{ext}_sha256"] = sha(source)
            shutil.copy2(source, PAPER / "figures/rustplotlib" / source.name)
    assert sha(original) == original_hash, "Original Figure 1 was modified"
    print(f"Alternative figure, original unchanged: {PAPER / 'figures/rustplotlib' / (NAME+'.pdf')}")


if __name__ == "__main__":
    main()
