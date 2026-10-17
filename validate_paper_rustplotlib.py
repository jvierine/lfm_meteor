"""Validate native Rust figure exports, not graphics scenes or legacy figures."""

import hashlib
import re
from pathlib import Path
import h5py
import numpy as np
import pymupdf
from build_paper_figures_rustplotlib import ROOT, OUTPUT, SOURCES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    review = OUTPUT / "pdf_review"
    review.mkdir(exist_ok=True)
    with h5py.File(OUTPUT / "validation.h5", "w") as record:
        record.attrs["architecture"] = (
            "Native Rust figure builders; scientific HDF5 only; no Matplotlib"
        )
        for path in (ROOT / "rustplotlib_figures/src").rglob("*.rs"):
            record.attrs[str(path.relative_to(ROOT))] = sha(path)
        library = record.create_group("rustplotlib_library_sources")
        for path in (ROOT.parent / "rustplotlib/src").rglob("*.rs"):
            library.attrs[str(path.relative_to(ROOT.parent / "rustplotlib"))] = sha(
                path
            )
        for name in (
            "build_paper_figures_rustplotlib.py",
            "prepare_native_radiant_data.py",
            "prepare_native_paper_data.py",
            "sanya_sky_data.py",
        ):
            record.attrs[name] = sha(ROOT / name)
        for name, (kind, source) in SOURCES.items():
            assert (OUTPUT / f"{name}.pdf").exists(), f"Missing native figure: {name}"
            with h5py.File(source) as data:
                assert (
                    "paths" not in data and "labels" not in data
                ), "Graphics scenes are forbidden"
            group = record.create_group(name)
            group.attrs["source"] = str(source)
            group.attrs["source_sha256"] = sha(source)
            group.attrs["native_builder"] = kind
            for extension in ("pdf", "png", "svg"):
                group.attrs[f"{extension}_sha256"] = sha(OUTPUT / f"{name}.{extension}")
            inclusion = (OUTPUT / f"{name}.include.tex").read_text()
            group.attrs["include_tex_sha256"] = sha(OUTPUT / f"{name}.include.tex")
            width = float(re.search(r"includegraphics\[width=([0-9.]+)mm", inclusion)[1])
            font = float(re.search(r"body/label font ([0-9.]+) TeX pt", inclusion)[1])
            with pymupdf.open(OUTPUT / f"{name}.pdf") as pdf:
                assert len(pdf) == 1
                page = pdf[0]
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(
                        1600 / page.rect.width, 1600 / page.rect.width
                    ),
                    alpha=False,
                )
                pixels = np.frombuffer(pix.samples, dtype=np.uint8)
                assert np.mean(pixels < 240) > 0.005
                export_width = page.rect.width * 25.4 / 72.
                assert abs(font * width / export_width - 10.) < 0.001, name
                group.attrs["article_width_mm"] = width
                group.attrs["article_font_tex_pt"] = font
                rgb = pixels.reshape(pix.height, pix.width, pix.n)[:, :, :3]
                yy, xx = np.where(np.min(rgb, axis=2) < 250)
                margins = np.array([xx.min(), yy.min(), pix.width-1-xx.max(), pix.height-1-yy.max()]) * export_width / pix.width
                assert np.max(margins) < 1.7, (name, margins)
                group["outer_ink_margins_mm"] = margins
                if kind == "radiants":
                    assert export_width <= 184.01
                    # Verify the actual PDF colour strip, not only source pixels.
                    columns = np.arange(int(pix.width * 0.86), int(pix.width * 0.92))
                    saturation = np.ptp(rgb, axis=2)
                    column = columns[
                        np.argmax(np.sum(saturation[:, columns] > 20, axis=0))
                    ]
                    rows = np.flatnonzero(saturation[:, column] > 20)
                    assert (
                        len(rows) > pix.height * 0.45
                    ), "Missing native vertical colourbar"
                    strip = rgb[rows[0] + 3 : rows[-1] - 3, column].astype(int)
                    step = int(np.max(np.abs(np.diff(strip, axis=0))))
                    assert step < 16, f"Native PDF colourbar seams: {step}"
                    group.attrs["colourbar_max_adjacent_rgb_step"] = step
                pix.save(review / f"{name}.png")
                group["page_size_pt"] = [page.rect.width, page.rect.height]
            print(f"Validated native PDF: {name}", flush=True)
        assert len(SOURCES) == 9, "Expected all nine statistical figures"
        record.attrs["validated_figure_count"] = len(SOURCES)


if __name__ == "__main__":
    main()
