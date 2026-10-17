"""The plotting boundary must remain completely independent of Matplotlib."""

import ast
from pathlib import Path
import unittest
import os
import subprocess
import tempfile
import h5py
import numpy as np
from build_paper_figures_rustplotlib import ROOT, SOURCES


class NativeFigureTests(unittest.TestCase):
    def test_all_figure_fonts_and_outer_margins_at_article_size(self):
        import re
        import pymupdf
        from build_paper_figures_rustplotlib import OUTPUT, PAPER
        for name in [*SOURCES, "sanya_system_geometry"]:
            with self.subTest(figure=name):
                inclusion = (PAPER / "figures/rustplotlib" / f"{name}.include.tex").read_text()
                width = float(re.search(r"includegraphics\[width=([0-9.]+)mm", inclusion)[1])
                font = float(re.search(r"body/label font ([0-9.]+) TeX pt", inclusion)[1])
                with pymupdf.open(PAPER / "figures/rustplotlib" / f"{name}.pdf") as pdf:
                    page = pdf[0]
                    export_width = page.rect.width * 25.4 / 72.
                    self.assertAlmostEqual(font * width / export_width, 10., places=3)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                    yy, xx = np.where(np.min(rgb, axis=2) < 250)
                    margins = np.array([xx.min(), yy.min(), pix.width-1-xx.max(), pix.height-1-yy.max()]) * export_width / pix.width
                    self.assertLess(np.max(margins), 1.7, str(margins))
                self.assertIn(r"\input{figures/rustplotlib/" + name + ".include.tex}", (PAPER / "main.tex").read_text())

    def test_radiant_font_size_at_actual_aa_article_inclusion(self):
        import re
        import pymupdf
        from build_paper_figures_rustplotlib import OUTPUT, PAPER
        name = "sun_centered_ecliptic_radiants"
        inclusion = (OUTPUT / f"{name}.include.tex").read_text()
        width = float(re.search(r"includegraphics\[width=([0-9.]+)mm", inclusion)[1])
        font = float(re.search(r"body/label font ([0-9.]+) TeX pt", inclusion)[1])
        with pymupdf.open(OUTPUT / f"{name}.pdf") as pdf:
            export_width = pdf[0].rect.width * 25.4 / 72.
        printed_font = font * width / export_width
        self.assertAlmostEqual(printed_font, 10., places=3)
        self.assertIn(r"\input{figures/rustplotlib/" + name + ".include.tex}", (PAPER / "main.tex").read_text())
        self.assertIn("Article linewidth changed", inclusion)
        self.assertIn("Article body font changed", inclusion)
        self.assertIn(r"\setlength\aa@fullwidth{184mm}", (PAPER / "aa.cls").read_text())

    def test_all_nine_native_builders_are_registered(self):
        self.assertEqual(len(SOURCES), 9)
        self.assertEqual({kind for kind, _ in SOURCES.values()},
                         {"detection", "height", "event", "mass", "rcs", "radiants", "orbits", "beat"})

    def test_mass_selection_and_asymmetric_intervals(self):
        with h5py.File(SOURCES["joint_fft_mass_distribution"][1]) as data:
            selected = data["selected"][()].astype(bool)
            self.assertEqual(np.count_nonzero(selected), 110)
            mass = data["initial_mass_kg"][()][selected]
            lo = data["mass_95_lo_kg"][()][selected]
            hi = data["mass_95_hi_kg"][()][selected]
            self.assertTrue(np.all(np.isfinite(lo) & np.isfinite(hi)))
            self.assertTrue(np.all((lo > 0) & (hi > lo)))
            self.assertFalse(np.allclose((hi-mass)/mass, (mass-lo)/mass))

    def test_event_scientific_arrays_are_copied_exactly(self):
        for kind, path in SOURCES.values():
            if kind != "event":
                continue
            with h5py.File(path) as data:
                source_path = data.attrs["source"]
            with h5py.File(path) as native, h5py.File(source_path) as source:
                def check(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        np.testing.assert_array_equal(obj[()], native[name][()], err_msg=name)
                source["joint_fit"].visititems(lambda name, obj: check("joint_fit/"+name, obj))
                source["fft_observations"].visititems(lambda name, obj: check("fft_observations/"+name, obj))

    def test_scientific_helpers_do_not_import_matplotlib(self):
        run = subprocess.run([os.sys.executable, "-c",
            "import prepare_native_paper_data; import fit_event_joint_delay_doppler_fft; "
            "import plot_height_histogram; import plot_pygdsm_station_sky_noise; "
            "import sys; assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_python_entrypoints_have_no_matplotlib_or_recipe_imports(self):
        for filename in (
            "build_paper_figures_rustplotlib.py",
            "prepare_native_radiant_data.py",
            "validate_paper_rustplotlib.py",
        ):
            tree = ast.parse((ROOT / filename).read_text())
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [name.name for name in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                self.assertFalse(
                    any(
                        name.startswith(("matplotlib", "plot_", "rustplotlib_scene"))
                        for name in modules
                    ),
                    filename,
                )

    def test_native_data_are_scientific_arrays_not_drawing_scenes(self):
        for name, (_, path) in SOURCES.items():
            with h5py.File(path) as data:
                self.assertNotIn("paths", data, name)
                self.assertNotIn("labels", data, name)
                self.assertNotIn("size_pt", data, name)

    def test_radiant_measurements_are_unchanged(self):
        with h5py.File(
            ROOT / "results/sun_centered_ecliptic_radiants.h5"
        ) as source, h5py.File(SOURCES["sun_centered_ecliptic_radiants"][1]) as native:
            for name in (
                "lambda_minus_sun_deg",
                "beta_ecliptic_deg",
                "speed_km_s",
                "t0_ns",
            ):
                np.testing.assert_array_equal(source[name][()], native[name][()])
            self.assertTrue(np.all(native["effective_hours"][()] >= 0.0))
            self.assertTrue(
                np.all(
                    native["effective_hours"][()] <= native["visible_hours"][()] + 1e-12
                )
            )

    def test_converter_entrypoints_are_gone(self):
        for filename in ("rustplotlib_scene.py", "rustplotlib_figures/src/main.rs"):
            self.assertFalse((ROOT / filename).exists())

    def test_rust_binary_rejects_legacy_graphics_scenes(self):
        binary = ROOT / "rustplotlib_figures/target/release/native-paper-figures"
        scene = ROOT / "results/paper_rustplotlib/appendix_dechirped_beat_example.h5"
        if not binary.exists() or not scene.exists():
            self.skipTest(
                "Build the native binary first; historical scene needed only for rejection test"
            )
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run(
                [str(binary), "beat", str(scene), str(Path(directory) / "forbidden")],
                env=dict(os.environ, DYLD_LIBRARY_PATH="/opt/anaconda3/lib"),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("not an exported graphics scene", run.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
