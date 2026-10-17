"""Native Rustplotlib launcher. No Matplotlib or graphics-scene generation.

Only genuinely native builders are supported. There is no converter fallback.
Python APIs are welcome, but must call Rustplotlib, never Matplotlib.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / "sanya_tristatic_paper"
OUTPUT = ROOT / "results/paper_rustplotlib_native"
SOURCES = {
    "all_detection_time_delay_velocity": (
        "detection",
        ROOT / "results/native_paper_data/detection.h5",
    ),
    "meteor_height_histogram": ("height", ROOT / "results/native_paper_data/height.h5"),
    "tristatic_event_fit_summary": (
        "event",
        ROOT / "results/native_paper_data/tri_0162_1713823274259349823.h5",
    ),
    "tristatic_event_fit_well_constrained_radius_tri_0022_1713803698934349060": (
        "event",
        ROOT / "results/native_paper_data/tri_0022_1713803698934349060.h5",
    ),
    "joint_fft_mass_distribution": (
        "mass",
        ROOT
        / "results/joint_fft_mass_distribution_profile_fixedmask_capped_logr61_20260709.h5",
    ),
    "tristatic_orbit_elements_minus1000d": (
        "orbits",
        ROOT / "results/tristatic_orbits_minus1000d_jopek_williams_20260703.h5",
    ),
    "sanya_rcs_from_snr_article": ("rcs", ROOT / "results/native_paper_data/rcs.h5"),
    "sun_centered_ecliptic_radiants": (
        "radiants",
        ROOT / "results/native_radiant_scientific_data.h5",
    ),
    "appendix_dechirped_beat_example": (
        "beat",
        ROOT / "results/appendix_dechirped_beat_example_v20260702b.h5",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=SOURCES, nargs="+", default=list(SOURCES))
    parser.add_argument("--install-only", action="store_true")
    args = parser.parse_args()
    print(
        "Native Rust builders: nine statistical figures (2--10); geometry has a separate native builder.",
        flush=True,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.install_only:
        env = dict(os.environ, HDF5_DIR="/opt/anaconda3")
        env.pop("DYLD_LIBRARY_PATH", None)
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(ROOT / "rustplotlib_figures/Cargo.toml"),
                "--bin",
                "native-paper-figures",
            ],
            env=env,
            check=True,
        )
        env["DYLD_LIBRARY_PATH"] = "/opt/anaconda3/lib"
        binary = ROOT / "rustplotlib_figures/target/release/native-paper-figures"
        for name in args.only:
            kind, source = SOURCES[name]
            subprocess.run(
                [str(binary), kind, str(source), str(OUTPUT / name)],
                env=env,
                check=True,
            )
        return
    import h5py

    with h5py.File(OUTPUT / "validation.h5") as record:
        for name in args.only:
            for extension in ("pdf", "png", "svg"):
                source = OUTPUT / f"{name}.{extension}"
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest != record[name].attrs[f"{extension}_sha256"]:
                    raise RuntimeError(f"Revalidate changed native figure: {source}")
                shutil.copyfile(source, PAPER / "figures/rustplotlib" / source.name)
            if (OUTPUT / f"{name}.include.tex").exists():
                source = OUTPUT / f"{name}.include.tex"
                if hashlib.sha256(source.read_bytes()).hexdigest() != record[name].attrs["include_tex_sha256"]:
                    raise RuntimeError("Revalidate changed article inclusion")
                shutil.copyfile(source, PAPER / "figures/rustplotlib" / source.name)


if __name__ == "__main__":
    main()
