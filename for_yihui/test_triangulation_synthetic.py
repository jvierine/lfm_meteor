import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jcoord


def load_for_yihui_module():
    spec = importlib.util.spec_from_file_location("for_yihui_script", SCRIPT_DIR / "for_yihui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fy = load_for_yihui_module()


def log(message):
    print(f"[test_triangulation_synthetic] {message}", flush=True)


class SyntheticTriangulationTest(unittest.TestCase):
    def test_fake_common_volume_target_triangulates_back_to_ecef_position(self):
        log("setting up station ECEF positions with vendored jcoord")
        station_ecef = {
            site: np.asarray(jcoord.geodetic2ecef(*llh), dtype=np.float64)
            for site, llh in fy.SITE_LLH.items()
        }
        for site, ecef in station_ecef.items():
            log(f"  {site:8s} ECEF: x={ecef[0]:.3f} m y={ecef[1]:.3f} m z={ecef[2]:.3f} m")

        target_range_m = 95e3
        sanya_az_deg = 14.996337890625
        sanya_el_deg = 74.9981689453125
        log(
            "placing fake target at common-volume center proxy: "
            f"{target_range_m / 1e3:.3f} km from Sanya along "
            f"az/el {sanya_az_deg:.12f}/{sanya_el_deg:.12f} deg"
        )
        target_llh = jcoord.az_el_r2geodetic(
            *fy.SITE_LLH["sanya"],
            sanya_az_deg,
            sanya_el_deg,
            target_range_m,
        )
        target_ecef = np.asarray(jcoord.geodetic2ecef(*target_llh), dtype=np.float64)
        log(
            "fake target geodetic: "
            f"lat={target_llh[0]:.12f} deg lon={target_llh[1]:.12f} deg alt={target_llh[2] / 1e3:.9f} km"
        )
        log(
            "fake target ECEF: "
            f"x={target_ecef[0]:.3f} m y={target_ecef[1]:.3f} m z={target_ecef[2]:.3f} m"
        )

        log("generating exact total path measurements from ECEF geometry")
        tx_to_target_m = np.linalg.norm(target_ecef - station_ecef["sanya"])
        measurements_m = np.asarray(
            [
                tx_to_target_m + np.linalg.norm(target_ecef - station_ecef["sanya"]),
                tx_to_target_m + np.linalg.norm(target_ecef - station_ecef["danzhou"]),
                tx_to_target_m + np.linalg.norm(target_ecef - station_ecef["wenchang"]),
            ],
            dtype=np.float64,
        )
        for site, path_m in zip(fy.ORDER, measurements_m):
            log(f"  {site:8s} total path: {path_m / 1e3:.9f} km")

        log("validating simulated measurements against for_yihui.predicted_paths")
        predicted_m = fy.predicted_paths(target_ecef)
        path_error_m = predicted_m - measurements_m
        for site, err_m in zip(fy.ORDER, path_error_m):
            log(f"  {site:8s} simulated measurement error: {err_m:.6e} m")
        np.testing.assert_allclose(path_error_m, 0.0, atol=1e-7)

        log("running for_yihui.triangulate on the synthetic path triplet")
        points, lat, lon, height_km, residuals_m, fit_info = fy.triangulate(measurements_m[None, :])
        solved_ecef = points[0]
        solved_error_m = np.linalg.norm(solved_ecef - target_ecef)
        log(
            "triangulated geodetic: "
            f"lat={lat[0]:.12f} deg lon={lon[0]:.12f} deg alt={height_km[0]:.9f} km"
        )
        log(
            "triangulated ECEF: "
            f"x={solved_ecef[0]:.3f} m y={solved_ecef[1]:.3f} m z={solved_ecef[2]:.3f} m"
        )
        log(f"3D ECEF position error: {solved_error_m:.6e} m")
        log(f"path residuals: {residuals_m[0]}")
        log(f"fit info: {fit_info}")

        self.assertEqual(fit_info["success_count"], 1)
        np.testing.assert_allclose(solved_ecef, target_ecef, atol=1e-3)
        np.testing.assert_allclose(residuals_m[0], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
