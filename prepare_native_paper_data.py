"""Prepare scientific arrays for Rust figure builders, never graphics scenes.

The import guard makes any accidental Matplotlib use a hard error. Existing
forward-model and antenna calculations are reused without calling plot methods.
"""

import argparse
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import h5py
import numpy as np


class ForbiddenMatplotlibLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError("Matplotlib is forbidden in native figure preparation")


class NoMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            # Astropy probes optional availability with find_spec; forbid the
            # actual import, not that harmless dependency probe.
            return importlib.util.spec_from_loader(
                fullname, ForbiddenMatplotlibLoader()
            )
        return None


sys.meta_path.insert(0, NoMatplotlib())
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/native_paper_data"
CATALOG = (
    ROOT / "results/tristatic_shrinking_radius_profile_fixedmask_capped_logr61_20260708"
)


def write_values(group, values):
    for key, value in values.items():
        if isinstance(value, str) or np.isscalar(value):
            group.attrs[key] = value
        else:
            group[key] = np.asarray(value)


def height():
    import plot_height_histogram as science

    alt, _ = science.collect_heights()
    mono, _, _ = science.collect_sanya_monostatic_heights()
    heights = np.linspace(80.0, 120.0, 401)
    profiles = science.receiver_gain_profiles(heights)
    beam = science.beam_panel_data(science.BEAM_TRACE_CATALOG_DIR)
    with h5py.File(OUT / "height.h5", "w") as h:
        h.attrs["science_recipe"] = (
            "plot_height_histogram.py: collect_heights, receiver_gain_profiles, beam_panel_data"
        )
        h["tristatic_height_km"] = alt
        h["monostatic_height_km"] = mono
        h["gain_height_km"] = heights
        for site, gain in profiles.items():
            h[f"gain_{site}"] = gain
        for key in ("east_edges", "north_edges", "gain_db", "n_events", "n_positions"):
            if np.isscalar(beam[key]):
                h.attrs[key] = beam[key]
            else:
                h[key] = beam[key]
        h["counts"] = beam["counts"].filled(np.nan)
        h["gain_grid_deg"] = beam["east_grid"][0]
        h.attrs["containment_level_db"] = science.load_sanya_containment_contour()[
            "cutoff_relative_gain_db"
        ]
    print("Scientific height/beam arrays prepared", flush=True)


def event(event_id):
    import fit_event_joint_delay_doppler_fft as science
    import fit_all_ballistic_snr_weighted as base

    source = CATALOG / f"joint_delay_doppler_fft_{event_id}.h5"
    with h5py.File(source) as original:
        joint = dict(original["joint_fit"].attrs)
        joint.update(
            {
                k: v[()]
                for k, v in original["joint_fit"].items()
                if isinstance(v, h5py.Dataset)
            }
        )
        joint = {k: v.decode() if isinstance(v, bytes) else v for k, v in joint.items()}
        density, _ = base.density_interpolator(joint["time_ns"], joint["x_itrs_m"])
        bands = science.model_uncertainty_bands(event_id, joint, density)
        full = science.full_event_beam_bootstrap_band(joint, density)
        display = joint.get("all_x_itrs_m", joint["x_itrs_m"])
        east, north = science.sanya_beam_offsets_deg(display)
        along = np.nanmean(joint["v_gcrs_mps"], axis=0)
        along /= np.linalg.norm(along)
        grid = np.linspace(-2.6, 2.6, 240)
        xx, yy = np.meshgrid(grid, grid)
        with h5py.File(OUT / f"{event_id}.h5", "w") as h:
            h.attrs["source"] = str(source)
            h.attrs["event_id"] = event_id
            for key in ("joint_fit", "fft_observations"):
                original.copy(key, h)
            d = h.create_group("derived")
            d["beam_east_deg"] = east
            d["beam_north_deg"] = north
            d["along_velocity_km_s"] = joint["v_gcrs_mps"] @ along / 1000.0
            d["gain_grid_deg"] = grid
            d["gain_db"] = science.beam_hist.sanya_beam_relative_gain_db(xx, yy)
            if bands is not None:
                write_values(d.create_group("bands"), bands)
            if full is not None:
                write_values(d.create_group("full_beam_band"), full)
    print(f"Scientific event arrays prepared: {event_id}", flush=True)


def rcs():
    with h5py.File(OUT / "rcs.h5", "w") as h:
        for group, path in [
            ("pulses", "results/sanya_rcs_from_snr.h5"),
            ("events", "results/mass_rcs_relation_wj_inbeam_good_mass_20260709.h5"),
        ]:
            with h5py.File(ROOT / path) as original:
                target = h.create_group(group)
                target.attrs["source"] = path
                for key in original:
                    original.copy(key, target)
    print("Scientific RCS arrays prepared", flush=True)


def detection():
    from astropy.time import Time
    import plot_pygdsm_station_sky_noise as sky_science
    from sanya_sky_data import GSM2008
    import noise_model

    with h5py.File(
        ROOT / "results/paper_rustplotlib/detection_observations.h5"
    ) as old, h5py.File(OUT / "detection.h5", "w") as h:
        observations = h.create_group("observations")
        for key in old:
            old.copy(key, observations)
        names = np.asarray(observations["site"].asstr()[()])
        time = observations["time_ns"][()]
        event = observations["event_id"].asstr()[()]
        day = int(time.min() // 86400000000000 * 86400000000000)
        h.attrs["utc_day_start_ns"] = day
        counts = h.create_group("counts")
        for site in ("sanya", "danzhou", "wenchang"):
            keep = names == site
            hrs = ((time[keep] - day) // 3600000000000).astype(int)
            group = counts.create_group(site)
            hours = np.arange(hrs.min(), hrs.max() + 1)
            group["hour"] = hours + 0.5
            group["count"] = [len(set(event[keep][hrs == hour])) for hour in hours]
        with h5py.File(ROOT / "results/tristatic_results.h5") as associations:
            ids = associations["summary_event_id"].asstr()[()]
            stamps = np.array(
                [int(value.rsplit("_", 1)[1]) - 8 * 3600000000000 for value in ids]
            )
        hours = ((stamps - day) // 3600000000000).astype(int)
        bins = np.arange(hours.min(), hours.max() + 1)
        group = counts.create_group("tristatic")
        group["hour"] = bins + 0.5
        group["count"] = [np.count_nonzero(hours == hour) for hour in bins]
        with h5py.File(
            ROOT / "results/sanya_4mhz_system_noise_power_100pulse.h5"
        ) as noise:
            nt = noise["bins/time_utc_mid_ns"][()]
            sid = noise["bins/station_id"][()]
            power = noise["bins/noise_power_mean_raw_voltage"][()].astype(float)
        start, stop = int(nt.min()), int(nt.max())
        step = 150000000000
        edges = np.arange(start, stop + step, step, dtype=np.int64)
        edges[-1] = stop
        model_ns = np.arange(start, stop + step, step, dtype=np.int64)
        model_ns[-1] = min(model_ns[-1], stop)
        times = Time(model_ns / 1e9, format="unix", scale="utc")
        model = GSM2008(430.0)
        noise_group = h.create_group("noise")
        for station, site in enumerate(("Sanya", "Danzhou", "Wenchang")):
            valid = (sid == station) & np.isfinite(power) & (power > 0.0)
            indices = np.searchsorted(edges, nt[valid], side="right") - 1
            vals = power[valid]
            quantile = 0.5 if station == 0 else 0.1
            indices_valid = np.unique(
                indices[(indices >= 0) & (indices < len(edges) - 1)]
            )
            floor = np.array(
                [np.quantile(vals[indices == i], quantile) for i in indices_valid]
            )
            centres = edges[:-1] + np.diff(edges) // 2
            sky, _ = sky_science.station_sky_temperature(
                model, site, times, 430.0, False, 5.0, 0.1, 1.0
            )
            group = noise_group.create_group(site.lower())
            group["floor_hour"] = (centres[indices_valid] - day) / 3.6e12
            group["floor_tsys_k"] = floor / noise_model.POWER_PER_K[site]
            group["model_hour"] = (model_ns - day) / 3.6e12
            group["model_tsys_k"] = sky + noise_model.FITTED_T_REC_K[site]
            group.attrs["receiver_temperature_k"] = noise_model.FITTED_T_REC_K[site]
            print(
                f"Scientific noise series: {site}, sky {sky.min():.2f}--{sky.max():.2f} K",
                flush=True,
            )
    print("Scientific detection/noise arrays prepared", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["height", "event", "rcs", "detection"])
    parser.add_argument("--event", default="tri_0162_1713823274259349823")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    {
        "height": height,
        "rcs": rcs,
        "detection": detection,
        "event": lambda: event(args.event),
    }[args.kind]()
    assert not any(
        k == "matplotlib" or k.startswith("matplotlib.") for k in sys.modules
    )
    filename = f"{args.event}.h5" if args.kind == "event" else f"{args.kind}.h5"
    with h5py.File(OUT / filename, "r+") as h:
        h.attrs["matplotlib_loaded"] = False
        h.attrs["data_preparation_program"] = __file__


if __name__ == "__main__":
    main()
