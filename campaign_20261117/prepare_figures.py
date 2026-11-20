"""Prepare scientific HDF5 inputs for the November Rustplotlib memo figures."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from analyze import SITES, STATIONS, ENU, C, save


def overview(root, out):
    day = int(np.datetime64("2026-11-17", "ns").astype(np.int64))
    times, delays, sites, counts = [], [], [], []
    for site in SITES:
        with h5py.File(root / ("20261117_"+site) / "catalogue.h5") as h:
            t = h["event_start_utc_ns"][:]
            r = h["event_median_apparent_range_km"][:]
            times.extend(t)
            delays.extend(r*1000/C*1e6*(2 if site == "sanya" else 1))
            sites.extend([site]*len(t))
            counts.append(np.histogram((t-day)/3.6e12, np.arange(11, 26))[0])
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "figure2.h5", "w") as h:
        h.attrs["utc_day_start_ns"] = day
        g = h.create_group("observations")
        g["time_ns"], g["delay_us"] = times, delays
        g.create_dataset("site", data=np.array(sites, dtype=h5py.string_dtype()))
        associations = h5py.File(out.parent / "associations.h5")
        indices = associations["station_event_indices"][:]
        associations.close()
        with h5py.File(root / "20261117_sanya/catalogue.h5") as source:
            tri = source["event_start_utc_ns"][:][indices[:, 0]]
        for site, count in zip(SITES + ("tristatic",), counts + [np.histogram((tri-day)/3.6e12, np.arange(11, 26))[0]]):
            g = h.create_group("counts/" + site)
            g["hour"], g["count"] = np.arange(11.5, 25), count.astype(float)
        for site in SITES:
            with h5py.File(root / ("20261117_"+site) / "noise.h5") as source:
                t = source["averages/utc_ns"][:]
                power = source["averages/raw_noise_power_adc2"][:]
            minutes = (t-day)//60000000000
            unique = np.unique(minutes)
            medians = np.array([np.median(power[minutes == minute]) for minute in unique])
            g = h.create_group("noise/"+site)
            g["floor_hour"], g["floor_power_1e9_adc2"] = (unique+.5)/60, medians/1e9
        h.attrs["noise_units"] = "ADC counts squared per complex sample; not calibrated kelvin"
        h.attrs["detections"] = "event-level median delay; no pulse-level thinning or inferred detections"
    print("PREPARED figure2", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/gpfs/pub/data_other/juha"))
    p.add_argument("--output", type=Path, default=Path("/gpfs/pub/data_other/juha/20261117_tristatic/figure_data"))
    a = p.parse_args()
    overview(a.root, a.output)


if __name__ == "__main__":
    main()
