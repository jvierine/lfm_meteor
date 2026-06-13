import os

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import CartesianRepresentation, GCRS, GeocentricTrueEcliptic, SkyCoord, get_sun
from astropy.time import Time


INPUT_H5 = os.path.join("results", "all_tristatic_ballistic_snr_weighted_v20260611c.h5")
OUTPUT_H5 = os.path.join("results", "sun_centered_ecliptic_radiants_v20260611c.h5")
OUTPUT_PNG = os.path.join("results", "sun_centered_ecliptic_radiants.png")
DIAGNOSTIC_OUTPUT_PNG = os.path.join("results", "sun_centered_ecliptic_radiants_diagnostic.png")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/sun_centered_ecliptic_radiants.png"
PLOT_CENTER_LONGITUDE_DEG = 270.0


def fixed_ecliptic_equinox(times):
    """Use one ecliptic coordinate system for all event radiants."""
    median_unix = float(np.nanmedian(times.unix))
    return Time(median_unix, format="unix", scale="utc")


def wrap180(deg):
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def wrap360(deg):
    return np.asarray(deg, dtype=np.float64) % 360.0


def centered_plot_longitude_deg(deg):
    return -wrap180(np.asarray(deg, dtype=np.float64) - PLOT_CENTER_LONGITUDE_DEG)


def centered_tick_labels():
    tick_positions_deg = np.arange(-150.0, 180.0, 30.0)
    return [f"{int(wrap360(PLOT_CENTER_LONGITUDE_DEG - tick))}°" for tick in tick_positions_deg]


def load_fits(path):
    with h5py.File(path, "r") as h:
        event_id = np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h["event_id"][:]])
        if "t0_ns" in h and "v0_gcrs_mps" in h:
            t0_ns = h["t0_ns"][:]
            v0_gcrs_mps = h["v0_gcrs_mps"][:]
            speed_km_s = h["speed_km_s"][:]
        else:
            t0_ns = np.asarray([h["points"][name]["time_ns"][0] for name in event_id], dtype=np.int64)
            v0_gcrs_mps = np.asarray([h["points"][name]["params"][:][3:6] for name in event_id], dtype=np.float64)
            if "start_speed_km_s" in h:
                speed_km_s = h["start_speed_km_s"][:]
            else:
                speed_km_s = np.linalg.norm(v0_gcrs_mps, axis=1) / 1e3
        rms_total_path_residual_m = h["rms_total_path_residual_m"][:]
        n_points = h["n_points"][:]
    return event_id, t0_ns, v0_gcrs_mps, speed_km_s, rms_total_path_residual_m, n_points


def calculate_radiants(t0_ns, v0_gcrs_mps):
    times = Time(np.asarray(t0_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    fixed_equinox = fixed_ecliptic_equinox(times)
    radiant_unit = -v0_gcrs_mps / np.linalg.norm(v0_gcrs_mps, axis=1)[:, None]
    radiant_gcrs = SkyCoord(
        GCRS(
            CartesianRepresentation(
                radiant_unit[:, 0] * u.one,
                radiant_unit[:, 1] * u.one,
                radiant_unit[:, 2] * u.one,
            ),
            obstime=times,
        )
    )
    ecliptic_frame = GeocentricTrueEcliptic(obstime=times, equinox=fixed_equinox)
    radiant_ecl = radiant_gcrs.transform_to(ecliptic_frame)
    sun_ecl = get_sun(times).transform_to(ecliptic_frame)

    lambda_deg = radiant_ecl.lon.to_value(u.deg)
    beta_deg = radiant_ecl.lat.to_value(u.deg)
    sun_lambda_deg = sun_ecl.lon.to_value(u.deg)
    sun_centered_lambda_deg = wrap360(lambda_deg - sun_lambda_deg)
    return lambda_deg, beta_deg, sun_lambda_deg, sun_centered_lambda_deg, fixed_equinox.isot


def write_h5(path, event_id, t0_ns, speed_km_s, rms_total_path_residual_m, n_points, lambda_deg, beta_deg, sun_lambda_deg, sun_centered_lambda_deg, fixed_equinox_iso):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["input_h5"] = INPUT_H5
        h.attrs["coordinate_frame"] = "GeocentricTrueEcliptic"
        h.attrs["fixed_ecliptic_equinox_utc"] = fixed_equinox_iso
        h.attrs["longitude_convention"] = "lambda_radiant - lambda_sun wrapped to [0, 360) deg"
        h.attrs["radiant_definition"] = "incoming radiant direction = -v0_gcrs"
        h["event_id"] = event_id.astype(string_dtype)
        h["t0_ns"] = t0_ns
        h["speed_km_s"] = speed_km_s
        h["rms_total_path_residual_m"] = rms_total_path_residual_m
        h["n_points"] = n_points
        h["lambda_ecliptic_deg"] = lambda_deg
        h["beta_ecliptic_deg"] = beta_deg
        h["sun_lambda_ecliptic_deg"] = sun_lambda_deg
        h["lambda_minus_sun_deg"] = sun_centered_lambda_deg


def plot_radiants(lambda_minus_sun_deg, beta_deg, speed_km_s):
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    x_rad = np.deg2rad(centered_plot_longitude_deg(lambda_minus_sun_deg))
    y_rad = np.deg2rad(beta_deg)

    fig = plt.figure(figsize=(8.4, 5.1))
    ax = fig.add_subplot(111, projection="hammer")
    sc = ax.scatter(
        x_rad,
        y_rad,
        c=speed_km_s,
        s=24,
        cmap="viridis",
        alpha=0.76,
        linewidths=0.25,
        edgecolors="white",
    )
    ax.grid(True, alpha=0.45)
    ax.set_xlabel(r"Sun-centered ecliptic longitude, $\lambda-\lambda_\odot$", labelpad=12)
    ax.set_ylabel(r"Ecliptic latitude, $\beta$", labelpad=12)
    ax.set_xticklabels(centered_tick_labels())
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.15, fraction=0.055)
    cb.set_label("Fitted speed (km s$^{-1}$)")
    cb.ax.xaxis.labelpad = 8
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    os.makedirs(os.path.dirname(PAPER_OUTPUT_PNG), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_radiant_diagnostic(lambda_minus_sun_deg, beta_deg, speed_km_s):
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    x_rad = np.deg2rad(centered_plot_longitude_deg(lambda_minus_sun_deg))
    y_rad = np.deg2rad(beta_deg)

    fig = plt.figure(figsize=(8.8, 5.8))
    ax = fig.add_subplot(111, projection="hammer")
    sc = ax.scatter(
        x_rad,
        y_rad,
        c=speed_km_s,
        s=24,
        cmap="viridis",
        alpha=0.68,
        linewidths=0.25,
        edgecolors="white",
        label="All fitted radiants",
    )

    source_markers = [
        ("Helion", 0.0, 0.0, "*", "#f4a300"),
        ("Apex", 270.0, 0.0, "^", "#1f77b4"),
        ("Anti-apex", 90.0, 0.0, "v", "#9467bd"),
        ("Antihelion", 180.0, 0.0, "s", "#555555"),
    ]
    labelled_sources = set()
    for label, lon_deg, lat_deg, marker, color in source_markers:
        ax.scatter(
            np.deg2rad(centered_plot_longitude_deg(lon_deg)),
            np.deg2rad(lat_deg),
            marker=marker,
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label=label if label not in labelled_sources else None,
        )
        labelled_sources.add(label)

    ax.grid(True, alpha=0.45)
    ax.set_xlabel(r"Sun-centered ecliptic longitude, $\lambda-\lambda_\odot$", labelpad=12)
    ax.set_ylabel(r"Ecliptic latitude, $\beta$", labelpad=12)
    ax.set_xticklabels(centered_tick_labels())
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.15, fraction=0.055)
    cb.set_label("Fitted geocentric speed (km s$^{-1}$)")
    cb.ax.xaxis.labelpad = 8
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.03), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    event_id, t0_ns, v0_gcrs_mps, speed_km_s, rms_total_path_residual_m, n_points = load_fits(INPUT_H5)
    lambda_deg, beta_deg, sun_lambda_deg, sun_centered_lambda_deg, fixed_equinox_iso = calculate_radiants(t0_ns, v0_gcrs_mps)
    write_h5(
        OUTPUT_H5,
        event_id,
        t0_ns,
        speed_km_s,
        rms_total_path_residual_m,
        n_points,
        lambda_deg,
        beta_deg,
        sun_lambda_deg,
        sun_centered_lambda_deg,
        fixed_equinox_iso,
    )
    plot_radiants(sun_centered_lambda_deg, beta_deg, speed_km_s)
    plot_radiant_diagnostic(sun_centered_lambda_deg, beta_deg, speed_km_s)
    print(f"radiants: {len(event_id)}")
    print(f"fixed ecliptic equinox UTC: {fixed_equinox_iso}")
    print(f"lambda-lambda_sun deg [0,360) median/range: {np.nanmedian(sun_centered_lambda_deg):.2f} / {np.nanmin(sun_centered_lambda_deg):.2f} to {np.nanmax(sun_centered_lambda_deg):.2f}")
    print(f"beta deg median/range: {np.nanmedian(beta_deg):.2f} / {np.nanmin(beta_deg):.2f} to {np.nanmax(beta_deg):.2f}")
    print(OUTPUT_H5)
    print(OUTPUT_PNG)
    print(DIAGNOSTIC_OUTPUT_PNG)
    print(PAPER_OUTPUT_PNG)


if __name__ == "__main__":
    main()
