import os

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, CartesianDifferential, CartesianRepresentation, EarthLocation, GCRS, GeocentricTrueEcliptic, ITRS, SkyCoord, get_sun
from astropy.time import Time

import sanya_opts as sc


INPUT_H5 = os.path.join("results", "all_tristatic_ballistic_snr_weighted_v20260613b.h5")
OUTPUT_H5 = os.path.join("results", "sun_centered_ecliptic_radiants_v20260613b.h5")
OUTPUT_PNG = os.path.join("results", "sun_centered_ecliptic_radiants.png")
DIAGNOSTIC_OUTPUT_PNG = os.path.join("results", "sun_centered_ecliptic_radiants_diagnostic.png")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/sun_centered_ecliptic_radiants.png"
PAPER_OUTPUT_PDF = "/Users/jvi019/src/sanya_tristatic_paper/figures/sun_centered_ecliptic_radiants.pdf"
PAPER_INTERVAL_TEX = "/Users/jvi019/src/sanya_tristatic_paper/tables/radiant_observation_interval.tex"
PLOT_CENTER_LONGITUDE_DEG = 270.0
VISIBILITY_TIME_STEP_MIN = 5.0
RADIANT_ALTITUDE_GAMMA = 1.47


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


def plot_longitude_to_sun_centered_deg(plot_longitude_deg):
    return wrap360(PLOT_CENTER_LONGITUDE_DEG - np.asarray(plot_longitude_deg, dtype=np.float64))


def draw_shifted_longitude_labels(ax, latitude_deg=-11.0):
    tick_positions_deg = np.arange(-150.0, 180.0, 30.0)
    ax.set_xticks(np.deg2rad(tick_positions_deg))
    ax.set_xticklabels([""] * len(tick_positions_deg))
    for tick_deg, label in zip(tick_positions_deg, centered_tick_labels()):
        ax.text(
            np.deg2rad(tick_deg),
            np.deg2rad(latitude_deg),
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
        )


def add_source_markers(ax, labelled=False):
    source_markers = [
        ("Helion", 0.0, 0.0, "o", "#ffd21f", 95),
        ("Apex", 270.0, 0.0, r"$\otimes$", "black", 175),
        ("Antihelion", 180.0, 0.0, "o", "black", 95),
    ]
    for label, lon_deg, lat_deg, marker, color, size in source_markers:
        ax.scatter(
            np.deg2rad(centered_plot_longitude_deg(lon_deg)),
            np.deg2rad(lat_deg),
            marker=marker,
            s=size,
            color=color,
            edgecolor="black" if marker == "o" else None,
            linewidth=0.35 if marker == "o" else 0.0,
            zorder=1.5,
            label=label if labelled else None,
        )


def load_fits(path):
    with h5py.File(path, "r") as h:
        event_id = np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h["event_id"][:]])
        if "t0_ns" in h and "v0_gcrs_mps" in h:
            t0_ns = h["t0_ns"][:]
            position_m = None
            velocity_mps = h["v0_gcrs_mps"][:]
            speed_km_s = h["speed_km_s"][:]
            state_frame = "GCRS"
        else:
            t0_ns = np.asarray([h["points"][name]["time_ns"][0] for name in event_id], dtype=np.int64)
            first_group = h["points"][event_id[0]]
            if "v_gcrs_mps" in first_group:
                position_m = None
                velocity_mps = np.asarray([h["points"][name]["v_gcrs_mps"][0] for name in event_id], dtype=np.float64)
                state_frame = "GCRS"
            else:
                position_m = np.asarray([h["points"][name]["x_itrs_m"][0] for name in event_id], dtype=np.float64)
                velocity_mps = np.asarray([h["points"][name]["v_itrs_mps"][0] for name in event_id], dtype=np.float64)
                state_frame = "ITRS"
            speed_km_s = h["start_speed_km_s"][:] if "start_speed_km_s" in h else np.linalg.norm(velocity_mps, axis=1) / 1e3
        rms_total_path_residual_m = h["rms_total_path_residual_m"][:]
        n_points = h["n_points"][:]
    return event_id, t0_ns, position_m, velocity_mps, speed_km_s, rms_total_path_residual_m, n_points, state_frame


def velocities_to_gcrs(t0_ns, position_m, velocity_mps, state_frame):
    times = Time(np.asarray(t0_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    if state_frame == "GCRS":
        return times, np.asarray(velocity_mps, dtype=np.float64)
    if state_frame != "ITRS":
        raise ValueError(f"Unsupported state frame: {state_frame}")
    positions = np.asarray(position_m, dtype=np.float64)
    velocities = np.asarray(velocity_mps, dtype=np.float64)
    representation = CartesianRepresentation(
        positions[:, 0] * u.m,
        positions[:, 1] * u.m,
        positions[:, 2] * u.m,
        differentials=CartesianDifferential(
            velocities[:, 0] * u.m / u.s,
            velocities[:, 1] * u.m / u.s,
            velocities[:, 2] * u.m / u.s,
        ),
    )
    state_gcrs = ITRS(representation, obstime=times).transform_to(GCRS(obstime=times))
    return times, state_gcrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T


def calculate_radiants(t0_ns, position_m, velocity_mps, state_frame):
    times, velocity_gcrs_mps = velocities_to_gcrs(t0_ns, position_m, velocity_mps, state_frame)
    fixed_equinox = fixed_ecliptic_equinox(times)
    radiant_unit = -velocity_gcrs_mps / np.linalg.norm(velocity_gcrs_mps, axis=1)[:, None]
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


def write_h5(path, event_id, t0_ns, speed_km_s, rms_total_path_residual_m, n_points, lambda_deg, beta_deg, sun_lambda_deg, sun_centered_lambda_deg, fixed_equinox_iso, state_frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["input_h5"] = INPUT_H5
        h.attrs["coordinate_frame"] = "GeocentricTrueEcliptic"
        h.attrs["input_state_frame"] = state_frame
        h.attrs["fixed_ecliptic_equinox_utc"] = fixed_equinox_iso
        h.attrs["longitude_convention"] = "lambda_radiant - lambda_sun wrapped to [0, 360) deg"
        h.attrs["radiant_definition"] = "apparent radiant direction = -inertial trajectory velocity"
        h["event_id"] = event_id.astype(string_dtype)
        h["t0_ns"] = t0_ns
        h["speed_km_s"] = speed_km_s
        h["rms_total_path_residual_m"] = rms_total_path_residual_m
        h["n_points"] = n_points
        h["lambda_ecliptic_deg"] = lambda_deg
        h["beta_ecliptic_deg"] = beta_deg
        h["sun_lambda_ecliptic_deg"] = sun_lambda_deg
        h["lambda_minus_sun_deg"] = sun_centered_lambda_deg


def observation_times(t0_ns):
    t0 = Time(float(np.nanmin(t0_ns)) / 1e9, format="unix", scale="utc")
    t1 = Time(float(np.nanmax(t0_ns)) / 1e9, format="unix", scale="utc")
    duration_h = (t1 - t0).to_value(u.hour)
    n_interval = max(1, int(np.ceil(duration_h * 60.0 / VISIBILITY_TIME_STEP_MIN)))
    midpoint_h = (np.arange(n_interval, dtype=np.float64) + 0.5) * duration_h / n_interval
    sample_times = t0 + midpoint_h * u.hour
    return t0, t1, sample_times


def local_mean_solar_time(utc_time):
    return utc_time + (float(sc.lon0[0]) / 15.0) * u.hour


def format_time_interval(t0, t1):
    return (
        t0.utc.strftime("%Y-%m-%d %H:%M:%S"),
        t1.utc.strftime("%Y-%m-%d %H:%M:%S"),
        local_mean_solar_time(t0).utc.strftime("%Y-%m-%d %H:%M:%S"),
        local_mean_solar_time(t1).utc.strftime("%Y-%m-%d %H:%M:%S"),
    )


def write_interval_tex(path, t0, t1):
    utc0, utc1, lst0, lst1 = format_time_interval(t0, t1)
    duration_h = (t1 - t0).to_value(u.hour)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by plot_sun_centered_ecliptic_radiants.py\n")
        fh.write("\\begin{tabular}{ll}\\toprule\n")
        fh.write("Quantity & Value \\\\\\midrule\n")
        fh.write(f"UTC measurement interval & {utc0}--{utc1} UTC \\\\\n")
        fh.write(f"Sanya mean local solar time interval & {lst0}--{lst1} \\\\\n")
        fh.write(f"Duration & {duration_h:.2f} h \\\\\n")
        fh.write(f"Sanya longitude used for solar time & {float(sc.lon0[0]):.4f}$^\\circ$E \\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")


def elevation_detection_efficiency(elevation_deg, gamma=RADIANT_ALTITUDE_GAMMA):
    elevation_deg = np.asarray(elevation_deg, dtype=np.float64)
    efficiency = np.sin(np.deg2rad(np.clip(elevation_deg, 0.0, 90.0))) ** gamma
    efficiency[elevation_deg <= 0.0] = 0.0
    return efficiency


def radiant_visibility_grid(sample_times, fixed_equinox_iso, n_lon=145, n_lat=73):
    plot_lon_deg = np.linspace(-180.0, 180.0, n_lon)
    beta_deg = np.linspace(-90.0, 90.0, n_lat)
    plot_lon_mesh, beta_mesh = np.meshgrid(plot_lon_deg, beta_deg)
    lambda_minus_sun = plot_longitude_to_sun_centered_deg(plot_lon_mesh)
    visible_counts = np.zeros_like(plot_lon_mesh, dtype=np.float64)
    effective_counts = np.zeros_like(plot_lon_mesh, dtype=np.float64)

    location = EarthLocation(lat=float(sc.lat0[0]) * u.deg, lon=float(sc.lon0[0]) * u.deg, height=float(sc.alt0[0]) * u.km)
    fixed_equinox = Time(fixed_equinox_iso, format="isot", scale="utc")
    ecliptic_frame = GeocentricTrueEcliptic(obstime=sample_times, equinox=fixed_equinox)
    sun_ecl = get_sun(sample_times).transform_to(ecliptic_frame)

    flat_beta = beta_mesh.ravel()
    flat_lambda_minus_sun = lambda_minus_sun.ravel()
    for time, sun_lon in zip(sample_times, sun_ecl.lon.to_value(u.deg)):
        lon_deg = wrap360(flat_lambda_minus_sun + sun_lon)
        coord = SkyCoord(
            lon=lon_deg * u.deg,
            lat=flat_beta * u.deg,
            frame=GeocentricTrueEcliptic(obstime=time, equinox=fixed_equinox),
        )
        alt = coord.transform_to(AltAz(obstime=time, location=location)).alt.to_value(u.deg)
        alt_grid = alt.reshape(plot_lon_mesh.shape)
        visible_counts += (alt_grid > 0.0).astype(np.float64)
        effective_counts += elevation_detection_efficiency(alt_grid)

    if len(sample_times) > 1:
        dt_h = (sample_times[1] - sample_times[0]).to_value(u.hour)
    else:
        dt_h = VISIBILITY_TIME_STEP_MIN / 60.0
    visibility_hours = visible_counts * dt_h
    effective_hours = effective_counts * dt_h
    return plot_lon_mesh, beta_mesh, visibility_hours, effective_hours


def add_visibility_overlay(ax, plot_lon_mesh, beta_mesh, visibility_hours, effective_hours):
    x_rad = np.deg2rad(plot_lon_mesh)
    y_rad = np.deg2rad(beta_mesh)
    zero_visible = np.ma.masked_where(visibility_hours > 0.0, np.ones_like(visibility_hours))
    ax.contourf(x_rad, y_rad, zero_visible, levels=[0.5, 1.5], colors=["0.86"], alpha=1.0, zorder=0)
    max_hours = float(np.nanmax(effective_hours))
    if max_hours >= 1.0:
        levels = np.arange(1.0, np.floor(max_hours) + 1.0, 1.0)
        contours = ax.contour(x_rad, y_rad, effective_hours, levels=levels, colors="0.35", linewidths=0.65, alpha=0.75, zorder=1)
        ax.clabel(contours, fmt=lambda value: f"{value:.0f} eff. h", fontsize=7, inline=True)


def plot_radiants(lambda_minus_sun_deg, beta_deg, speed_km_s, visibility):
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
    add_visibility_overlay(ax, *visibility)
    add_source_markers(ax)
    sc = ax.scatter(
        x_rad,
        y_rad,
        c=speed_km_s,
        s=24,
        cmap="turbo",
        alpha=0.76,
        linewidths=0.25,
        edgecolors="white",
        zorder=3,
    )
    ax.grid(True, alpha=0.45)
    ax.set_xlabel(r"Sun-centered ecliptic longitude, $\lambda-\lambda_\odot$", labelpad=12)
    ax.set_ylabel(r"Ecliptic latitude, $\beta$", labelpad=12)
    draw_shifted_longitude_labels(ax)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.15, fraction=0.055)
    cb.set_label(r"Fitted geocentric velocity, $v_g$ (km s$^{-1}$)")
    cb.ax.xaxis.labelpad = 8
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    os.makedirs(os.path.dirname(PAPER_OUTPUT_PNG), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_PNG, dpi=240, bbox_inches="tight")
    fig.savefig(PAPER_OUTPUT_PDF, bbox_inches="tight")
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
    add_source_markers(ax, labelled=True)
    sc = ax.scatter(
        x_rad,
        y_rad,
        c=speed_km_s,
        s=24,
        cmap="turbo",
        alpha=0.68,
        linewidths=0.25,
        edgecolors="white",
        zorder=3,
        label="All fitted radiants",
    )

    ax.grid(True, alpha=0.45)
    ax.set_xlabel(r"Sun-centered ecliptic longitude, $\lambda-\lambda_\odot$", labelpad=12)
    ax.set_ylabel(r"Ecliptic latitude, $\beta$", labelpad=12)
    draw_shifted_longitude_labels(ax)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.15, fraction=0.055)
    cb.set_label(r"Fitted geocentric velocity, $v_g$ (km s$^{-1}$)")
    cb.ax.xaxis.labelpad = 8
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.03), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    event_id, t0_ns, position_m, velocity_mps, speed_km_s, rms_total_path_residual_m, n_points, state_frame = load_fits(INPUT_H5)
    lambda_deg, beta_deg, sun_lambda_deg, sun_centered_lambda_deg, fixed_equinox_iso = calculate_radiants(t0_ns, position_m, velocity_mps, state_frame)
    t0, t1, sample_times = observation_times(t0_ns)
    visibility = radiant_visibility_grid(sample_times, fixed_equinox_iso)
    write_interval_tex(PAPER_INTERVAL_TEX, t0, t1)
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
        state_frame,
    )
    plot_radiants(sun_centered_lambda_deg, beta_deg, speed_km_s, visibility)
    plot_radiant_diagnostic(sun_centered_lambda_deg, beta_deg, speed_km_s)
    print(f"radiants: {len(event_id)}")
    print(f"fixed ecliptic equinox UTC: {fixed_equinox_iso}")
    print(f"measurement interval UTC: {format_time_interval(t0, t1)[0]} to {format_time_interval(t0, t1)[1]}")
    print(f"measurement interval local solar: {format_time_interval(t0, t1)[2]} to {format_time_interval(t0, t1)[3]}")
    print(f"lambda-lambda_sun deg [0,360) median/range: {np.nanmedian(sun_centered_lambda_deg):.2f} / {np.nanmin(sun_centered_lambda_deg):.2f} to {np.nanmax(sun_centered_lambda_deg):.2f}")
    print(f"beta deg median/range: {np.nanmedian(beta_deg):.2f} / {np.nanmin(beta_deg):.2f} to {np.nanmax(beta_deg):.2f}")
    print(OUTPUT_H5)
    print(OUTPUT_PNG)
    print(DIAGNOSTIC_OUTPUT_PNG)
    print(PAPER_OUTPUT_PNG)
    print(PAPER_OUTPUT_PDF)
    print(PAPER_INTERVAL_TEX)


if __name__ == "__main__":
    main()
