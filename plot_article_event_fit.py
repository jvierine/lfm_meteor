import argparse
import os
import shutil

import astropy.units as u
import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_memo09_antenna_gain_patterns as gain_model

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI is None:
    COMM = None
    RANK = 0
    SIZE = 1
else:
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
DEFAULT_EVENT_ID = "tri_0134_1713850083054349899"
OUTPUT_BASE = "results/article_event_fit"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
SITE_LABELS = ("Sanya", "Danzhou", "Wenchang")
COMMON_VOLUME_LAT_DEG = 18.567821
COMMON_VOLUME_LON_DEG = 109.683719
COMMON_VOLUME_ALT_KM = 94.988
SANYA_TX_BEAMWIDTH_3DB_DEG = 0.9
SANYA_PATTERN_MIN_HALF_SPAN_KM = 4.0


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def choose_event(h, requested):
    event_ids = decode_strings(h["event_id"][:])
    if requested:
        matches = np.flatnonzero(event_ids == requested)
        if len(matches) == 0:
            raise ValueError(f"Event {requested} not found in {INPUT_H5}")
        idx = int(matches[0])
        return idx, event_ids[idx]

    n_points = np.asarray(h["n_points"][:], dtype=float)
    weighted_rms = np.asarray(h["weighted_rms"][:], dtype=float)
    rms = np.asarray(h["rms_total_path_residual_m"][:], dtype=float)
    score = np.abs(weighted_rms - 1.0) + 0.01 * np.abs(rms - np.nanmedian(rms))
    score[n_points < 20] = np.inf
    idx = int(np.nanargmin(score))
    return idx, event_ids[idx]


def ecef_to_gcrs(points_ecef_m, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    itrs = ITRS(
        CartesianRepresentation(
            points_ecef_m[:, 0] * u.m,
            points_ecef_m[:, 1] * u.m,
            points_ecef_m[:, 2] * u.m,
        ),
        obstime=obstime,
    )
    return itrs.transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m).T


def triangulate_points(measured_total_paths_m):
    x0 = gfit.initial_guess(15.0, 75.0, float(np.nanmedian(measured_total_paths_m[:, 0]) / 2e3))
    points = []
    for measured in measured_total_paths_m:
        point = gfit.solve_position_from_total_paths_m(measured, x0)
        points.append(point)
        x0 = point
    return np.asarray(points, dtype=np.float64)


def lfm_corrected_point_solutions(measured_total_paths_m, x_itrs_m, v_itrs_mps):
    _paths, path_rates_mps = gfit.link_total_paths_and_rates_m(
        x_itrs_m,
        v_itrs_mps,
        gfit.LINK_TX_POSITIONS_M,
        gfit.LINK_RX_POSITIONS_M,
    )
    corrected_paths_m = measured_total_paths_m - gfit.lfm_total_path_bias_m(path_rates_mps)
    return triangulate_points(corrected_paths_m)


def path_jacobian_itrs(point_itrs_m):
    rows = []
    for tx_m, rx_m in zip(gfit.LINK_TX_POSITIONS_M, gfit.LINK_RX_POSITIONS_M):
        tx_vec = point_itrs_m - tx_m
        rx_vec = point_itrs_m - rx_m
        rows.append(tx_vec / np.linalg.norm(tx_vec) + rx_vec / np.linalg.norm(rx_vec))
    return np.asarray(rows, dtype=np.float64)


def itrs_to_gcrs_jacobian(point_itrs_m, time_ns, step_m=1.0):
    base = np.asarray(point_itrs_m, dtype=np.float64)
    samples = []
    for dim in range(3):
        plus = base.copy()
        minus = base.copy()
        plus[dim] += step_m
        minus[dim] -= step_m
        gcrs = ecef_to_gcrs(np.vstack([plus, minus]), np.array([time_ns, time_ns], dtype=np.int64))
        samples.append((gcrs[0] - gcrs[1]) / (2.0 * step_m))
    return np.column_stack(samples)


def position_covariances_gcrs(points_itrs_m, times_ns, sigma_m):
    covariances = []
    for point, time_ns, sigma_row in zip(points_itrs_m, times_ns, sigma_m):
        hmat = path_jacobian_itrs(point)
        weights = np.diag(1.0 / np.maximum(np.asarray(sigma_row, dtype=float), 1e-6) ** 2)
        cov_itrs = np.linalg.pinv(hmat.T @ weights @ hmat)
        jac = itrs_to_gcrs_jacobian(point, int(time_ns))
        covariances.append(jac @ cov_itrs @ jac.T)
    return np.asarray(covariances, dtype=np.float64)


def unit_vector(vector):
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Cannot normalize zero or non-finite vector")
    return np.asarray(vector, dtype=np.float64) / norm


def event_axes(fit_gcrs_m, velocity_gcrs_mps):
    along = unit_vector(np.nanmean(velocity_gcrs_mps, axis=0))
    radial = unit_vector(np.nanmean(fit_gcrs_m, axis=0))
    vertical_plane = radial - np.dot(radial, along) * along
    if np.linalg.norm(vertical_plane) < 1e-6:
        vertical_plane = np.array([0.0, 0.0, 1.0]) - along[2] * along
    cross = unit_vector(vertical_plane)
    return along, cross


def projected_sigma(covariances, axis):
    return np.sqrt(np.maximum(np.einsum("i,nij,j->n", axis, covariances, axis), 0.0))


def fit_band_sigma(t_rel_s, param_covariance, axis):
    cov6 = np.asarray(param_covariance[:6, :6], dtype=np.float64)
    sigma = []
    for t_s in np.asarray(t_rel_s, dtype=float):
        grad = np.concatenate([axis, t_s * axis])
        sigma.append(np.sqrt(max(float(grad @ cov6 @ grad), 0.0)))
    return np.asarray(sigma, dtype=np.float64)


def velocity_band_sigma(param_covariance, axis, n_samples):
    cov6 = np.asarray(param_covariance[:6, :6], dtype=np.float64)
    grad = np.concatenate([np.zeros(3, dtype=np.float64), axis])
    sigma = np.sqrt(max(float(grad @ cov6 @ grad), 0.0))
    return np.full(int(n_samples), sigma, dtype=np.float64)


def enu_basis(lat_deg, lon_deg):
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=np.float64)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    return east, north, up


def horizontal_offsets_km(points_ecef_m, origin_ecef_m, lat_deg, lon_deg):
    east, north, _up = enu_basis(lat_deg, lon_deg)
    rel = np.asarray(points_ecef_m, dtype=np.float64) - np.asarray(origin_ecef_m, dtype=np.float64)[None, :]
    east_km = (rel @ east) / 1e3
    north_km = (rel @ north) / 1e3
    return east_km, north_km


def ecef_unit_to_enu(unit_ecef, lat_deg, lon_deg):
    east, north, up = enu_basis(lat_deg, lon_deg)
    vector = np.asarray(unit_ecef, dtype=np.float64)
    return np.stack([vector @ east, vector @ north, vector @ up], axis=-1)


def sanya_relative_gain_db_at_offsets(east_km, north_km, origin_ecef_m):
    east_axis, north_axis, _up_axis = enu_basis(COMMON_VOLUME_LAT_DEG, COMMON_VOLUME_LON_DEG)
    points = (
        np.asarray(origin_ecef_m, dtype=np.float64)[None, None, :]
        + np.asarray(east_km, dtype=np.float64)[..., None] * 1e3 * east_axis[None, None, :]
        + np.asarray(north_km, dtype=np.float64)[..., None] * 1e3 * north_axis[None, None, :]
    )
    los_ecef = points - gfit.LINK_TX_POSITIONS_M[0][None, None, :]
    los_ecef = los_ecef / np.linalg.norm(los_ecef, axis=-1, keepdims=True)
    san_lat, san_lon, _san_alt = jcoord.ecef2geodetic(*gfit.LINK_TX_POSITIONS_M[0])
    los_enu = ecef_unit_to_enu(los_ecef, san_lat, san_lon)

    site = gain_model.SITES[0]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    _normal, tilt_axis, panel_cross_axis = gain_model.panel_axes(site)
    power = gain_model.aperture_power(
        los_enu,
        pointing,
        tilt_axis,
        panel_cross_axis,
        site.dim_tilt_plane_m,
        site.dim_cross_tilt_m,
    )
    return 10.0 * np.log10(np.maximum(power, 1e-10))


def retained_sanya_snr_db(group, n_points):
    snr = np.asarray(group["snr_db"][:], dtype=np.float64)
    if snr.ndim == 2:
        snr = snr[:, 0]
    if len(snr) == n_points:
        return snr
    if len(snr) > n_points:
        return snr[:n_points]
    out = np.full(n_points, np.nan, dtype=np.float64)
    out[: len(snr)] = snr
    fill = np.nanmedian(snr) if np.any(np.isfinite(snr)) else 0.0
    out[~np.isfinite(out)] = fill
    return out


def b_uncertainty(b_drag, log10_b_std):
    if not np.isfinite(log10_b_std):
        return np.nan
    return np.log(10.0) * float(b_drag) * float(log10_b_std)


def fit_annotation_text(h, group, idx):
    model = str(h.attrs.get("trajectory_model_package", ""))
    if "initial_radius_m" in h and "initial_mass_kg" in h:
        radius_um = float(h["initial_radius_m"][idx]) * 1e6
        mass_kg = float(h["initial_mass_kg"][idx])
        log10_radius_std = np.nan
        if "log10_radius_std" in h:
            log10_radius_std = float(h["log10_radius_std"][idx])
        sigma_radius_um = np.nan
        if np.isfinite(log10_radius_std):
            sigma_radius_um = np.log(10.0) * radius_um * log10_radius_std
        if np.isfinite(sigma_radius_um):
            return f"$r_0 = {radius_um:.2g} \\pm {sigma_radius_um:.1g}$ $\\mu$m\n$m_0 = {mass_kg:.2g}$ kg"
        return f"$r_0 = {radius_um:.2g}$ $\\mu$m\n$m_0 = {mass_kg:.2g}$ kg"
    if "radius_m" in group and "mass_kg" in group:
        radius_um = float(group["radius_m"][0]) * 1e6
        mass_kg = float(group["mass_kg"][0])
        return f"$r_0 = {radius_um:.2g}$ $\\mu$m\n$m_0 = {mass_kg:.2g}$ kg"
    if "b_drag_m2_per_kg" in h:
        b_drag = float(h["b_drag_m2_per_kg"][idx])
        log10_b_std = float(h["log10_b_std"][idx]) if "log10_b_std" in h else np.nan
        sigma_b = b_uncertainty(b_drag, log10_b_std)
        b_text = f"{b_drag:.2g}"
        if np.isfinite(sigma_b):
            b_text = f"{b_drag:.2g} \\pm {sigma_b:.1g}"
        return f"$B = {b_text}$ m$^2$ kg$^{{-1}}$"
    if "ceplecha" in model.lower():
        return "Ceplecha drag--ablation fit"
    return "weighted trajectory fit"


def retained_sigma_m(group, residuals_m):
    sigma_m = group["sigma_m"][:]
    if sigma_m.shape == residuals_m.shape:
        return sigma_m
    normalized = group["normalized_residuals"][:]
    sigma = np.full_like(residuals_m, np.nan, dtype=np.float64)
    good = np.isfinite(normalized) & (np.abs(normalized) > 1e-12)
    sigma[good] = np.abs(residuals_m[good] / normalized[good])
    fallback = np.nanmedian(sigma)
    if not np.isfinite(fallback):
        fallback = 1.0
    sigma[~np.isfinite(sigma)] = fallback
    return sigma


def plot_event(h, idx, event_id, output_base):
    g = h["points"][event_id]
    time_ns = g["time_ns"][:]
    t_rel_s = g["t_rel_s"][:]
    measured_total_paths_m = g["measured_total_paths_m"][:]
    residuals_m = g["residuals_m"][:]
    sigma_m = retained_sigma_m(g, residuals_m)
    sanya_snr_db = retained_sanya_snr_db(g, len(time_ns))
    fit_gcrs_m = g["x_gcrs_m"][:]
    fit_v_gcrs_mps = g["v_gcrs_mps"][:]
    fit_itrs_m = g["x_itrs_m"][:]
    fit_alt_km = g["alt_km"][:]
    param_covariance = g["parameter_covariance"][:]
    has_all_pulses = "all_time_ns" in g and "all_keep_rows" in g and "all_x_itrs_m" in g
    if has_all_pulses:
        all_time_ns = g["all_time_ns"][:]
        all_t_rel_s = g["all_t_rel_s"][:]
        all_measured_total_paths_m = g["all_measured_total_paths_m"][:]
        all_keep_rows = np.asarray(g["all_keep_rows"][:], dtype=bool)
        all_sigma_m = g["all_sigma_m"][:] if "all_sigma_m" in g else np.full_like(all_measured_total_paths_m, np.nan)
        all_fit_gcrs_m = g["all_x_gcrs_m"][:]
        all_fit_v_gcrs_mps = g["all_v_gcrs_mps"][:]
        all_fit_itrs_m = g["all_x_itrs_m"][:]
        all_fit_v_itrs_mps = g["all_v_itrs_mps"][:]
        all_fit_alt_km = g["all_alt_km"][:]
        all_sanya_snr_db = np.asarray(g["all_snr_db"][:], dtype=np.float64)
        if all_sanya_snr_db.ndim == 2:
            all_sanya_snr_db = all_sanya_snr_db[:, 0]
    else:
        all_time_ns = time_ns
        all_t_rel_s = t_rel_s
        all_measured_total_paths_m = measured_total_paths_m
        all_keep_rows = np.ones(len(time_ns), dtype=bool)
        all_sigma_m = sigma_m
        all_fit_gcrs_m = fit_gcrs_m
        all_fit_v_gcrs_mps = fit_v_gcrs_mps
        all_fit_itrs_m = fit_itrs_m
        all_fit_v_itrs_mps = g["v_itrs_mps"][:]
        all_fit_alt_km = fit_alt_km
        all_sanya_snr_db = sanya_snr_db

    measured_itrs_m = lfm_corrected_point_solutions(measured_total_paths_m, fit_itrs_m, g["v_itrs_mps"][:])
    measured_gcrs_m = ecef_to_gcrs(measured_itrs_m, time_ns)
    measured_alt_km = np.asarray([jcoord.ecef2geodetic(*p)[2] / 1e3 for p in measured_itrs_m], dtype=np.float64)
    pos_cov_gcrs = position_covariances_gcrs(measured_itrs_m, time_ns, sigma_m)
    all_measured_itrs_m = lfm_corrected_point_solutions(
        all_measured_total_paths_m,
        all_fit_itrs_m,
        all_fit_v_itrs_mps,
    )
    all_measured_gcrs_m = ecef_to_gcrs(all_measured_itrs_m, all_time_ns)
    all_measured_alt_km = np.asarray([jcoord.ecef2geodetic(*p)[2] / 1e3 for p in all_measured_itrs_m], dtype=np.float64)
    all_pos_cov_gcrs = position_covariances_gcrs(all_measured_itrs_m, all_time_ns, all_sigma_m)

    along_axis, cross_axis = event_axes(fit_gcrs_m, fit_v_gcrs_mps)
    origin = fit_gcrs_m[0]
    fit_along_km = ((fit_gcrs_m - origin) @ along_axis) / 1e3
    fit_cross_km = ((fit_gcrs_m - origin) @ cross_axis) / 1e3
    meas_along_km = ((measured_gcrs_m - origin) @ along_axis) / 1e3
    meas_cross_km = ((measured_gcrs_m - origin) @ cross_axis) / 1e3

    meas_along_sigma_km = projected_sigma(pos_cov_gcrs, along_axis) / 1e3
    meas_cross_sigma_km = projected_sigma(pos_cov_gcrs, cross_axis) / 1e3
    radial_axes = np.asarray([unit_vector(p) for p in measured_gcrs_m])
    meas_alt_sigma_km = np.asarray(
        [np.sqrt(max(float(a @ c @ a), 0.0)) / 1e3 for a, c in zip(radial_axes, pos_cov_gcrs)],
        dtype=np.float64,
    )

    fit_along_95_km = 1.96 * fit_band_sigma(t_rel_s, param_covariance, along_axis) / 1e3
    fit_cross_95_km = 1.96 * fit_band_sigma(t_rel_s, param_covariance, cross_axis) / 1e3
    fit_alt_95_km = np.asarray(
        [1.96 * fit_band_sigma([t_s], param_covariance, unit_vector(p))[0] / 1e3 for t_s, p in zip(t_rel_s, fit_gcrs_m)],
        dtype=np.float64,
    )

    along_residual_m = (meas_along_km - fit_along_km) * 1e3
    along_sigma_m = meas_along_sigma_km * 1e3
    all_fit_along_km = ((all_fit_gcrs_m - origin) @ along_axis) / 1e3
    all_fit_cross_km = ((all_fit_gcrs_m - origin) @ cross_axis) / 1e3
    all_meas_along_km = ((all_measured_gcrs_m - origin) @ along_axis) / 1e3
    all_meas_cross_km = ((all_measured_gcrs_m - origin) @ cross_axis) / 1e3
    all_meas_along_sigma_km = projected_sigma(all_pos_cov_gcrs, along_axis) / 1e3
    all_meas_cross_sigma_km = projected_sigma(all_pos_cov_gcrs, cross_axis) / 1e3
    all_radial_axes = np.asarray([unit_vector(p) for p in all_measured_gcrs_m])
    all_meas_alt_sigma_km = np.asarray(
        [np.sqrt(max(float(a @ c @ a), 0.0)) / 1e3 for a, c in zip(all_radial_axes, all_pos_cov_gcrs)],
        dtype=np.float64,
    )
    all_along_residual_m = (all_meas_along_km - all_fit_along_km) * 1e3
    all_along_sigma_m = all_meas_along_sigma_km * 1e3
    all_cross_residual_m = (all_meas_cross_km - all_fit_cross_km) * 1e3
    all_cross_sigma_m = all_meas_cross_sigma_km * 1e3
    along_fit_95_m = fit_along_95_km * 1e3
    along_speed_km_s = (fit_v_gcrs_mps @ along_axis) / 1e3
    along_speed_95_km_s = 1.96 * velocity_band_sigma(param_covariance, along_axis, len(t_rel_s)) / 1e3
    beam_center_ecef_m = np.asarray(
        jcoord.geodetic2ecef(COMMON_VOLUME_LAT_DEG, COMMON_VOLUME_LON_DEG, COMMON_VOLUME_ALT_KM * 1e3),
        dtype=np.float64,
    )
    measured_east_km, measured_north_km = horizontal_offsets_km(
        measured_itrs_m,
        beam_center_ecef_m,
        COMMON_VOLUME_LAT_DEG,
        COMMON_VOLUME_LON_DEG,
    )
    all_measured_east_km, all_measured_north_km = horizontal_offsets_km(
        all_measured_itrs_m,
        beam_center_ecef_m,
        COMMON_VOLUME_LAT_DEG,
        COMMON_VOLUME_LON_DEG,
    )
    fit_east_km, fit_north_km = horizontal_offsets_km(
        fit_itrs_m,
        beam_center_ecef_m,
        COMMON_VOLUME_LAT_DEG,
        COMMON_VOLUME_LON_DEG,
    )
    beam_radius_km = (
        np.linalg.norm(beam_center_ecef_m - gfit.LINK_TX_POSITIONS_M[0])
        * np.tan(np.deg2rad(0.5 * SANYA_TX_BEAMWIDTH_3DB_DEG))
        / 1e3
    )

    rms = float(h["rms_total_path_residual_m"][idx])
    weighted_rms = float(h["weighted_rms"][idx])
    start_speed = float(h["start_speed_km_s"][idx])
    end_speed = float(h["end_speed_km_s"][idx])
    start_alt = float(h["start_alt_km"][idx])
    end_alt = float(h["end_alt_km"][idx])
    n_points = int(h["n_points"][idx])
    duration_s = float(t_rel_s[-1] - t_rel_s[0]) if len(t_rel_s) else np.nan
    sigma_floor_m = float(h.attrs["sigma_floor_m"])
    sigma_0_m = float(h.attrs["sigma_0_m"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), constrained_layout=True)
    measured_color = "0.15"
    fit_color = "#1b7837"
    band_color = "#a6dba0"
    residual_color = "#2166ac"
    expected_color = "#b2182b"

    ax_map = axes[0, 0]
    rejected = ~all_keep_rows
    if np.any(rejected):
        ax_map.scatter(
            all_measured_east_km[rejected],
            all_measured_north_km[rejected],
            c="0.72",
            s=22,
            edgecolors="none",
            linewidths=0.0,
            label="rejected measurement",
            zorder=2.5,
        )
    snr_scatter = ax_map.scatter(
        all_measured_east_km[all_keep_rows],
        all_measured_north_km[all_keep_rows],
        c=all_sanya_snr_db[all_keep_rows],
        s=22,
        cmap="viridis",
        edgecolors="none",
        linewidths=0.0,
        zorder=3,
    )
    ax_map.plot(fit_east_km, fit_north_km, color=fit_color, lw=1.8, label="_nolegend_", zorder=2)
    padding_km = 0.15
    all_east = np.concatenate([measured_east_km, fit_east_km, [-beam_radius_km, beam_radius_km]])
    all_north = np.concatenate([measured_north_km, fit_north_km, [-beam_radius_km, beam_radius_km]])
    xmid = 0.5 * (np.nanmin(all_east) + np.nanmax(all_east))
    ymid = 0.5 * (np.nanmin(all_north) + np.nanmax(all_north))
    half_span = 0.5 * max(np.nanmax(all_east) - np.nanmin(all_east), np.nanmax(all_north) - np.nanmin(all_north))
    half_span = max(half_span + padding_km, SANYA_PATTERN_MIN_HALF_SPAN_KM)
    pattern_grid = np.linspace(-half_span, half_span, 401)
    pattern_east_km, pattern_north_km = np.meshgrid(pattern_grid, pattern_grid)
    relative_gain_db = sanya_relative_gain_db_at_offsets(pattern_east_km, pattern_north_km, beam_center_ecef_m)
    contours = ax_map.contour(
        pattern_east_km,
        pattern_north_km,
        relative_gain_db,
        levels=[-30.0, -20.0, -13.3, -3.0],
        colors=["0.65", "0.45", "0.25", "0.10"],
        linewidths=[0.7, 0.8, 0.95, 1.2],
        linestyles=[":", "--", "-.", "-"],
        zorder=1,
    )
    ax_map.clabel(contours, fmt={-30.0: "-30", -20.0: "-20", -13.3: "-13", -3.0: "-3 dB"}, fontsize=7)
    ax_map.set_xlim(xmid - half_span, xmid + half_span)
    ax_map.set_ylim(ymid - half_span, ymid + half_span)
    ax_map.set_xlabel("East (km)")
    ax_map.set_ylabel("North (km)")
    ax_map.grid(True, color="0.88", lw=0.7)
    cbar = fig.colorbar(snr_scatter, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("Sanya SNR (dB)")

    panels = [
        (
            axes[0, 1],
            meas_cross_km * 1e3,
            meas_cross_sigma_km * 1e3,
            fit_cross_km * 1e3,
            fit_cross_95_km * 1e3,
            "Cross-track displacement (m)",
            residual_color,
            "position minus fit ±1σ",
        ),
        (
            axes[1, 0],
            measured_alt_km,
            meas_alt_sigma_km,
            fit_alt_km,
            fit_alt_95_km,
            "Altitude (km)",
            measured_color,
            "position solution ±1σ",
        ),
    ]
    for ax, measured_y, measured_sigma, fit_y, fit_95, ylabel, point_color, point_label in panels:
        ax.fill_between(t_rel_s, fit_y - fit_95, fit_y + fit_95, color=band_color, alpha=0.55, lw=0, label="95% fit band")
        ax.plot(t_rel_s, fit_y, color=fit_color, lw=1.8, label="weighted fit")
        if np.any(rejected):
            if ylabel == "Cross-track displacement (m)":
                rejected_y = all_meas_cross_km[rejected] * 1e3
                rejected_sigma = all_cross_sigma_m[rejected]
            else:
                rejected_y = all_measured_alt_km[rejected]
                rejected_sigma = all_meas_alt_sigma_km[rejected]
            ax.errorbar(
                all_t_rel_s[rejected],
                rejected_y,
                yerr=rejected_sigma,
                fmt="o",
                ms=3.0,
                lw=0.7,
                capsize=1.3,
                color="0.65",
                ecolor="0.72",
                alpha=0.85,
                label="rejected measurement",
                zorder=2,
            )
        ax.errorbar(
            t_rel_s,
            measured_y,
            yerr=measured_sigma,
            fmt="o",
            ms=3.0,
            lw=0.8,
            capsize=1.5,
            color=point_color,
            ecolor="0.45",
            label=point_label,
        )
        ax.set_xlabel("Time since first matched pulse (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.88", lw=0.7)

    axes[1, 0].text(
        0.04,
        0.05,
        fit_annotation_text(h, g, idx),
        transform=axes[1, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
    )
    bottom_left_handles, bottom_left_labels = axes[1, 0].get_legend_handles_labels()
    bottom_left_items = [
        (handle, label)
        for handle, label in zip(bottom_left_handles, bottom_left_labels)
        if label in {"95% fit band", "weighted fit"}
    ]
    axes[1, 0].legend(
        [handle for handle, _label in bottom_left_items],
        [label for _handle, label in bottom_left_items],
        loc="upper right",
        frameon=False,
    )
    top_right_handles, top_right_labels = axes[0, 1].get_legend_handles_labels()
    top_right_items = [
        (handle, label)
        for handle, label in zip(top_right_handles, top_right_labels)
        if label in {"position minus fit ±1σ", "rejected measurement"}
    ]
    axes[0, 1].legend(
        [handle for handle, _label in top_right_items],
        [label for _handle, label in top_right_items],
        loc="best",
        frameon=False,
    )

    ax = axes[1, 1]
    ax.fill_between(t_rel_s, -along_fit_95_m, along_fit_95_m, color=band_color, alpha=0.55, lw=0, label="95% fit band")
    if np.any(rejected):
        ax.errorbar(
            all_t_rel_s[rejected],
            all_along_residual_m[rejected],
            yerr=all_along_sigma_m[rejected],
            fmt="o",
            ms=3.0,
            lw=0.7,
            capsize=1.3,
            color="0.65",
            ecolor="0.72",
            alpha=0.85,
            label="rejected measurement",
            zorder=2,
        )
    ax.errorbar(
        t_rel_s,
        along_residual_m,
        yerr=along_sigma_m,
        fmt="o",
        ms=3.0,
        lw=0.8,
        capsize=1.5,
        color=residual_color,
        ecolor="0.45",
        label="position minus fit ±1σ",
    )
    ax.axhline(0.0, color=fit_color, lw=1.4, label="weighted fit")
    ax.set_xlabel("Time since first matched pulse (s)")
    ax.set_ylabel("Along-track displacement (m)")
    ax.grid(True, color="0.88", lw=0.7)

    axr = ax.twinx()
    velocity_color = "#984ea3"
    velocity_band_color = "#cab2d6"
    axr.fill_between(
        t_rel_s,
        along_speed_km_s - along_speed_95_km_s,
        along_speed_km_s + along_speed_95_km_s,
        color=velocity_band_color,
        alpha=0.45,
        lw=0,
        label="velocity 95%",
    )
    axr.plot(t_rel_s, along_speed_km_s, color=velocity_color, lw=1.5, label="along-track speed")
    axr.set_ylabel("Along-track speed (km s$^{-1}$)", color=velocity_color)
    axr.tick_params(axis="y", colors=velocity_color)

    right_handles, right_labels = axr.get_legend_handles_labels()
    keep = [label in {"velocity 95%", "along-track speed"} for label in right_labels]
    handles = [handle for handle, use in zip(right_handles, keep) if use]
    labels = [label for label, use in zip(right_labels, keep) if use]
    ax.legend(handles, labels, loc="lower right", frameon=False)

    utc_label = np.datetime_as_string(np.datetime64(int(time_ns[0]), "ns"), unit="ms").replace("T", " ")
    annotation = f"{utc_label} UTC"
    fig.suptitle(annotation)

    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    png = f"{output_base}_{event_id}.png"
    pdf = f"{output_base}_{event_id}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description="Create an article-style event fit plot with uncertainty.")
    parser.add_argument("--input-h5", default=INPUT_H5)
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--copy-to-article", action="store_true")
    parser.add_argument("--all", action="store_true", help="Plot every fitted tri-static event in the input HDF5 file.")
    parser.add_argument("--limit", type=int, default=None, help="Only plot the first N events, useful for smoke tests.")
    args = parser.parse_args()

    with h5py.File(args.input_h5, "r") as h:
        if args.all:
            event_ids = decode_strings(h["event_id"][:])
            if args.limit is not None:
                event_ids = event_ids[: args.limit]
            written = []
            indexed_event_ids = list(enumerate(event_ids))
            local_indexed_event_ids = indexed_event_ids[RANK::SIZE]
            if RANK == 0:
                print(f"plotting {len(event_ids)} events with MPI ranks={SIZE}", flush=True)
            for local_count, (idx, event_id) in enumerate(local_indexed_event_ids, start=1):
                png, pdf = plot_event(h, idx, event_id, args.output_base)
                written.append((png, pdf))
                print(
                    f"[rank {RANK}] {local_count:04d}/{len(local_indexed_event_ids):04d} "
                    f"global={idx + 1:04d}/{len(event_ids):04d} {event_id} -> {png}",
                    flush=True,
                )
        else:
            if RANK != 0:
                return
            idx, event_id = choose_event(h, args.event_id)
            png, pdf = plot_event(h, idx, event_id, args.output_base)
            written = [(png, pdf)]

    for png, pdf in written:
        print(f"wrote {png}")
        print(f"wrote {pdf}")
    if args.copy_to_article:
        os.makedirs(ARTICLE_FIGURE_DIR, exist_ok=True)
        copied = []
        for png, pdf in written:
            for path in (png, pdf):
                target = os.path.join(ARTICLE_FIGURE_DIR, os.path.basename(path))
                shutil.copy2(path, target)
                copied.append(target)
        for path in copied:
            print(f"copied {path}")


if __name__ == "__main__":
    main()
