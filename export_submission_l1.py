"""Add explicit measurement-derived rates and inferred state vectors to L1.

Consumes verified archive bytes, never changes fits, orbit products, or L0.
The track-rate convention follows plot_all_detection_time_delay.fit_track.
ECEF transforms follow fit_shrinking_voltage.full_products and Model.paths.
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np

from package_submission import sha

STR = h5py.string_dtype("utf-8")


def track_rates(time_ns, half_path_m):
    time_ns = np.asarray(time_ns, dtype=np.int64)
    y = np.asarray(half_path_m, dtype=float)
    valid = np.isfinite(y)
    rate = np.full(y.shape, np.nan)
    error = rate.copy()
    if valid.sum() < 2 or len(np.unique(time_ns[valid])) < 2:
        return rate, error, 0
    t = (time_ns - time_ns[valid][0]).astype(float) * 1e-9
    center = t[valid].mean()
    scale = np.max(abs(t[valid] - center))
    z = (t - center) / scale
    degree = min(2, len(np.unique(time_ns[valid])) - 1)
    design = np.vander(z[valid], degree + 1, increasing=True)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y[valid], rcond=None)
    if rank != degree + 1:
        return rate, error, 0
    derivative = np.zeros((len(y), degree + 1))
    for i in range(1, degree + 1):
        derivative[:, i] = i*z**(i-1)/scale
    rate[valid] = (derivative @ coefficients)[valid]
    dof = valid.sum() - degree - 1
    if dof > 0:
        residual = y[valid] - design @ coefficients
        cov = np.linalg.pinv(design.T @ design) * (residual @ residual / dof)
        error[valid] = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", derivative, cov, derivative), 0))[valid]
    return rate, error, degree


def state_vectors(center, params, frames, position_ecef, speed_m_s):
    v0 = center[3:] + 1000*params[3:6]
    direction = v0/np.linalg.norm(v0)
    vg = speed_m_s[:, None]*direction
    rp, rv = frames[:, 6:15].reshape(-1, 3, 3), frames[:, 24:33].reshape(-1, 3, 3)
    vp, vv = frames[:, 15:24].reshape(-1, 3, 3), frames[:, 33:42].reshape(-1, 3, 3)
    xg = np.linalg.solve(rp, (position_ecef - frames[:, :3] - np.einsum("nij,nj->ni", rv, vg))[..., None])[..., 0]
    ve = frames[:, 3:6] + np.einsum("nij,nj->ni", vp, xg) + np.einsum("nij,nj->ni", vv, vg)
    delta = xg - (center[:3] + 1000*params[:3])
    transverse = delta - np.outer(delta @ direction, direction)
    error = float(np.max(np.linalg.norm(transverse, axis=1)))
    if error > 1e-3:
        raise ValueError(f"Stored positions disagree with fit direction/origin: {error} m")
    return xg, vg, ve, error


def put(group, key, value, units=None, meaning=None):
    a = np.asarray(value)
    kw = dict(compression="gzip", shuffle=True) if a.ndim and a.size else {}
    d = group.create_dataset(key, data=value, **kw)
    if units:
        d.attrs["units"] = units
    if meaning:
        d.attrs["meaning"] = meaning
    return d


def verify_archive(root, component, name):
    path = root / component / name
    with h5py.File(root / component / "checksums.h5") as h:
        assert sha(path) == h.attrs[name], path
    return path


def export(root, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    detection_tar = verify_archive(root, "cuts-core", "detections.tar.gz")
    fit_tar = verify_archive(root, "catalog-core", "fits.tar.gz")
    snr_map = {}
    detections = output / "detection_rates.h5"
    with tarfile.open(detection_tar) as tar:
        blob = tar.extractfile("L1/detections.h5").read()
    with h5py.File(io.BytesIO(blob)) as source, h5py.File(detections, "x") as out:
        out.attrs.update(schema="sanya-l1-detection-rates-v1", source_archive_sha256=sha(detection_tar),
            quantity_type="measurement-derived apparent half-path range rate, not independent Doppler or 3D velocity",
            rate_method="Unweighted degree-2 polynomial of finite measured ranges vs UTC time; degree-1 fallback",
            uncertainty="OLS residual standard error assumes independent residuals; not calibrated total uncertainty",
            delay_doppler_warning="LFM apparent range includes Doppler coupling; range-rate is not an unbiased independent path velocity",
            independent_doppler_status="Unavailable in original detector export; no independent Doppler estimate substituted",
            source_program=Path(__file__).read_text())
        for event, original in source.items():
            src = original["detections"]
            g = out.create_group(event)
            original.copy("detections", g, name="measured")
            for k, v in original.attrs.items():
                g.attrs[k] = v
            times = src["time_ns_utc"][:].astype(np.int64)
            r = src["corrected_range_km"][:]*1000
            rates, errors, degree = track_rates(times, r)
            g.attrs.update(rate_polynomial_degree=degree, fit_uses="All finite detection rows; original selected_for_cut retained separately")
            d = g.create_group("measurement_derived")
            put(d, "apparent_half_path_rate_m_s", rates, "m s-1", "Positive increasing legacy half-path range")
            put(d, "apparent_total_path_rate_m_s", 2*rates, "m s-1", "Twice half-path range rate; not object speed")
            put(d, "half_path_rate_ols_standard_error_m_s", errors, "m s-1")
            put(d, "independent_doppler_velocity_m_s", np.full(len(times), np.nan), "m s-1", "Not identifiable from the exported detector fields")
            put(d, "independent_doppler_available", np.zeros(len(times), dtype=bool))
            put(d, "rate_available", np.isfinite(rates))
            snr_map[event] = dict(zip(times, src["snr_peak_db"][:]))
    del blob
    events = {}
    with tarfile.open(fit_tar, "r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[:2] == ("support", "original_inputs"):
                event, kind = Path(parts[2]).stem, "original"
            elif len(parts) == 4 and parts[:2] == ("L1", "events"):
                event, kind = parts[2], parts[3]
            else:
                continue
            if kind not in ("original", "input.h5", "fit.h5", "physical_states.h5", "plot.h5", "beam_rcs.h5", "rcs_mass.h5"):
                continue
            with h5py.File(io.BytesIO(tar.extractfile(member).read())) as h:
                keys = {"original":("time_ns", "station", "frames"),
                        "input.h5":("center", "time_ns", "station"),
                        "fit.h5":("params",)}.get(kind, tuple(h.keys()))
                # Beam contours are groups, not per-detection measurements.
                fields = {k:h[k][()] for k in keys if isinstance(h[k], h5py.Dataset)}
                events.setdefault(event, {})[kind] = (fields, dict(h.attrs))
    orbit = root / "orbits-refreshed/orbits/orbits.h5"
    with h5py.File(orbit) as h:
        accepted = set(h["event_id"].asstr()[:])
        excluded = dict(zip(h["excluded_event_id"].asstr()[:], h["exclusion_reason"].asstr()[:]))
    states = output / "inferred_states.h5"
    report = []
    with h5py.File(states, "x") as out:
        out.attrs.update(schema="sanya-l1-inferred-states-v1", source_archive_sha256=sha(fit_tar),
            quantity_type="Inferred best-fit nominal-time trajectory states; not independent measured velocities",
            time_convention="Original physical_states nominal transmit-pulse epochs, not retarded scatter time",
            model="Existing direct shrinking-radius fit, fixed GCRS direction; no refit",
            vector_uncertainty_status="Full selected parameter ensembles supplied separately; no vector covariance claimed",
            orbit_selection_sha256=sha(orbit),
            source_program=Path(__file__).read_text())
        for event, parts in sorted(events.items()):
            original, _ = parts["original"]
            inp, attrs = parts["input.h5"]
            fit, _ = parts["fit.h5"]
            physical, _ = parts["physical_states.h5"]
            plot, pattrs = parts["plot.h5"]
            keys = list(zip(physical["station"], physical["time_ns"]))
            indices = {(int(s), int(t)):i for i, (s,t) in enumerate(zip(original["station"], original["time_ns"]))}
            frames = original["frames"][[indices[(int(s),int(t))] for s,t in keys]]
            xg, vg, ve, error = state_vectors(inp["center"], fit["params"], frames,
                                             physical["position_ecef_m"], physical["speed_km_s"]*1000)
            g = out.create_group(event)
            for name, value in physical.items():
                put(g, name, value)
            put(g, "position_gcrs_m", xg, "m")
            put(g, "velocity_gcrs_m_s", vg, "m s-1")
            put(g, "velocity_ecef_m_s", ve, "m s-1")
            retained = set(zip(inp["station"], inp["time_ns"]))
            put(g, "used_in_fit", np.array([k in retained for k in keys]))
            files = str(attrs["source_event_files"]).split(";")
            snr = [snr_map[Path(files[int(s)]).stem].get(int(t), np.nan) for s,t in keys]
            put(g, "measured_detection_snr_db", snr, "dB")
            put(g, "measured_detection_snr_available", np.isfinite(snr))
            beam, _ = parts["beam_rcs.h5"]
            rcs_by_time = dict(zip(beam["time_ns"], beam["rcs_dbsm"]))
            rcs = np.array([rcs_by_time.get(int(t), np.nan) if s == 0 else np.nan for s,t in keys])
            put(g, "calibrated_sanya_rcs_dbsm", rcs, "dB m2", "Sanya-only calibration from existing beam_rcs product; other stations not supplied")
            put(g, "calibrated_sanya_rcs_available", np.isfinite(rcs))
            put(g, "initial_radius_m", pattrs["best_initial_radius_m"], "m")
            put(g, "initial_mass_kg", pattrs["best_initial_mass_kg"], "kg")
            _, mass_attrs = parts["rcs_mass.h5"]
            mg = g.create_group("rcs_mass")
            for name in ("mass_kg", "lo95_scatter_kg", "hi95_scatter_kg"):
                put(mg, name, mass_attrs[name], "kg")
            for name in ("meaning", "uncertainty", "lookup_warning", "lookup_correction"):
                mg.attrs[name] = mass_attrs[name]
            g.attrs.update(max_position_direction_consistency_error_m=error,
                           included_in_orbit_catalog=event in accepted,
                           event_exclusion_reason=excluded.get(event, ""),
                           uncertainty_method=str(pattrs.get("uncertainty_method", "conditional physical-model posterior")),
                           exclusions="used_in_fit identifies point exclusions; event acceptance is supplied by final orbit/review catalogue")
            ug = g.create_group("published_uncertainty")
            for name in ("initial_radius_m_quantiles", "initial_mass_kg_quantiles", "path_lower95_m", "path_upper95_m",
                         "speed_lower95_km_s", "speed_upper95_km_s", "speed_time_s", "time_ns", "station"):
                if name in plot:
                    put(ug, name, plot[name])
            ug.attrs["alignment"] = "Path bands use plot station/time_ns; speed bands use speed_time_s, not physical_states row order"
            ug.attrs["interpretation"] = str(pattrs.get("uncertainty_status", "unspecified"))
            ug.attrs["vector_covariance_available"] = False
            report.append(dict(event=event, points=len(keys), snr_missing=int(np.sum(~np.isfinite(snr))),
                               max_transverse_error_m=error))
        from package_submission import write_rows
        write_rows(out, "export_audit", report)
    with h5py.File(output / "checksums.h5", "x") as h:
        for path in (detections, states):
            h.attrs[path.name] = sha(path)
            h.attrs[path.name + "_bytes"] = path.stat().st_size
    print(json.dumps(dict(detection_events=len(snr_map), fitted_events=len(report),
        state_points=sum(r["points"] for r in report), missing_snr=sum(r["snr_missing"] for r in report),
        maximum_transverse_error_m=max(r["max_transverse_error_m"] for r in report),
        products={p.name:p.stat().st_size for p in (detections, states)})))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("staging", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    export(a.staging, a.output)
