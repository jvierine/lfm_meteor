"""Prepare scientific HDF5 inputs for the November Rustplotlib memo figures."""
import argparse
import hashlib
from pathlib import Path
import h5py
import numpy as np
from analyze import SITES, STATIONS, ENU, C, save
from derived import beam, beam_frame, beam_power, angular_offsets, directions_from_offsets


def retained_ids(root):
    path=root/"association_selection.h5"
    if not path.exists(): return None
    with h5py.File(path) as h:
        if h.attrs["associations_sha256"] != hashlib.sha256((root/"associations.h5").read_bytes()).hexdigest():
            raise ValueError("association selection is stale; rerun overlap.py before preparing figures")
        return {name.decode() if isinstance(name,bytes) else str(name)
                for name,keep in zip(h["event_id"][:],h["accepted"][:]) if keep}


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
        accepted=retained_ids(out.parent)
        if accepted is not None:
            indices=indices[["tri_%06d"%i in accepted for i in range(len(indices))]]
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


def classification(a, e, inc):
    with np.errstate(invalid="ignore", divide="ignore"):
        high = (inc > 75) | (e >= 1)
        tj = 1/a + 2*5.2038**(-1.5)*np.sqrt(np.maximum(a*(1-e*e), 0))*np.cos(np.deg2rad(inc))
        votes = np.array([(tj < .58) | high,
            (np.log(a*(1+e)/(1-e))-1 > 0) | high,
            (a*(1+e) > 4.6) | high,
            (np.sqrt(a**3)*e > 2.5) | high,
            (-.01720209895**2/(2*a) > -5.28e-5) | high])
    valid = np.isfinite(a) & np.isfinite(e) & (a > 0) & (e < 1)
    return np.mean(votes[:, valid]) if valid.any() else np.nan


def event_data(folder):
    from astropy.coordinates import EarthLocation, SkyCoord, GCRS, GeocentricTrueEcliptic, get_sun
    from astropy import units as u
    from astropy.time import Time
    with h5py.File(folder/"fit.h5") as h:
        d = {k: h[k][()] for k in h}
    if not np.isfinite(d["path_residual_m"]).all() or np.median(np.abs(d["residual_beat_hz"])) > 5000:
        return None
    physical, rcs = {}, {}
    if (folder/"physical.h5").exists():
        with h5py.File(folder/"physical.h5") as h:
            physical = {k: h[k][()] for k in h}
            physical["resolved"] = bool(h.attrs.get("mass_resolved_profile",False))
    if (folder/"rcs.h5").exists():
        with h5py.File(folder/"rcs.h5") as h:
            rcs = {k: h[k][()] for k in h}
    position = physical.get("position_ecef_m", d["position_ecef_m"])
    location = EarthLocation.from_geocentric(position[:, 0]*u.m, position[:, 1]*u.m, position[:, 2]*u.m)
    height = location.height.to_value(u.km)
    if not (60 < np.median(height) < 160):
        return None
    v = physical.get("initial_gcrs_velocity_mps", d["initial_gcrs_velocity_mps"])
    epoch = Time(np.longdouble(d["time_ns"].min())/1e9, format="unix", scale="utc")
    sky = SkyCoord(x=-v[0]*u.one, y=-v[1]*u.one, z=-v[2]*u.one,
                   representation_type="cartesian", frame=GCRS(obstime=epoch))
    radiant = sky.transform_to(GeocentricTrueEcliptic(equinox=epoch))
    sun = get_sun(epoch).transform_to(GeocentricTrueEcliptic(equinox=epoch))
    return dict(folder=folder, fit=d, physical=physical, rcs=rcs, height=height,
        longitude=location.lon.deg, latitude=location.lat.deg,
        radiant_lon=(radiant.lon.deg-sun.lon.deg)%360, radiant_lat=radiant.lat.deg,
        speed=np.linalg.norm(v)/1000, sun_lon=sun.lon.deg)


def create_event_input(event):
    d, physical, rcs = event["fit"], event["physical"], event["rcs"]
    folder = event["folder"]
    time, station = d["time_s"], d["station"]
    order = np.argsort(time)
    model = physical.get("model_apparent_path_m",d["model_apparent_path_m"])
    residual_beat=d["residual_beat_hz"]
    if "physical_path_m" in physical:
        residual_beat=d["measured_gate_beat_hz"]-physical["physical_doppler_hz"]+(4e6/200e-6)/C*(d["matched_filter_path_m"]-physical["physical_path_m"])
    sigma = C/(4e6/200e-6)*d["conditional_beat_std_hz"]
    speed = physical.get("speed_mps", d["speed_mps"])/1000
    k = station == 0
    gain_db, gain, distance, east, north = beam(physical.get("position_ecef_m", d["position_ecef_m"])[k])
    fields = dict(time_s=time, station=station.astype(float), seed_apparent_path_m=d["apparent_path_m"],
        modeled_path_m=model, path_lower95_m=model, path_upper95_m=model,
        path_measurement_lower95_m=d["apparent_path_m"]-1.96*sigma,
        path_measurement_upper95_m=d["apparent_path_m"]+1.96*sigma,
        residual_beat_hz=residual_beat, conditional_beat_std_hz=d["conditional_beat_std_hz"],
        speed_time_s=time[order], speed_km_s=speed[order],
        speed_lower95_km_s=speed[order], speed_upper95_km_s=speed[order])
    with h5py.File(folder/"event_plot.h5", "w") as h:
        for key, value in fields.items(): h[key] = value
        h.attrs["compact_article_layout"] = True
        h.attrs["show_model_uncertainty"] = False
        g = h.create_group("beam")
        g["time_s"], g["relative_gain_db"] = time[k], gain_db
        # If absolute calibration is not ready, show the fitted SNR instead.
        values = rcs.get("rcs_dbsm", d["snr_db"][k])
        use_rcs = bool(rcs) and np.any(np.isfinite(values) & (gain_db >= -10))
        g["rcs_dbsm"] = values if use_rcs else d["snr_db"][k]
        g.attrs["echo_quantity"] = "RCS (dBsm)" if use_rcs else "SNR (dB)"
        g["east_m"], g["north_m"], g["height_km"] = east, north, event["height"][k]
        extent = max(1000, np.max(np.abs(np.r_[east, north]))*1.15)
        g.attrs["extent_m"] = extent
        # -3 dB square-aperture main-beam contour at the fitted mean range.
        contour = g.create_group("contours/main")
        angle = np.linspace(0, 2*np.pi, 181)
        from scipy.optimize import brentq
        axis, _, _ = beam_frame()
        radius = np.array([brentq(lambda r: beam_power(directions_from_offsets(
            r*np.cos(a), r*np.sin(a)), axis)-10**(-3/10), 0., .9) for a in angle])
        contour["half_power"] = np.median(distance)*np.sin(np.deg2rad(radius))[:,None]*np.c_[np.cos(angle),np.sin(angle)]
    return str(folder/"event_plot.h5")


def catalogue(root, out, workers):
    import concurrent.futures
    folders = sorted(x.parent for x in (out.parent/"events").glob("tri_*/fit.h5"))
    accepted=retained_ids(out.parent)
    if accepted is not None: folders=[f for f in folders if f.name in accepted]
    if not folders: return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        events = [e for e in pool.map(event_data, folders, chunksize=1) if e is not None]
    if not events: return
    save(out/"figure4.h5", {"residual_m": np.concatenate([e["physical"].get("path_residual_m",e["fit"]["path_residual_m"]) for e in events])})
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        event_inputs = list(pool.map(create_event_input, events, chunksize=1))
    save(out.parent/"event_plot_manifest.h5", {"paths": np.array(event_inputs, dtype="S512")})
    atlas=[]
    for e in events:
        name=e["folder"].name
        when=str(np.datetime64(int(e["fit"]["time_ns"].min()),"ns"))[:23].replace("T"," ")
        caption="Event %s, %s UTC. Apparent paths, speed, residuals, echo evolution and beam-plane trajectory." % (name.replace("_",r"\_"),when)
        if e["physical"].get("resolved",False):
            mantissa,exponent=("%.2e"%e["physical"]["initial_mass_kg"]).split("e")
            caption+=r" The reference-atmosphere dynamic mass is $%s\times10^{%d}$ kg."%(mantissa,int(exponent))
        atlas.append(r"\begin{minipage}{\linewidth}\centering\includegraphics{memos/november2026/events/%s/event.pdf}\captionof{figure}{%s}\end{minipage}\par\clearpage"%(name,caption))
    (out.parent/"event_atlas.tex").write_text("\n".join(atlas)+"\n")
    (out.parent/"event_examples.tex").write_text("\n".join(atlas[:6])+"\n")

    # Height and pointing offsets use geodetic positions, not slant range.
    with h5py.File(root/"20261117_sanya/catalogue.h5") as h:
        ranges = h["event_median_apparent_range_km"][:]
    mono = np.sqrt(6371**2+ranges**2+2*6371*ranges*np.sin(np.deg2rad(75)))-6371
    heights = np.concatenate([e["height"] for e in events])
    east, north = [], []
    for e in events:
        p = e["physical"].get("position_ecef_m", e["fit"]["position_ecef_m"])
        local = (p-STATIONS[0]) @ ENU.T
        dx, dy = angular_offsets(local/np.linalg.norm(local,axis=1)[:,None])
        east.extend(dx)
        north.extend(dy)
    grid = np.linspace(-2.6, 2.6, 105)
    counts, xe, ye = np.histogram2d(east, north, bins=(grid, grid))
    xx, yy = np.meshgrid(grid, grid)
    axis, sky_east, sky_north = beam_frame()
    gain_grid = 10*np.log10(np.maximum(beam_power(directions_from_offsets(xx,yy),axis),1e-30))
    gh = np.linspace(80, 120, 401)
    # Relative coverage from actual beam geometry; no absolute remote-gain claim.
    a, el = np.deg2rad([15,75])
    axis = ENU.T @ np.array([np.sin(a)*np.cos(el),np.cos(a)*np.cos(el),np.sin(el)])
    points = STATIONS[0] + (gh*1000/np.sin(el))[:,None]*axis
    remote = []
    for s in (1,2):
        with h5py.File(root/("20261117_"+SITES[s])/"event_cuts/event_000000.h5") as h:
            pointing=(float(h["pulses/rx_azimuth_deg"][0]),float(h["pulses/rx_elevation_deg"][0]))
        from analyze import enu_matrix, LLH
        matrix = enu_matrix(*LLH[s,:2])
        direction = (points-STATIONS[s]) @ matrix.T
        direction /= np.linalg.norm(direction,axis=1)[:,None]
        az, elev = np.deg2rad(pointing)
        b = np.array([np.sin(az)*np.cos(elev),np.cos(az)*np.cos(elev),np.sin(elev)])
        bore_az,bore_el=np.deg2rad((158.3,70.) if s==1 else (221.9,70.))
        normal=np.array([np.sin(bore_az)*np.cos(bore_el),np.cos(bore_az)*np.cos(bore_el),np.sin(bore_el)])
        tilt=np.array([0.,0.,1.])-normal[2]*normal;tilt/=np.linalg.norm(tilt)
        cross=np.cross(normal,tilt)
        power=np.sinc(32/(C/430e6)*((direction-b)@tilt))**2*np.sinc(24/(C/430e6)*((direction-b)@cross))**2
        remote.append(43+25*np.log10(normal@b)+10*np.log10(np.maximum(power,1e-15)))
    tri_event_height=np.array([np.median(e["height"]) for e in events])
    save(out/"figure3.h5", dict(tristatic_height_km=tri_event_height, monostatic_height_km=mono,
        gain_height_km=gh, gain_Danzhou=remote[0], gain_Wenchang=remote[1],
        counts=counts.T, east_edges=xe, north_edges=ye, gain_grid_deg=grid,gain_db=gain_grid),
        {"containment_level_db": -3., "remote_gain_model": "rectangular aperture; header pointing",
         "beam_coordinates": "orthonormal east/north sky tangent plane; gnomonic angles",
         "beam_axis_enu": axis, "beam_east_enu": sky_east, "beam_north_enu": sky_north})
    event_height = np.array([np.median(e["height"]) for e in events])
    common=np.logical_and.reduce([g>=np.max(g)-3 for g in remote])
    if not common.any(): raise ValueError("recorded remote beams have no shared half-power altitude interval")
    common_lo,common_hi=gh[common].min(),gh[common].max()
    with h5py.File(out/"altitude_correction.h5","w") as h:
        h.attrs.update(monostatic_events=len(mono),tristatic_events=len(events),
            common_lower_km=common_lo,common_upper_km=common_hi,
            monostatic_common_events=np.sum((mono>=common_lo)&(mono<=common_hi)),
            tristatic_common_events=np.sum((event_height>=common_lo)&(event_height<=common_hi)))

    physical = [e for e in events if e["physical"]]
    if physical and any(e["physical"]["resolved"] for e in physical):
        save(out/"figure7.h5", dict(selected=np.array([e["physical"]["resolved"] for e in physical]),
            initial_mass_kg=np.array([e["physical"]["initial_mass_kg"] for e in physical]),
            initial_speed_km_s=np.array([e["speed"] for e in physical]),
            mass_95_lo_kg=np.array([e["physical"]["mass_local95_kg"][0] for e in physical]),
            mass_95_hi_kg=np.array([e["physical"]["mass_local95_kg"][1] for e in physical])),
            {"interval_method":"conditional radius-profile likelihood; reference atmosphere", "bin_width_dex": .48})
    rcs_events = [e for e in events if e["rcs"]]
    pairs = [e for e in rcs_events if e["physical"].get("resolved",False) and np.isfinite(e["rcs"]["integrated_mass_loss_kg"])]
    if rcs_events and pairs:
        with h5py.File(out/"figure8.h5","w") as h:
            g=h.create_group("pulses")
            for target,source in [("rcs_dbsm","rcs_dbsm"),("alt_km","height_km"),("speed_km_s","speed_km_s")]:
                g[target]=np.concatenate([e["rcs"][source][e["rcs"]["relative_gain_db"]>=-3] for e in rcs_events])
            g=h.create_group("mass_comparison")
            for target,values in [
                ("dynamic_mass_kg",[e["physical"]["initial_mass_kg"] for e in pairs]),
                ("rcs_mass_kg",[e["rcs"]["integrated_mass_loss_kg"] for e in pairs]),
                ("dynamic_lo95_kg",[e["physical"]["mass_local95_kg"][0] for e in pairs]),
                ("dynamic_hi95_kg",[e["physical"]["mass_local95_kg"][1] for e in pairs])]: g[target]=values
            # RCS intervals are not invented when calibration/error propagation is pending.
            g["delta_mass_lo95_noise_only_kg"]=g["rcs_mass_kg"][:]
            g["delta_mass_hi95_noise_only_kg"]=g["rcs_mass_kg"][:]
            h.attrs["rcs_status"]="reference 120 K; RCS uncertainty pending"

    # Effective visibility uses zero below the horizon and the original tidal-
    # rate convention above it. The fixed radiant grid is evaluated at mid-time.
    from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic, GCRS
    from astropy.time import Time
    from astropy import units as u
    day=int(np.datetime64("2026-11-17","ns").astype(np.int64))
    starts=[]; stops=[]
    with h5py.File(root/"20261117_sanya/catalogue.h5") as h:
        starts=h["event_start_utc_ns"][:];stops=h["event_stop_utc_ns"][:]
    epoch=Time(np.longdouble(int(starts.min()+((stops.max()-starts.min())//2)))/1e9,format="unix",scale="utc")
    xg=np.arange(-180.,181.,5.); yg=np.arange(-90.,91.,5.)
    xx,yy=np.meshgrid(xg,yg)
    sun=np.mean([e["sun_lon"] for e in events])
    sky=SkyCoord(lon=(270-xx.ravel()+sun)*u.deg,lat=yy.ravel()*u.deg,frame=GeocentricTrueEcliptic(equinox=epoch)).transform_to(GCRS(obstime=epoch))
    grid_ns=np.arange(starts.min(),stops.max()+1,60000000000,dtype=np.int64)
    tt=Time(grid_ns.astype(np.longdouble)/1e9,format="unix",scale="utc")
    lst=tt.sidereal_time("apparent",longitude=109.6222*u.deg).rad
    dec=sky.dec.rad;ra=sky.ra.rad;phi=np.deg2rad(18.3492)
    sine=np.sin(phi)*np.sin(dec[:,None])+np.cos(phi)*np.cos(dec[:,None])*np.cos(lst[None,:]-ra[:,None])
    visible=(sine>0).sum(axis=1)/60
    effective=np.maximum(sine,0)**1.47
    save(out/"figure9.h5",dict(lambda_minus_sun_deg=np.array([e["radiant_lon"] for e in events]),
        beta_ecliptic_deg=np.array([e["radiant_lat"] for e in events]),speed_km_s=np.array([e["speed"] for e in events]),
        signed_westward_apex_longitude_deg=xg,grid_latitude_deg=yg,visible_hours=visible.reshape(xx.shape),effective_hours=(effective.sum(axis=1)/60).reshape(xx.shape)))
    orbital=[]
    for e in events:
        path=e["folder"]/"orbits.h5"
        if path.exists():
            with h5py.File(path) as h: orbital.append((e,{k:h[k][:] for k in h}))
    if orbital:
        with h5py.File(out/"figure10.h5","w") as h:
            h.create_dataset("event_id",data=np.array([e["folder"].name for e,d in orbital],dtype=h5py.string_dtype()))
            for key in ("a_au","e","i_deg","q_au","Q_au","specific_energy_j_kg"):
                h[key]=np.array([d[key] for e,d in orbital])
            h["cometary_probability_bound"]=[classification(d["a_au"],d["e"],d["i_deg"]) for e,d in orbital]
            h["plot_keep"]=[bool(np.isfinite(d["a_au"]).all() and np.quantile(d["Q_au"],.975)-np.quantile(d["Q_au"],.025)<20) for e,d in orbital]
    save(out.parent/"catalogue_summary.h5",dict(event_id=np.array([e["folder"].name for e in events],dtype="S32"),
        speed_km_s=np.array([e["speed"] for e in events]),height_km=event_height,
        path_rms_m=np.array([np.sqrt(np.mean(e["physical"].get("path_residual_m",e["fit"]["path_residual_m"])**2)) for e in events])),
        {"fitted_events":len(folders),"diagnostic_pass_events":len(events),"physical_events":len(physical),"orbit_events":len(orbital)})
    print("PREPARED catalogue",len(events),"physical",len(physical),"orbits",len(orbital),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/gpfs/pub/data_other/juha"))
    p.add_argument("--output", type=Path, default=Path("/gpfs/pub/data_other/juha/20261117_tristatic/figure_data"))
    p.add_argument("--workers",type=int,default=1)
    a = p.parse_args()
    overview(a.root, a.output)
    catalogue(a.root,a.output,a.workers)


if __name__ == "__main__":
    main()
