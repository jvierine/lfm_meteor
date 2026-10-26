"""Initial-mass inversion from calibrated main-lobe RCS, never dynamic mass.

Weighted dBsm-RCS least squares on the observed altitude interval.
The first-detection ITRS velocity is the approximate entry velocity.
"""
import argparse
from pathlib import Path
import sys
import warnings
import h5py
import numpy as np
from scipy.constants import c
from scipy.optimize import minimize_scalar
from spacecoords import frames
import metablate
from metablate.models.kero_szasz_2008 import KeroSzasz2008,KeroSzaszOptions,KeroSzaszInitialState
import sweep_sanya_ablation_detectability as sweep
from estimate_sanya_do_detectability import sphere_rcs

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/sanya_do_detectability'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-id')
    parser.add_argument('--event-ids',nargs='+')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--output',type=Path,default=DATA/'rcs_mass_fits.h5')
    parser.add_argument('--input-dir',type=Path,default=DATA)
    parser.add_argument('--rcs-input',type=Path,default=ROOT/'results/sanya_rcs_from_snr.h5')
    parser.add_argument('--density-kg-m3',type=float,default=3300.)
    args=parser.parse_args()
    if args.density_kg_m3<=0:parser.error('Bulk density must be positive')
    from sweep_sanya_ablation_detectability import collision_lengths,source_quantities
    material=metablate.material.get('asteroidal',as_dict=False)
    material.bulk_density=args.density_kg_m3
    with h5py.File(args.input_dir/'detectability.h5') as f:critical=f.attrs['critical_density_m3']
    with h5py.File(args.input_dir/'ablation_mass_sweep.h5') as f:
        rad=f['contour_dimensionless_radius'][:];shape=f['contour_max_shape'][:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.rcs_input) as rcs, h5py.File(args.input_dir/'rcs_mass_inversion_inputs.h5') as states, h5py.File(args.output,'w') as out:
        out.attrs.update(status='running',objective='dBsm RCS weighted least squares',
            scope='all tri-static events, independent of dynamic-mass quality',
            material='asteroidal thermal/composition parameters; only bulk density varied; plasma source treated as Si',
            bulk_density_kg_m3=material.bulk_density,
            rcs_input=str(rcs.filename),trajectory_input=str(states.filename),
            uncertainty='none yet; weights describe ideal single-pulse power noise only, not systematics',
            speed_convention='first-detection ITRS speed used as approximate speed at 150 km',
            scattering='head-on projected critical contour Rayleigh/geometric; aspect dependence not modeled')
        out['source']=Path(__file__).read_text()
        ids=rcs['event_id'].asstr()[:]
        selected=set(args.event_ids or ([args.event_id] if args.event_id else states.keys()))
        roster=[s for s in sorted(states) if s in selected]
        if args.limit:roster=roster[:args.limit]
        for event in roster:
            g=out.create_group(event)
            rows=np.flatnonzero(ids==event)
            gain_offset=rcs['sanya_relative_gain_db'][rows]
            rows=rows[np.isfinite(gain_offset)&(gain_offset>=-3.)]
            g.attrs['n_rcs_points']=len(rows)
            g.attrs['minimum_relative_gain_db']=-3.
            if len(rows)<1:
                g.attrs['status']='insufficient main-lobe RCS samples'
                out.flush();continue
            s=states[event]
            position=s['x_itrs_m'][0];velocity=s['v_itrs_mps'][0]
            lat,lon,height=frames.ecef_to_geodetic_wgs84(*position,degrees=True)
            up=np.asarray(frames.geodetic_wgs84_to_ecef(lat,lon,height+100,degrees=True))-position
            up=up/np.linalg.norm(up)
            speed=np.linalg.norm(velocity)
            zenith=np.degrees(np.arccos(np.clip(-np.dot(velocity/speed,up),-1,1)))
            g.attrs.update(initial_detection_speed_m_s=speed,zenith_deg=zenith,
                latitude_deg=lat,longitude_deg=lon,epoch_ns=int(s['time_ns'][0]))
            if zenith>=90 or speed<11000:
                g.attrs['status']='nonincoming or outside adopted Si ionization range'
                out.flush();continue
            sweep.LAT,sweep.LON=float(lat),float(lon)
            sweep.EPOCH=np.datetime64(int(s['time_ns'][0]),'ns')
            atmosphere=sweep.CachedAtmosphere()
            altitude=rcs['alt_km'][rows]*1000
            observed=rcs['rcs_m2'][rows]
            snr=10**(rcs['sanya_snr_db'][rows]/10)
            error=observed*np.sqrt(1+2*snr)/snr
            observed_dbsm=10*np.log10(observed)
            error_db=(10/np.log(10))*np.sqrt(1+2*snr)/snr
            for key,value in dict(observed_altitude_m=altitude,observed_rcs_m2=observed,
                ideal_power_noise_std_m2=error,observed_rcs_dbsm=observed_dbsm,
                ideal_power_noise_std_db=error_db,observed_time_ns=rcs['time_ns'][rows],
                observed_relative_gain_db=rcs['sanya_relative_gain_db'][rows],
                initial_detection_position_itrs_m=position,initial_detection_velocity_itrs_m_s=velocity).items():g[key]=value
            cache={}

            def evaluate(logmass):
                key=round(float(logmass),9)
                if key in cache:return cache[key]
                mass=10**logmass
                model=KeroSzasz2008(KeroSzaszOptions(atmosphere=atmosphere,
                    material=material,start_altitude=150e3,minimum_mass=mass*1e-7,
                    max_step_size=.04,max_time=50.,integral_resolution=40,
                    method_kwargs=dict(rtol=2e-6,atol=[1e-5,mass*1e-10,1e-5,1e-4])))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    result=model.run(KeroSzaszInitialState(epoch=sweep.EPOCH,
                        position_ecef=position,velocity_ecef=velocity,mass=mass,
                        drag_coefficient=1.,heat_transfer_coefficient=1.))
                h=np.asarray(frames.ecef_to_geodetic_wgs84(*result.position_ecef,degrees=True)[2])
                v=np.maximum(result.velocity,1.)
                nA=np.interp(h,atmosphere.alt,atmosphere.number)
                rho=np.interp(h,atmosphere.alt,atmosphere.raw[:,0])
                _,length=collision_lengths(v,np.maximum(result.temperature,1.),nA)
                x=18.5e-6*np.maximum(v/1000-11,0)**2*(v/1000)**.8
                beta=x/(1+x)
                _,_,nstar=source_quantities(np.maximum(-result.massloss,0.),v,beta,length,rho/nA)
                ratio=np.divide(critical,nstar,out=np.full_like(nstar,np.inf),where=nstar>1e-280)
                active=(ratio>=shape[-1])&(ratio<=shape[0])
                radius=np.zeros_like(v)
                radius[active]=length[active]*np.exp(np.interp(np.log(ratio[active]),np.log(shape[::-1]),np.log(rad[::-1])))
                sigma=np.zeros_like(v);sigma[active]=sphere_rcs(radius[active],c/430e6)
                order=np.argsort(h)
                overlap=(altitude>=np.min(h))&(altitude<=np.max(h))
                predicted=np.interp(altitude,h[order],sigma[order],left=np.nan,right=np.nan)
                # A fixed -300 dBsm numerical floor penalizes vanishing plasma;
                # it is not a detector threshold or a substituted observation.
                predicted_dbsm=10*np.log10(np.maximum(predicted,1e-30))
                cost=float(np.sum(((predicted_dbsm[overlap]-observed_dbsm[overlap])/error_db[overlap])**2)) if np.sum(overlap)>=1 else np.inf
                track=dict(altitude_m=h,rcs_m2=sigma,velocity_m_s=result.velocity,
                    temperature_k=result.temperature,mass_kg=result.mass,massloss_kg_s=result.massloss,
                    lambda_T_m=length,n_star_m3=nstar,predicted_rcs_m2=predicted,
                    predicted_rcs_dbsm=predicted_dbsm,fit_overlap_mask=overlap)
                cache[key]=(cost,track)
                return cache[key]

            grid=np.arange(-12.,-4.99,.5)
            costs=np.array([evaluate(x)[0] for x in grid])
            if not np.any(np.isfinite(costs)):
                g.attrs['status']='no supported altitude overlap; no fit'
                g['grid_log10_mass_kg']=grid;g['grid_chi_square']=costs
                out.flush();continue
            j=int(np.argmin(costs))
            if j in (0,len(grid)-1):
                best=float(grid[j]);status='mass search boundary; no supported interior minimum'
            else:
                result=minimize_scalar(lambda x:evaluate(x)[0],bounds=(grid[j-1],grid[j+1]),
                    method='bounded',options={'xatol':.002})
                best=float(result.x);status='pilot fit; uncertainty and model adequacy not yet assessed'
            cost,track=evaluate(best)
            n_overlap=int(np.sum(track['fit_overlap_mask']))
            g.attrs.update(status=status,initial_mass_kg=10**best,
                initial_radius_m=(3*10**best/(4*np.pi*material.bulk_density))**(1/3),
                chi_square=cost,nominal_dof=n_overlap-1,n_fit_points=n_overlap,
                single_point_fit=n_overlap==1,
                n_coarse_local_minima=int(np.sum((costs[1:-1]<costs[:-2])&(costs[1:-1]<costs[2:]))))
            g.attrs['rms_rcs_residual_db']=float(np.sqrt(np.nanmean((track['predicted_rcs_dbsm']-observed_dbsm)**2)))
            g['grid_log10_mass_kg']=grid;g['grid_chi_square']=costs
            for name,value in track.items():g.create_dataset('best/'+name,data=value,compression='gzip')
            g['evaluated_log10_mass_kg']=np.array(list(cache))
            g['evaluated_chi_square']=np.array([x[0] for x in cache.values()])
            out.flush()
            print(event,'n=',len(rows),'zenith=',zenith,'speed=',speed/1000,'mass=',10**best,'chi2=',cost,flush=True)
        out.attrs['status']='complete for selected roster'


if __name__=='__main__':main()
