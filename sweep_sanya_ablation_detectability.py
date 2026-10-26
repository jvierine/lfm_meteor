"""Vertical MetAblate entry -> Si ionization -> D-O -> contour RCS.

Uses the peak RCS along the full profile and its local on-axis Sanya range.
This is an idealized beam-envelope limit, not a fixed-beam crossing simulation.
"""
from pathlib import Path
import sys
import warnings
import argparse
from dataclasses import asdict

import h5py
import numpy as np
import xarray as xr
import metablate
from metablate.models.kero_szasz_2008 import KeroSzasz2008, KeroSzaszOptions, KeroSzaszInitialState
from spacecoords import frames
from scipy.constants import c, k, atomic_mass, epsilon_0, m_e, e
from scipy.optimize import brentq
from pymsis import msis
from pymsis.utils import get_f107_ap
from estimate_sanya_do_detectability import maximum_along_wake, sphere_rcs
from noise_model import MEDIAN_TSYS_K

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/sanya_do_detectability/ablation_mass_sweep.h5'
EPOCH=np.datetime64('2024-04-22T18:00:00','ns')
LAT,LON=18.567821,109.683719


def collision_lengths(speed,temperature,number_density):
    """Si collision lengths; same prescription as meteor_rcs slide examples."""
    cross_section=5.61e-19*(np.asarray(speed)/1000)**(-.8)
    thermal_speed=np.sqrt(k*np.asarray(temperature)/(28.085*atomic_mass))
    return 1/(number_density*cross_section),thermal_speed/(number_density*cross_section*speed)


def source_quantities(loss,speed,beta,length,mass_air):
    mass_si=28.085*atomic_mass
    production=beta*loss/mass_si
    q=production/speed
    return production,q,q*(1+mass_si/mass_air)/(2*np.sqrt(6*np.pi)*length**2)


class CachedAtmosphere(metablate.atmosphere.Atmosphere):
    def __init__(self):
        base=metablate.atmosphere.AtmPymsis()
        super().__init__(supported_species=list(base.species.values()))
        self.mean_mass=base.mean_mass
        self.alt=np.arange(20000.,180001.,100.)
        self.f107,self.f107a,self.ap=get_f107_ap(np.array([EPOCH]))
        self.raw=np.squeeze(msis.run([EPOCH],[LON],[LAT],self.alt/1000,
            self.f107,self.f107a,self.ap,version=2.1)).astype(float)
        self.number=np.nansum(self.raw[:,[1,2,3,4,5,6,7,9]],axis=1)
        names=['Total','N2','O2','O','He','H','Ar','N','AnomalousO','NO','Temperature']
        self.values=dict(zip(names,self.raw.T))

    def density(self,*args,**kwargs):
        alt=np.atleast_1d(kwargs['alt'])
        data={name: (['time','lon','lat','alt'],
            np.interp(alt,self.alt,value).reshape(1,1,1,-1))
            for name,value in self.values.items()}
        return xr.Dataset(data,coords=dict(time=[kwargs.get('time',EPOCH)],
            lon=[LON],lat=[LAT],alt=alt))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radar',choices=['sanya','maarsy'],default='sanya')
    parser.add_argument('--threshold-db',type=float,default=14.)
    args=parser.parse_args()
    global LAT,LON
    if args.radar=='maarsy':
        LAT,LON=69.298,16.041
        frequency,power,gain_dbi,pulse,temperature,elevation,site_alt=53.5e6,800e3,33.5,32e-6,5000.,90.,5.4
        output=OUT.with_name(f'maarsy_ablation_mass_sweep_{args.threshold_db:g}db.h5')
    else:
        frequency,power,gain_dbi,pulse,temperature,elevation,site_alt=430e6,4.7e6,45.62359445256746,200e-6,MEDIAN_TSYS_K['Sanya'],75.,50.
        output=OUT
    threshold=args.threshold_db
    atmosphere=CachedAtmosphere()
    material=metablate.material.get('asteroidal',as_dict=False)
    sys.path.insert(0,str(ROOT.parent/'manims'))
    from meteor_rcs.prepare_assets import collision_lengths,source_quantities
    # Monotone inversion of the projected critical contour, independent of
    # the atmospheric scale. Reused for every sample and mass trial.
    dim_radius=np.geomspace(1e-8,1e5,260)
    max_shape=np.array([maximum_along_wake(r)[0] for r in dim_radius])
    assert np.all(np.diff(max_shape)<0)
    critical=epsilon_0*m_e*(2*np.pi*frequency)**2/e**2
    wavelength=c/frequency
    gain=10**(gain_dbi/10)
    constant=power*gain**2*wavelength**2*pulse/((4*np.pi)**3*k*temperature)
    records={}
    OUT.parent.mkdir(exist_ok=True)
    with h5py.File(output,'w') as archive:
        archive.attrs.update(initial_zenith_deg=0.,material='asteroidal',
            density_kg_m3=material.bulk_density,threshold_db=threshold,
            metablate_version=metablate.__version__,start_altitude_m=150000.,
            geometry=f'on-axis {elevation:g} deg radar envelope; vertical entry; head-on contour RCS approximation',
            mass_criterion='SNR at altitude of maximum RCS; 58.2-130 km for MAARSY, 20-150 km for Sanya',
            ionization='Si primary Jones fit as meteor_rcs slides; all ablated mass represented as Si',
            sputtering=False,drag_coefficient=1.,heat_transfer_coefficient=1.,
            radar=args.radar,frequency_hz=frequency,peak_power_w=power,gain_dbi=gain_dbi,
            pulse_s=pulse,noise_temperature_k=temperature,elevation_deg=elevation,
            station_altitude_m=site_alt,critical_density_m3=critical,link_constant=constant,
            latitude_deg=LAT,longitude_deg=LON,epoch_utc=str(EPOCH))
        archive['source']=Path(__file__).read_text()
        for key,value in asdict(material).items():archive.attrs['material_'+key]=value
        for key,value in dict(altitude_m=atmosphere.alt,msis=atmosphere.raw,
            thermal_number_density_m3=atmosphere.number,f107=atmosphere.f107,
            f107a=atmosphere.f107a,ap=atmosphere.ap).items():archive['atmosphere/'+key]=value
        archive['contour_dimensionless_radius']=dim_radius
        archive['contour_max_shape']=max_shape

        def simulate(speed,logmass,refine=False):
            cachekey=(speed,round(logmass,10),refine)
            if cachekey in records:return records[cachekey]
            mass=10**logmass
            model=KeroSzasz2008(KeroSzaszOptions(atmosphere=atmosphere,
                material=material,start_altitude=150e3,minimum_mass=mass*1e-7,
                max_step_size=.01 if refine else .04,max_time=30.,
                integral_resolution=40,sputtering=False,
                method_kwargs=dict(rtol=1e-7 if refine else 2e-6,
                    atol=[1e-5,mass*1e-10,1e-5,1e-4])))
            position=frames.geodetic_wgs84_to_ecef(LAT,LON,100e3,degrees=True)
            direction=-frames.azel_to_ecef(LAT,LON,0.,90.,degrees=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                result=model.run(KeroSzaszInitialState(epoch=EPOCH,
                    position_ecef=np.asarray(position),velocity_ecef=np.asarray(direction)*speed*1000,
                    mass=mass,drag_coefficient=1.,heat_transfer_coefficient=1.))
            altitude=frames.ecef_to_geodetic_wgs84(*result.position_ecef,degrees=True)[2]
            v=np.asarray(result.velocity)
            loss=np.maximum(-result.massloss,0.)
            nA=np.interp(altitude,atmosphere.alt,atmosphere.number)
            rho=np.interp(altitude,atmosphere.alt,atmosphere.raw[:,0])
            _,length=collision_lengths(np.maximum(v,1.),np.maximum(result.temperature,1.),nA)
            vk=np.maximum(v/1000,0.)
            ion_x=18.5e-6*np.maximum(vk-11,0.)**2*vk**.8
            beta=ion_x/(1+ion_x)
            production,q,nstar=source_quantities(loss,np.maximum(v,1.),beta,length,rho/nA)
            ratio=np.divide(critical,nstar,out=np.full_like(nstar,np.inf),where=nstar>1e-280)
            # Never extrapolate a nonzero contour outside the tabulated range.
            active=(nstar>0)&(ratio<=max_shape[0])
            radius=np.zeros_like(v)
            assert np.all(ratio[active]>=max_shape[-1])
            radius[active]=length[active]*np.exp(np.interp(np.log(ratio[active]),
                np.log(max_shape[::-1]),np.log(dim_radius[::-1])))
            rcs=np.zeros_like(radius)
            rcs[active]=sphere_rcs(radius[active],wavelength)
            ranges=-(6371000+site_alt)*np.sin(np.deg2rad(elevation))+np.sqrt(
                (6371000+altitude)**2-(6371000+site_alt)**2*np.cos(np.deg2rad(elevation))**2)
            snr=constant*rcs/np.maximum(ranges,1.)**4
            low,high=(58200.,130000.) if args.radar=='maarsy' else (20000.,150001.)
            valid=(altitude>=low)&(altitude<=high)&np.isfinite(snr)
            in_window=bool(np.any(valid))
            peak=int(np.argmax(np.where(valid,rcs,-1))) if in_window else 0
            metric=10*np.log10(max(snr[peak],1e-300)) if in_window else -3000.
            track=dict(time_s=result.t,altitude_m=altitude,velocity_m_s=v,mass_kg=result.mass,
                temperature_k=result.temperature,mass_loss_rate_kg_s=loss,
                beta=beta,production_s1=production,line_density_m1=q,n_star_m3=nstar,
                lambda_T_m=length,critical_contour_radius_m=radius,rcs_m2=rcs,snr=snr)
            trial=archive.require_group(f'trials/v{speed:g}').create_group(f'{len(records):04d}')
            trial.attrs.update(initial_mass_kg=mass,snr_at_peak_rcs_db=metric,
                peak_rcs_m2=rcs[peak],peak_rcs_altitude_km=altitude[peak]/1000,refined=refine)
            for key,value in track.items():trial.create_dataset(key,data=value,compression='gzip')
            archive.flush()
            print(f'v={speed:g} m={mass:.5g} peak RCS={rcs[peak]:.4g} at {altitude[peak]/1000:.2f} km SNR={metric:.3f} dB',flush=True)
            records[cachekey]=(metric,trial.name,peak)
            return records[cachekey]

        for speed in [10.,30.,50.,72.]:
            summary=archive.create_group(f'summary/v{speed:g}')
            summary.attrs['initial_velocity_km_s']=speed
            if speed==10:
                simulate(speed,-8.)
                summary.attrs['status']='No finite threshold under adopted Si ionization law (U<11 km/s)'
                archive.flush()
                continue
            grid=np.arange(-15.,-2.9,1.)
            bracket=None
            previous=None
            for logmass in grid:
                value=simulate(speed,logmass)[0]-threshold
                if previous is not None and previous[1]<0<=value:
                    bracket=(previous[0],logmass)
                    break
                previous=(logmass,value)
            if bracket is None:
                summary.attrs['status']='No threshold bracket in searched mass range'
                continue
            root=brentq(lambda x:simulate(speed,x)[0]-threshold,*bracket,xtol=.001)
            metric,path,peak=simulate(speed,root,True)
            summary.attrs.update(status='threshold found',initial_mass_kg=10**root,
                snr_at_peak_rcs_db=metric,profile_path=path,
                diameter_m=(6*10**root/(np.pi*material.bulk_density))**(1/3))
            trial=archive[path]
            summary.attrs['peak_rcs_altitude_km']=trial['altitude_m'][peak]/1000
            summary.attrs['peak_rcs_m2']=trial['rcs_m2'][peak]
            archive.flush()
        archive.attrs['complete']=True
    print(output,flush=True)


if __name__=='__main__':main()
