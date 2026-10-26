"""Replot archived Figure 8 and test our scattering on archived contour radii.

This reuses published FDTD results; it does not rerun Maxwell's equations.
"""
from pathlib import Path
import hashlib
import io
import zipfile
import h5py
import numpy as np
from scipy.io import loadmat
from estimate_sanya_do_detectability import sphere_rcs

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/sugar2021_reproduction'


def main():
    archive=OUT/'source/RCS_Data.zip'
    assert hashlib.md5(archive.read_bytes()).hexdigest()=='0f77d3247f135be91aa2a6dd5db00526'
    vp=np.sqrt(1/(4*np.pi*1e-7*8.845e-12))
    rows={'do':[],'pic':[]}
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if 'freq38450000' not in name or not name.endswith('.mat'):
                continue
            model='do' if '/type3_' in name else 'pic'
            sig=loadmat(io.BytesIO(source.read(name)),simplify_cells=True)['sig']
            if 'k' not in sig:
                continue
            for j,wavenumber in enumerate(np.atleast_1d(sig['k'])):
                if abs((wavenumber-2*np.pi*38.45e6/vp)/wavenumber)>=.01:
                    continue
                coefficient=1.397739812665341e10 if model=='do' else 131663.**2
                rate=sig['wpfactor']**2/coefficient*5e13
                radius=np.atleast_1d(sig['rp'])[j]
                rcs=np.atleast_1d(sig['sig'])[j]
                approximation=sphere_rcs(radius,2*np.pi/wavenumber) if radius>0 else 0.
                rows[model].append((rate,rcs,radius,approximation,wavenumber))
    with h5py.File(OUT/'figure8.h5','w') as out:
        out.attrs.update(source_program=Path(__file__).read_text(),source_doi='10.5281/zenodo.4514583',
            simulation_frequency_hz=38.45e6,new_fdtd_simulations=False,
            comparison='Apply our scattering formula to saved FDTD critical-contour radii; no inferred plasma normalization')
        for model,values in rows.items():
            a=np.array(sorted(values))
            assert len(a)==16
            group=out.create_group(model)
            for j,name in enumerate(['ablation_rate_particles_s','fdtd_rcs_m2','archived_radius_m',
                                      'sphere_rcs_m2','wavenumber_m_inv']):
                group[name]=a[:,j]
            fit=np.polyfit(np.log(a[:,0]),np.log(a[:,1]),1)
            group.attrs.update(power_exponent=fit[0],power_prefactor=np.exp(fit[1]))
            valid=a[:,2]>0
            differences=10*np.log10(a[valid,3]/a[valid,1])
            print(model,'N',len(a),'slope',fit[0],'zero contour count',sum(~valid),
                  'same-radius RCS difference min/median/max dB',np.percentile(differences,[0,50,100]))
            print('rate, FDTD dBsm, radius m, contour-sphere dBsm')
            for row in a:
                print(row[0],10*np.log10(row[1]),row[2],10*np.log10(row[3]) if row[3]>0 else '-inf')


if __name__=='__main__':
    main()
