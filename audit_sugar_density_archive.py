"""Audit Sugar's archived D-O density against independently evaluated shape.

Run on avaruus: the original 512^3 MAT file is ~1 GB. Only reduced samples
and a 2-D slice are saved to the output HDF5; no field evolution is performed.
"""
from pathlib import Path
import hashlib
import urllib.request
import h5py
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.constants import epsilon_0,m_e,e
from estimate_sanya_do_detectability import density

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/sugar2021_reproduction'
NAME='domodel_nx512_ny512_nz512_dx0.005_lambdat0.0811_fast.mat'


def main():
    source=OUT/'source'/NAME
    source.parent.mkdir(parents=True,exist_ok=True)
    if not source.exists():
        print('Downloading original D-O density archive',flush=True)
        urllib.request.urlretrieve('https://zenodo.org/api/records/4514583/files/'+NAME+'/content',source)
    md5=hashlib.md5()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            md5.update(block)
    assert md5.hexdigest()=='15fde3f1eb52ae30a922ef2955472462'
    print('Archive fields',whosmat(source),flush=True)
    loaded=loadmat(source)
    fields={k:v for k,v in loaded.items() if not k.startswith('__')}
    print('Field shapes',[(k,np.shape(v)) for k,v in fields.items()],flush=True)
    array=fields.get('np',fields.get('wp'))
    n=array.shape[0]
    dx=.005
    coord=(np.arange(n)-(n-1)/2)*dx
    middle=n//2-1
    length=951/(1e19*5.61e-19*40**(-.8)*40000)
    # Several axial signs and permutations identify the archived coordinate convention.
    targets=[(-.2,.02),(-.1,.1),(-.02,.02),(.02,.02),(.1,.1),(.2,.02),(.5,.1),(1.,.2)]
    with h5py.File(OUT/'density_audit.h5','w') as out:
        out.attrs.update(source_program=Path(__file__).read_text(),source_md5=md5.hexdigest(),
            velocity_m_s=40000.,thermal_speed_m_s=951.,neutral_number_m3=1e19,
            lambda_T_m=length,dx_m=dx,archive_field='np' if 'np' in fields else 'wp')
        out['coordinate_m']=coord
        out['archive_slice']=array[:,:,middle]
        out.attrs['archive_min']=float(np.nanmin(array))
        out.attrs['archive_max']=float(np.nanmax(array))
        for perm in range(3):
            rows=[]
            for z,r in targets:
                iz=np.argmin(abs(coord-z));ir=np.argmin(abs(coord-r))
                indices=([iz,ir,middle],[ir,iz,middle],[ir,middle,iz])[perm]
                p=np.array([coord[k] for k in indices])
                axial=p[perm]
                transverse=np.sqrt(np.sum(np.delete(p,perm)**2))
                expected=float(density(axial/length,transverse/length,128))
                reversed_shape=float(density(-axial/length,transverse/length,128))
                value=float(array[tuple(indices)])
                rows.append([axial,transverse,value,expected,reversed_shape,value/expected,value/reversed_shape])
            out[f'permutation_{perm}']=rows
            print('perm',perm,'z,r,value,D(+z),D(-z),ratios',np.array(rows),flush=True)
        q=6e10;vt=951.;na=1e19;sig=5.61e-19*40**(-.8);beta=.7055413586
        source_strength=q/(4*np.sqrt(2*np.pi)*vt)
        nstar=8*np.pi*source_strength*na*(1+3.8e-26/4.7e-26)*(sig*beta/(4*np.pi))/(np.sqrt(3)*length)
        out.attrs['nstar_for_Q6e10_m3']=nstar
        out.attrs['C_in_make_figures_for_Q6e10']=nstar/np.pi
        print('Our nstar for Q=6e10:',nstar,'archived plotting coefficient C:',nstar/np.pi,flush=True)
    print('Wrote reduced density audit',flush=True)


if __name__=='__main__':
    main()
