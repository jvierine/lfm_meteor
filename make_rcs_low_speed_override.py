"""Preserve raw FDTD results; impose the requested 20 -> 10 km/s column copy."""
import argparse
import hashlib
from pathlib import Path
import h5py
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.source.resolve()==a.output.resolve():raise ValueError('Do not overwrite the original lookup')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.source) as src,h5py.File(a.output.with_suffix('.partial.h5'),'w') as dst:
        for key in src:src.copy(key,dst)
        for key,value in src.attrs.items():dst.attrs[key]=value
        speed=src['velocity_km_s'][:]
        ten=np.flatnonzero(speed==10);twenty=np.flatnonzero(speed==20)
        if len(ten)!=1 or len(twenty)!=1:raise ValueError('Expected unique 10 and 20 km/s columns')
        raw=src['rcs_dbsm'][:];corrected=raw.copy()
        corrected[:,ten[0],:]=raw[:,twenty[0],:]
        dst['rcs_dbsm'][...]=corrected
        dst.attrs.update(lookup_correction='User-imposed: RCS at 10 km/s copied from 20 km/s at all altitudes and source strengths; not a new FDTD calculation',
                         original_lookup=str(a.source),original_lookup_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                         correction_program=Path(__file__).read_text())
        np.testing.assert_array_equal(dst['rcs_dbsm'][:,ten[0],:],raw[:,twenty[0],:])
        np.testing.assert_array_equal(dst['rcs_dbsm'][:,1:,:],raw[:,1:,:])
    a.output.with_suffix('.partial.h5').replace(a.output)
    print(a.output)

if __name__=='__main__':main()
