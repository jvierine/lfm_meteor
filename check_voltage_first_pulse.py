"""Isolated sensitivity refit; never changes catalogue selections."""
import sys
import h5py
import numpy as np
import run_native_voltage_catalog as driver

original_prepare=driver.prepare


def prepare(path, events, reference):
    data=original_prepare(path,events,reference)
    candidates=np.flatnonzero(data['station']==0)
    index=candidates[np.argmin(data['time_s'][candidates])]
    keep=np.arange(len(data['time_s']))!=index
    row_keys=('time_s','time_ns','frames','rx','station','real','imag','noise_power',
              'whitener_real','whitener_imag','sample_time_s','gate')
    with h5py.File(path,'a') as h:
        h.attrs['excluded_sanya_time_ns']=int(data['time_ns'][index])
        h.attrs['selection_reason']='Sensitivity test: exclude first usable Sanya pulse only'
        for key in row_keys:
            data[key]=data[key][keep]
            del h[key];h[key]=data[key]
    return data


if __name__=='__main__':
    driver.prepare=prepare
    driver.main()
