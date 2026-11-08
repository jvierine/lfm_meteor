"""Reuse the appendix numerical recipe without importing its plotting code."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import scipy.signal as sig

import sanya_opts as sc

ROOT = Path(__file__).resolve().parent


def numerical_functions(filename, names, namespace):
    source = ROOT / filename
    tree = ast.parse(source.read_text(), filename=str(source))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)


def main():
    output = ROOT / 'results/single_column_20261108'
    beat = {'np': np, 'DEFAULT_REFERENCE_CHIRP_RATE_SCALE': 1.0}
    numerical_functions('estimate_single_pulse_dechirped_fft_doppler.py',
                        ['quadratic_peak_frequency', 'lfm_reference_for_offsets', 'fft_power_metrics'], beat)
    namespace = dict(np=np, h5py=h5py, os=os, sig=sig, sc=sc,
                     beat=SimpleNamespace(**beat), SCRIPT_VERSION='v20260702b',
                     gfit=SimpleNamespace(C=299792458., LFM_DURATION_S=200e-6,
                                          REFERENCE_CHIRP_RATE_SCALE=1., CHIRP_RATE_HZ_PER_S=2e10))
    numerical_functions('plot_appendix_dechirped_beat_example.py',
                        ['load_fit_context', 'estimate_with_canonical_pulse_length', 'write_h5'], namespace)
    # The raw event stores UTC+8 stamps; the archived fit uses UTC.
    load_context = namespace['load_fit_context']
    namespace['load_fit_context'] = lambda path, stamp, site: load_context(path, stamp - 8 * 3600 * 10**9, site)
    fit = ROOT / 'results/rcs_integrated_mass_fdtd_20261028/wj_bootstrap_sources/joint_delay_doppler_fft_tri_0156_1713822937704349518.h5'
    result = namespace['estimate_with_canonical_pulse_length'](
        str(output / 'appendix_raw.h5'), 18, 64, 50., 32, fit_h5=str(fit))
    namespace['write_h5'](result, str(output / 'beat'), __file__)
    print('Beat peak:', result['peak_frequency_hz'], 'Hz; SNR:', result['snr_peak_db'], 'dB')


if __name__ == '__main__':
    main()
