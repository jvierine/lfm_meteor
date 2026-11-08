"""Compact mass-marginal comparison; metadata records different statistical targets."""
import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['old','new','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    probability = np.linspace(.001,.999,501)
    with h5py.File(a.output,'w') as out:
        out.attrs.update(old_method='Receiver-noise-only parametric posterior',
            new_method='Independent pulse posterior bootstrap of fitted-parameter functional',
            physical_uncertainty_validated=False)
        for path in sorted(a.new.glob('tri_*/posterior.h5')):
            event = path.parent.name
            with h5py.File(a.old/event/'posterior.h5') as h:
                logr = h['samples'][:,6]
                weights = h['weights'][:]
                keep = np.isfinite(logr)&(weights>0)
                order = np.argsort(logr[keep])
                logm = np.log10(4*np.pi/3*3000)+3*logr[keep][order]
                w = weights[keep][order]
                cdf = (np.cumsum(w)-.5*w)/w.sum()
                old = np.interp(probability,cdf,logm)
                old_quantiles = h['mass_kg_quantiles'][:]
            with h5py.File(path) as h:
                new = np.quantile(np.log10(h['mass_kg'][:]),probability)
                g = out.create_group(event)
                g['old_logmass'] = old
                g['new_logmass'] = new
                g['probability'] = probability
                g['old_mass_quantiles_kg'] = old_quantiles
                g['new_mass_quantiles_kg'] = h['mass_kg_quantiles'][:]
                g.attrs.update(numerical_checks_passed=h.attrs['numerical_checks_passed'],
                    prior_boundary_fraction=h.attrs['prior_boundary_fraction'],
                    elapsed_seconds=h.attrs['elapsed_seconds'])
                print(event,'old',old_quantiles,'new',g['new_mass_quantiles_kg'][:],
                      'checks',g.attrs['numerical_checks_passed'],flush=True)


if __name__ == '__main__':
    main()
