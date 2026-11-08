"""Summarize finite repeated-simulation coverage of pulse posterior bootstrap."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import binomtest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--posterior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    names, coverage, bias, checks, boundary, missing = [], [], [], [], [], []
    for folder in sorted(a.inputs.glob('sim_*')):
        path = a.posterior/folder.name/'posterior.h5'
        if not path.exists():
            missing.append(folder.name)
            continue
        with h5py.File(folder/'truth.h5') as h:
            truth = h['params'][:]
        with h5py.File(path) as h:
            q = h['parameter_quantiles'][:]
            samples = h['samples'][:]
            coverage.append((q[0] <= truth) & (truth <= q[2]))
            bias.append((h['best_params'][:]-truth)/np.std(samples,axis=0,ddof=1))
            checks.append(bool(h.attrs['numerical_checks_passed']))
            boundary.append(float(h.attrs['prior_boundary_fraction']))
        names.append(folder.name)
    if not names:
        raise RuntimeError('No completed simulation posteriors')
    coverage, bias = np.asarray(coverage), np.asarray(bias)
    count, n = int(coverage[:,6].sum()), len(names)
    test = binomtest(count,n,.95,alternative='less')
    interval = binomtest(count,n).proportion_ci(method='exact')
    with h5py.File(a.output,'w') as h:
        h['event_id'] = np.asarray(names,dtype=h5py.string_dtype())
        h['missing_event_id'] = np.asarray(missing,dtype=h5py.string_dtype())
        h['parameter_covered'] = coverage
        h['parameter_coverage_count'] = coverage.sum(axis=0)
        h['standardized_best_fit_error'] = bias
        h['numerical_checks_passed'] = checks
        h['prior_boundary_fraction'] = boundary
        h.attrs.update(completed=n,missing=len(missing),numerical_successes=sum(checks),
            mass_coverage_count=count,mass_coverage_fraction=count/n,
            mass_coverage_exact_lower95=interval.low,mass_coverage_exact_upper95=interval.high,
            mass_undercoverage_pvalue=test.pvalue,
            conditional_simulation_check_passed=not missing and all(checks) and test.pvalue>=.05,
            scope='Finite simulation check at one measured geometry/truth; not proof of total physical coverage',
            uncertainty_method='Stratified independent pulse posterior bootstrap')
    print('COVERAGE',count,'/',n,'numerical',sum(checks),'/',n,
          'exact95',[interval.low,interval.high],'undercoverage_p',test.pvalue,
          'missing',len(missing),flush=True)


if __name__ == '__main__':
    main()
