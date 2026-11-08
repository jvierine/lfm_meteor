"""Export one consistent conditional posterior and current diagnostics per event."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import h5py
from mpi4py import MPI
import numpy as np

from fit_cached_shrinking_voltage import native
from fit_shrinking_voltage import Model, export, full_products, save
from voltage_path_uncertainty import add_intervals
from sample_seven_parameter_importance import quantiles


def systematic_indices(weights, size):
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.
    return np.searchsorted(cdf, (np.arange(size) + .5) / size)


def export_one(root, event, library):
    source = root / 'inputs' / event
    folder = root / 'catalog' / event
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'input.h5', folder / 'input.h5')
    with h5py.File(source / 'input.h5') as h:
        data = {k: h[k][()] for k in h}
        original = Path(h.attrs['original_input'])
    posterior = root / 'posterior' / event / 'posterior.h5'
    with h5py.File(posterior) as h:
        samples, weights = h['samples'][:], h['weights'][:]
        q = h['proposal_center'][:]
        verified = bool(h.attrs['convergence_verified'])
        radius = h['radius_m_quantiles'][:]
        mass = h['mass_kg_quantiles'][:]
        diagnostics = {k: h.attrs[k] for k in ('importance_ess', 'maximum_normalized_weight',
                       'maximum_exact_likelihood_difference', 'batch_log_normalizer_std')}
        input_hash = h.attrs['source_input_sha256']
    assert hashlib.sha256((folder/'input.h5').read_bytes()).hexdigest() == input_hash
    lib = native(library)
    model = Model(lib, folder, data)
    try:
        prediction = model.prediction(q)
        timestamp = time.time()
        fit = {'fit_unix_time': timestamp}
        export(folder, model, q, fit)
        with h5py.File(folder / 'plot.h5') as h:
            fields, attrs = {k: h[k][()] for k in h}, dict(h.attrs)
        save(folder/'fit.h5', dict(params=q, predicted_real=prediction.real,
             predicted_imag=prediction.imag, seed_apparent_path_m=fields['seed_apparent_path_m']),
             dict(model='direct shrinking radius; exact cached joint voltage likelihood',
                  optimizer_success=True, fit_unix_time=timestamp, Gamma=1., Cd=2.,
                  material_density_kg_m3=3000., sigma_s2_m2=1e-8,
                  uncertainty_method='independent defensive importance posterior',
                  importance_verified=verified,
                  posterior_sha256=hashlib.sha256(posterior.read_bytes()).hexdigest()))
        boundary = bool(radius[0] <= 1.01e-7 or radius[2] >= .0099
                        or 10**q[6] <= 1.001e-7 or 10**q[6] >= .00999)
        selected = verified and not boundary and mass[2] < 10*mass[0]
        reason = ('posterior sampling unresolved' if not verified else
                  'radius prior boundary' if boundary else
                  '95 percent mass interval spans at least a decade' if not selected else '')
        attrs.update(uncertainty_converged=verified, uncertainty_prior_boundary=boundary,
             uncertainty_method='independent defensive importance posterior',
             uncertainty_status=('95% conditional physical-model posterior interval' if verified
                                 else 'Posterior unresolved; excluded from dynamic mass estimates'),
             mass_status=('Direct shrinking-radius mass; conditional on density, drag and ablation'
                          if verified else 'No accepted dynamic mass: posterior sampling unresolved'),
             dynamic_mass_included=selected, dynamic_mass_exclusion_reason=reason,
             uncertainty_alpha=.5, **diagnostics)
        if verified:
            indices = systematic_indices(weights, 2048)
            paths, speeds = [], []
            order = np.argsort(data['time_s'])
            for draw in samples[indices]:
                state = model.states(draw)
                paths.append(model.paths(draw, state))
                speeds.append(state[order, 1]/1000)
            fields['path_lower95_m'], fields['path_upper95_m'] = np.quantile(paths, [.025,.975], axis=0)
            fields['speed_lower95_km_s'], fields['speed_upper95_km_s'] = np.quantile(speeds, [.025,.975], axis=0)
            fields['initial_radius_m_quantiles'] = radius
            fields['initial_mass_kg_quantiles'] = mass
            draw_indices = systematic_indices(weights, 65536)
            draws = samples[draw_indices]
            save(folder/'state_posterior.h5', dict(samples=draws),
                 dict(convergence_verified=True, method=attrs['uncertainty_method'],
                      resampling='Deterministic equal-weight stratified resampling of weighted posterior',
                      posterior_source=str(posterior), importance_ess=diagnostics['importance_ess']))
            save(folder/'dynamic_mass_posterior.h5',
                 dict(radius_m=10**draws[:,6], mass_kg=4*np.pi/3*3000*10**(3*draws[:,6])),
                 dict(convergence_verified=True, uncertainty_method=attrs['uncertainty_method'],
                      posterior_source=str(posterior)))
            v0 = data['center'][3:] + 1000*samples[:,3:6]
            fields['initial_speed_mps_quantiles'] = quantiles(np.linalg.norm(v0,axis=1),weights)
        save(folder/'plot.h5', fields, attrs)
        # Restore the original full pulse list only for diagnostics and RCS.
        full_products(SimpleNamespace(source=original.parent.parent), event, folder, model, q)
        add_intervals(folder)
        save(folder/'complete.h5', {}, dict(finished_unix=time.time(),
             posterior_verified=verified, dynamic_mass_selected=selected,
             dynamic_mass_exclusion_reason=reason))
        print('EXPORTED', event, 'verified', verified, 'mass', selected, reason, flush=True)
    finally:
        lib.voltage_free(model.handle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--events', nargs='+')
    a = p.parse_args()
    comm = MPI.COMM_WORLD
    events = a.events or sorted(x.parent.name for x in (a.root/'inputs').glob('tri_*/input.h5'))
    errors = []
    for event in events[comm.rank::comm.size]:
        try:
            export_one(a.root, event, a.library)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            errors.append((event,repr(exc)))
    gathered = comm.gather(errors,root=0)
    if comm.rank == 0:
        errors = sum(gathered, [])
        save(a.root/'export_status.h5',
             {'error_event': np.asarray([x[0] for x in errors],dtype='S'),
              'error_reason': np.asarray([x[1] for x in errors],dtype='S')},
             dict(expected_events=len(events), exported_events=len(events)-len(errors),
                  completed_unix=time.time()))


if __name__ == '__main__':
    main()
