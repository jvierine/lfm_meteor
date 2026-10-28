"""Audited, correlated-noise posterior refits of cached tri-static observations.

This is a separate validation catalogue, never an implicit replacement for the
published catalogue. Noise correlation is an empirical-Bayes estimate from a
refitted WJ trajectory. Conditional posterior intervals exclude uncertainty in
that correlation estimate, atmospheric density, drag and material composition.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap
import time

import emcee
import h5py
import jcoord
import numpy as np
from scipy import linalg, optimize, special, stats

import fit_all_ballistic_snr_weighted as base
import fit_all_ceplecha_snr_weighted as cepl
import fit_event_joint_delay_doppler_fft as fit
from fit_whipple_jacchia_catalog_from_h5 import load_group, event_id_from_path, whipple_seed_grid


VERSION = 'correlated_mass_posterior_v1_20261028'
LOG_RADIUS_BOUNDS = (-7., -2.)


def attach_geodetic_height(density, times):
    # Earth rotation changes longitude, not ellipsoidal height. Over a meteor
    # event the fixed epoch GCRS-to-ITRS rotation suffices for the height query.
    offset, _, rotation, _, _, _ = fit.precomputed_gcrs_to_itrs_linear(np.asarray(times[:1], dtype=np.int64))
    density.height_from_gcrs = lambda pos: float(jcoord.ecef2geodetic(*(offset[0] + rotation[0] @ pos))[2])
    return density


def normal_scores(values, nu):
    # Inverse CDF gives a Gaussian copula without changing Student-t marginals.
    return special.ndtri(np.clip(special.stdtr(nu, values), 1e-12, 1 - 1e-12))


def copula_loglike(residual, scales, nu, chol):
    z = residual / scales
    g = normal_scores(z, nu)
    white = linalg.solve_triangular(chol, g, lower=True, check_finite=False)
    return (np.sum(stats.t.logpdf(z, nu) - np.log(scales))
            - .5 * (white @ white - g @ g) - np.log(np.diag(chol)).sum())


def correlation_parameters(z, mask, nu):
    g = normal_scores(z, nu)
    both = np.all(mask, axis=1)
    rho = 0.
    if both.sum() >= 12:
        # Winsorization protects covariance estimation from model outliers.
        x = np.clip(g[both], -2.5, 2.5)
        rho = np.corrcoef(x.T)[0, 1]
        rho = float(np.clip(np.nan_to_num(rho) * both.sum() / (both.sum() + 20), -.95, .95))
    phis = []
    for c in range(2):
        k = mask[:-1, c] & mask[1:, c]
        if k.sum() >= 12:
            a = np.clip(g[:-1, c][k], -2.5, 2.5)
            b = np.clip(g[1:, c][k], -2.5, 2.5)
            phis.append(np.nan_to_num(np.corrcoef(a, b)[0, 1]) * k.sum() / (k.sum() + 20))
    # Positive continuous-time AR(1) permits gaps and unequal pulse intervals.
    phi = float(np.clip(max(phis, default=0.), 0., .95))
    return rho, phi


def correlation_cholesky(times, mask, rho, phi):
    rows, channels = np.nonzero(mask)
    cadence = np.median(np.diff(times))
    lag = np.abs(times[rows, None] - times[None, rows]) / cadence
    temporal = np.where(lag == 0, 1., 0.) if phi == 0 else phi ** lag
    corr = temporal * np.where(channels[:, None] == channels[None, :], 1., rho)
    return np.linalg.cholesky(corr + np.eye(len(rows)) * 1e-10)


class EventLikelihood:
    def __init__(self, source, rho_of_alt, reference):
        self.source = source
        self.rho_of_alt = rho_of_alt
        self.times = np.asarray(source['time_ns'], dtype=np.int64)
        self.t = (self.times - self.times[0]).astype(float) * 1e-9
        self.measured = np.asarray(source['measured_total_paths_m'])
        self.beat = np.asarray(source['observed_fft_beat_hz'])
        self.path_sigma = np.asarray(source['path_sigma_m'])
        self.fft_sigma = np.broadcast_to(source['fft_sigma_hz'], self.beat.shape).copy()
        self.pk = np.asarray(source['path_keep'], bool) & np.isfinite(self.measured) & (self.path_sigma > 0)
        self.fk = np.asarray(source['fft_keep'], bool) & np.isfinite(self.beat) & (self.fft_sigma > 0)
        self.center = np.r_[reference['params'][:3], reference['v_gcrs_mps'][0]]
        self.nu = np.array([float(source.get('student_t_nu_delay', 1.5)), float(source.get('student_t_nu_fft', 3.))])
        self.alpha = fit.gfit.NOMINAL_CHIRP_RATE_HZ_PER_S * float(source.get('reference_chirp_rate_scale', 1.)) / fit.gfit.C
        if source.get('fft_model', '') not in ('range_offset_corrected_beat', 'ambiguity_residual'):
            raise ValueError('Audit requires the canonical range-offset-corrected beat convention')
        self.blocks = []
        for c in range(3):
            mask = np.c_[self.pk[:, c], self.fk[:, c]]
            z = np.c_[reference['path_residuals_m'][:, c] / self.path_sigma[:, c],
                      reference['fft_residuals_hz'][:, c] / self.fft_sigma[:, c]]
            rho, phi = correlation_parameters(z, mask, self.nu)
            chol = correlation_cholesky(self.t, mask, rho, phi)
            self.blocks.append((mask, chol, rho, phi))

    def unpack(self, q):
        return np.r_[self.center + q[:6] * 1000., q[6]], q[7:10] * 1000., np.exp(q[10:16]).reshape(3, 2)

    def model(self, q):
        p, bias, scales = self.unpack(q)
        m = fit.forward_model_for_kind(p, self.t, self.times, self.rho_of_alt, 'ceplecha')
        if not m['ceplecha_success']:
            raise ValueError('ODE integration failed')
        path = m['apparent_path_length_m'] - self.measured
        beat = m['doppler_hz'] - self.alpha * (self.measured - m['path_length_m']) + bias - self.beat
        return m, path, beat, scales

    def log_likelihood(self, q):
        try:
            m, path, beat, scales = self.model(q)
            if not np.all(np.isfinite(m['mass_kg'])) or np.any(m['mass_kg'] <= 0):
                return -np.inf
            value = 0.
            for c, (mask, chol, _, _) in enumerate(self.blocks):
                residual = np.c_[path[:, c], beat[:, c]][mask]
                sigma = (np.c_[self.path_sigma[:, c], self.fft_sigma[:, c]] * scales[c])[mask]
                nu = np.broadcast_to(self.nu, mask.shape)[mask]
                value += copula_loglike(residual, sigma, nu, chol)
            return float(value) if np.isfinite(value) else -np.inf
        except (ValueError, FloatingPointError, RuntimeError):
            return -np.inf

    def __call__(self, q):
        p, _, _ = self.unpack(q)
        if (np.any(np.abs(q[:3]) > 20) or np.any(np.abs(p[3:6]) > 80000)
                or not LOG_RADIUS_BOUNDS[0] < q[6] < LOG_RADIUS_BOUNDS[1]
                or np.any(np.abs(q[7:10]) > 500) or np.any(np.abs(q[10:]) > 4)):
            return -np.inf
        # Uniform log radius, broad bounded state/bias; lognormal noise scales.
        return self.log_likelihood(q) - .5 * np.sum((q[10:] / np.log(3.)) ** 2)


def refit_reference(source, density):
    candidates = []
    for p in whipple_seed_grid(source, 8):
        try:
            c = fit.fit_joint_delay_doppler(
                source['measured_total_paths_m'], source['time_ns'], density, p,
                source['path_sigma_m'], source['observed_fft_beat_hz'], source['fft_keep'], source['fft_sigma_hz'],
                path_keep=source['path_keep'], model_kind='whipple_speed', fit_station_bias=True,
                residual_likelihood='student_t', max_nfev_override=500,
                student_t_nu_delay=float(source.get('student_t_nu_delay', 1.5)),
                student_t_nu_fft=float(source.get('student_t_nu_fft', 3.)))
            if c['optimizer_success'] and c['ceplecha_success']:
                candidates.append(c)
        except (ValueError, RuntimeError):
            continue
    if not candidates:
        raise RuntimeError('No converged WJ reference fit')
    return min(candidates, key=lambda c: c['optimizer_cost'])


def posterior_flags(chain, tau, mass, log_radius):
    flags = []
    if not np.all(np.isfinite(tau)) or chain.shape[0] < 50 * np.nanmax(tau):
        flags.append('chain_too_short_for_autocorrelation_time')
    if np.mean(log_radius > LOG_RADIUS_BOUNDS[1] - .25) > .025:
        flags.append('upper_mass_tail_prior_limited')
    if np.mean(log_radius < LOG_RADIUS_BOUNDS[0] + .25) > .025:
        flags.append('lower_mass_tail_prior_limited')
    if np.percentile(mass, 97.5) / np.percentile(mass, 2.5) > 10:
        flags.append('mass_interval_wider_than_one_decade')
    return flags


def save_group(group, data):
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool, np.number)):
            group.attrs[key] = value
        else:
            arr = np.asarray(value)
            if arr.dtype.kind == 'U':
                arr = arr.astype(h5py.string_dtype())
            group.create_dataset(key, data=arr, compression='gzip' if arr.ndim and arr.size > 50 else None)


def event_plot(path, event, like, reference, q, samples, flags, snr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model, path_resid, beat_resid, scales = like.model(q)
    speed = np.linalg.norm(model['v_gcrs_mps'], axis=1) / 1000
    curves = []
    for sample in samples[np.linspace(0, len(samples) - 1, min(100, len(samples))).astype(int)]:
        try:
            curves.append(np.linalg.norm(like.model(sample)[0]['v_gcrs_mps'], axis=1) / 1000)
        except (ValueError, RuntimeError):
            pass
    mass = 4 * np.pi * cepl.METEOROID_DENSITY_KG_M3 * 10 ** (3 * samples[:, 6]) / 3
    with plt.rc_context({'font.size': 10}):
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
        colors = ['#377eb8', '#e08024', '#319052']
        for c in range(3):
            k = like.pk[:, c]
            zero = like.measured[0, c]
            axes[0, 0].plot(like.t[k], (like.measured[k, c] - zero) / 1000, '.', color=colors[c])
            axes[0, 0].plot(like.t, (model['apparent_path_length_m'][:, c] - zero) / 1000, color=colors[c], label=fit.SITE_LABELS[c])
            axes[1, 0].plot(like.t[k], path_resid[k, c], '.', color=colors[c])
            error_rows = np.flatnonzero(k)[::max(1, int(np.sum(k)) // 16)]
            axes[1, 0].errorbar(like.t[error_rows], path_resid[error_rows, c],
                               yerr=like.path_sigma[error_rows, c] * scales[c, 0] * stats.t.ppf(.84, like.nu[0]),
                               fmt='none', color=colors[c], alpha=.4, linewidth=.7)
            k = like.fk[:, c]
            axes[1, 1].plot(like.t[k], beat_resid[k, c] * fit.gfit.RADAR_WAVELENGTH_M, '.', color=colors[c])
            error_rows = np.flatnonzero(k)[::max(1, int(np.sum(k)) // 16)]
            axes[1, 1].errorbar(like.t[error_rows], beat_resid[error_rows, c] * fit.gfit.RADAR_WAVELENGTH_M,
                               yerr=like.fft_sigma[error_rows, c] * scales[c, 1] * stats.t.ppf(.84, like.nu[1]) * fit.gfit.RADAR_WAVELENGTH_M,
                               fmt='none', color=colors[c], alpha=.4, linewidth=.7)
            if np.shape(snr) == like.measured.shape:
                axes[2, 1].plot(like.t, snr[:, c], '.', color=colors[c])
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('Apparent path offset (km)')
        axes[0, 1].plot(like.t, reference['speed_km_s'], '--', color='.35', label='Whipple-Jacchia')
        axes[0, 1].plot(like.t, speed, color='black', label='Shrinking radius')
        if curves:
            lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
            axes[0, 1].fill_between(like.t, lo, hi, color='.6', alpha=.35)
        axes[0, 1].legend()
        axes[0, 1].set_ylabel('Speed (km/s)')
        axes[1, 0].set_ylabel('Delay residual (m)')
        axes[1, 1].set_ylabel('Beat-equivalent path-rate residual (m/s)')
        for ax in axes[1]:
            ax.axhline(0, color='.4', linewidth=.7)
            ax.text(.02, .97, '68% measurement intervals', transform=ax.transAxes, va='top', fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
        bins = np.linspace(*LOG_RADIUS_BOUNDS, 81)
        axes[2, 0].hist(samples[:, 6] + 6, bins=bins + 6, density=True, histtype='step', color='black')
        qmass = np.percentile(mass, [2.5, 50, 97.5])
        axes[2, 0].set_xlabel(r'$\log_{10}(r_0/\mu\mathrm{m})$')
        axes[2, 0].set_ylabel('Marginal probability density')
        axes[2, 0].text(.03, .97, f'$m_0$ = {qmass[1]:.2g} kg\n95%: [{qmass[0]:.2g}, {qmass[2]:.2g}] kg',
                        transform=axes[2, 0].transAxes, va='top', bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
        axes[2, 1].set_ylabel('Matched-filter SNR (dB)')
        for i, ax in enumerate(axes.flat):
            if i != 4:
                ax.set_xlabel('Time since first detection (s)')
            ax.grid(alpha=.2)
        status = '; '.join(flags).replace('_', ' ') if flags else 'Conditional posterior; chain-length checks passed'
        fig.suptitle(event + '\n' + textwrap.fill(status, width=95), fontsize=9)
        fig.savefig(path.with_suffix('.pdf'))
        fig.savefig(path.with_suffix('.png'), dpi=130)
        plt.close(fig)


def process_event(source_path, output, steps, walkers, seed, max_steps):
    start = time.monotonic()
    event = event_id_from_path(source_path)
    out = Path(output) / ('joint_delay_doppler_fft_' + event)
    if all(out.with_suffix(suffix).exists() for suffix in ('.h5', '.pdf', '.png')):
        return {'event_id': event, 'status': 'existing'}
    try:
        with h5py.File(source_path) as h:
            source = load_group(h['joint_fit'])
            fft = load_group(h['fft_observations'])
        density, density_meta = base.density_interpolator(source['time_ns'], source['x_itrs_m'])
        attach_geodetic_height(density, source['time_ns'])
        density.ceplecha_sigma_kg_j = 1e-8
        density_meta['height_convention'] = 'WGS84 geodetic height using fixed event-epoch GCRS-to-ITRS rotation'
        density_meta['ablation_convention'] = 'sigma=Lambda/(2*Gamma*Q)=1e-8 kg/J; Gamma=1; dm/dt=-sigma*rho*A*v^3'
        reference = refit_reference(source, density)
        print(json.dumps({'event_id': event, 'stage': 'reference_refit_complete'}), flush=True)
        like = EventLikelihood(source, density, reference)
        solutions = []
        bounds = [(-20, 20)] * 3 + [(-80 - c / 1000, 80 - c / 1000) for c in like.center[3:6]]
        bounds += [(-6.99999, -2.00001)] + [(-499, 499)] * 3 + [(-3.999, 3.999)] * 6
        for logr in (-5., -4.5, -4., -3., -2.1):
            q = np.r_[np.zeros(6), logr, reference['station_fft_bias_hz'] / 1000, np.zeros(6)]
            def objective(x):
                value = like(x)
                return -value if np.isfinite(value) else 1e30
            result = optimize.minimize(objective,
                                       q, method='Nelder-Mead', bounds=bounds,
                                       options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-3})
            if np.isfinite(like(result.x)):
                solutions.append(result)
        if not solutions:
            raise RuntimeError('No finite posterior initialization')
        solutions.sort(key=lambda r: r.fun)
        print(json.dumps({'event_id': event, 'stage': 'posterior_initialization_complete', 'seconds': time.monotonic() - start}), flush=True)
        rng = np.random.default_rng(seed + int(hashlib.sha256(event.encode()).hexdigest()[:8], 16))
        positions = []
        # Populate multiple good radius modes, not only the best optimizer basin.
        good = [r for r in solutions if r.fun < solutions[0].fun + 20]
        for i in range(walkers):
            center = good[i % len(good)].x
            for _ in range(1000):
                q = center + rng.normal(size=16) * np.r_[np.full(3, .005), np.full(3, .02), .04, np.full(3, .01), np.full(6, .04)]
                if np.isfinite(like(q)):
                    positions.append(q)
                    break
            else:
                raise RuntimeError('Cannot initialize finite posterior walkers')
        sampler = emcee.EnsembleSampler(walkers, 16, like)
        sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(np.asarray(positions), steps, progress=False)
        while True:
            check_chain = sampler.get_chain(discard=sampler.iteration // 2)
            check_tau = emcee.autocorr.integrated_time(check_chain, quiet=True, tol=0)
            enough = np.all(np.isfinite(check_tau)) and len(check_chain) >= 50 * np.max(check_tau)
            print(json.dumps({'event_id': event, 'stage': 'sampling', 'steps': sampler.iteration,
                              'max_tau': float(np.nanmax(check_tau)), 'long_enough': bool(enough)}), flush=True)
            if enough or sampler.iteration >= max_steps:
                break
            sampler.run_mcmc(None, min(steps, max_steps - sampler.iteration), progress=False)
        steps = sampler.iteration
        burn = steps // 2
        chain = sampler.get_chain(discard=burn)
        tau = emcee.autocorr.integrated_time(chain, quiet=True, tol=0)
        samples = chain[::max(1, len(chain) // 100)].reshape(-1, 16)
        mass = 4 * np.pi * cepl.METEOROID_DENSITY_KG_M3 * 10 ** (3 * samples[:, 6]) / 3
        flags = posterior_flags(chain, tau, mass, samples[:, 6])
        if not solutions[0].success:
            flags.append('map_optimizer_not_converged')
        # Compare halves of each chain to catch obvious persistent mass drift.
        halves = np.split(chain[:2 * (len(chain) // 2), :, 6], 2)
        if abs(np.median(halves[0]) - np.median(halves[1])) > .1:
            flags.append('mass_chain_nonstationary')
        logprob = sampler.get_log_prob(discard=burn)
        q = chain.reshape(-1, 16)[np.argmax(logprob)]
        model, path_resid, beat_resid, scales = like.model(q)
        if np.max(scales) > 3.:
            flags.append('noise_scale_above_three_times_reference')
        path_rms = float(np.sqrt(np.mean(path_resid[like.pk] ** 2)))
        beat_rms = float(np.sqrt(np.mean(beat_resid[like.fk] ** 2)))
        if (path_rms > 2 * max(reference['rms_total_path_residual_m'], 1.)
                or beat_rms > 2 * max(reference['rms_fft_residual_hz'], 1.)):
            flags.append('shrinking_radius_mismatch_relative_to_wj')
        posterior = dict(samples=samples, mass_kg=mass, radius_m=10 ** samples[:, 6],
                         mass_quantiles_kg=np.percentile(mass, [2.5, 50, 97.5]),
                         autocorrelation_time_steps=tau, acceptance_fraction=sampler.acceptance_fraction,
                         noise_correlation=np.asarray([(b[2], b[3]) for b in like.blocks]),
                         flags=';'.join(flags), n_steps=steps, n_walkers=walkers,
                         rms_path_residual_m=path_rms, rms_beat_residual_hz=beat_rms,
                         prior='uniform log10 radius [-7,-2] m; broad bounded state/bias; log noise scale Normal(0,log(3)^2)',
                         limitations='Conditional on empirical noise correlation, fixed MSIS density, bulk density 3000 kg/m3, drag Gamma=1 and ablation sigma=1e-8. No unconditional coverage claim.',
                         likelihood='Gaussian copula with Student-t marginals, separable within-station time/channel correlation; independent stations')
        joint = dict(params=like.unpack(q)[0], time_ns=like.times, t_rel_s=like.t,
                     measured_total_paths_m=like.measured, observed_fft_beat_hz=like.beat,
                     path_keep=like.pk, fft_keep=like.fk, path_sigma_m=like.path_sigma,
                     fft_sigma_hz=like.fft_sigma, station_fft_bias_hz=like.unpack(q)[1],
                     noise_scale_multiplier=scales, path_residuals_m=path_resid,
                     fft_residuals_hz=beat_resid, **model)
        with h5py.File(out.with_suffix('.partial.h5'), 'w') as h:
            h.attrs['version'] = VERSION
            h.attrs['source_event_h5'] = str(source_path)
            h.attrs['source_event_sha256'] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
            h.attrs['observation_provenance'] = 'Cached per-pulse matched-filter delays and dechirped beat frequencies; source acceptance masks preserved. Raw voltages not re-extracted in this validation run.'
            h.attrs['elapsed_s'] = time.monotonic() - start
            h.attrs['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            h.attrs['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            import meteor_trajectory_models.ceplecha as ode
            h.attrs['ode_source_sha256'] = hashlib.sha256(Path(ode.__file__).read_bytes()).hexdigest()
            h.attrs['drag_convention'] = 'free molecular Gamma=1, equivalent aerodynamic Cd=2; spherical acceleration coefficient 3/4'
            save_group(h.create_group('joint_fit'), joint)
            save_group(h.create_group('wj_reference'), reference)
            save_group(h.create_group('mass_posterior'), posterior)
            save_group(h.create_group('msis'), density_meta)
            h['msis']['altitude_km'] = base.MSIS_ALT_GRID_KM
            h['msis']['density_kg_m3'] = density(base.MSIS_ALT_GRID_KM * 1000.)
        out.with_suffix('.partial.h5').replace(out.with_suffix('.h5'))
        event_plot(out, event, like, reference, q, samples, flags, fft.get('fft_snr_db', []))
        return {'event_id': event, 'status': 'done', 'flags': ';'.join(flags), 'elapsed_s': time.monotonic() - start}
    except Exception as exc:
        import traceback
        error = traceback.format_exc()
        out.with_suffix('.error.log').write_text(error)
        return {'event_id': event, 'status': 'error', 'error': str(exc), 'elapsed_s': time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--event-id', action='append')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--max-steps', type=int, default=16000)
    parser.add_argument('--walkers', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob('joint_delay_doppler_fft_tri_*.h5'))
    if args.event_id:
        paths = [p for p in paths if event_id_from_path(p) in args.event_id]
    if not paths:
        raise RuntimeError('No input event files')
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(process_event, p, args.output_dir, args.steps, args.walkers, args.seed, args.max_steps) for p in paths]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
            with h5py.File(args.output_dir / 'run_status.partial.h5', 'w') as h:
                h.attrs['n_requested'] = len(paths)
                h.attrs['n_finished'] = len(rows)
                h.create_dataset('event_id', data=[r['event_id'] for r in rows], dtype=h5py.string_dtype())
                h.create_dataset('status', data=[r['status'] for r in rows], dtype=h5py.string_dtype())
            (args.output_dir / 'run_status.partial.h5').replace(args.output_dir / 'run_status.h5')


if __name__ == '__main__':
    main()
