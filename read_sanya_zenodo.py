"""Read extracted Sanya Zenodo HDF5 products and plot one event and the orbits.

Install: pip install numpy h5py matplotlib
Run: python read_sanya_zenodo.py /path/to/extracted/data --event tri_0012
Extract sanya-results.tar.gz first. No raw-voltage processing is needed.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def read_event(root, event):
    matches = sorted((root / 'L1/events').glob(event + '*'))
    if len(matches) != 1:
        raise ValueError(f'Expected one event matching {event}; found {len(matches)}')
    folder = matches[0]
    result = {}
    for name in ('beam_rcs', 'physical_states'):
        with h5py.File(folder / (name + '.h5')) as f:
            result[name] = {k: f[k][:] for k in f if isinstance(f[k], h5py.Dataset)}
    return folder.name, result


def plot_event(name, data, output):
    beam, state = data['beam_rcs'], data['physical_states']
    origin = min(beam['time_ns'].min(), state['time_ns'].min())
    tb = (beam['time_ns'] - origin) / 1e9
    ts = (state['time_ns'] - origin) / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    axes[0, 0].plot(tb, beam['snr_db'], '.-', color='tab:blue')
    axes[0, 0].set(ylabel='Sanya SNR (dB)', xlabel='Time from first sample (s)')
    axes[0, 1].plot(tb, beam['rcs_dbsm'], '.-', color='tab:orange')
    axes[0, 1].set(ylabel='Sanya RCS (dBsm)', xlabel='Time from first sample (s)')
    # The same physical time can occur at multiple stations; plot it once.
    _, indices = np.unique(state['time_ns'], return_index=True)
    axes[1, 0].plot(ts[indices], state['speed_km_s'][indices], '.-')
    axes[1, 0].set(ylabel='Fitted speed (km/s)', xlabel='Time from first sample (s)')
    position = state['position_ecef_m'][indices]
    displacement = (position - position[0]) / 1000
    for j, label in enumerate(('ECEF X', 'ECEF Y', 'ECEF Z')):
        axes[1, 1].plot(ts[indices], displacement[:, j], label=label)
    axes[1, 1].set(ylabel='Displacement from first position (km)',
                   xlabel='Time from first sample (s)')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(output / (name + '.png'), dpi=180)
    return fig


def plot_orbits(root, output):
    with h5py.File(root / 'orbits/orbits.h5') as f:
        valid = f['valid'][:].astype(bool)
        values = {k: np.where(valid, f[k][:], np.nan)
                  for k in ('a_au', 'e', 'q_au', 'i_deg')}
        probability = f['cometary_probability_bound'][:]
        names = f['event_id'].asstr()[:]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    colors = np.where(probability >= .5, 'tab:blue', 'tab:orange')
    for ax, xkey, ykey, xlabel, ylabel in (
        (axes[0], 'a_au', 'e', 'Semimajor axis (AU)', 'Eccentricity'),
        (axes[1], 'q_au', 'i_deg', 'Perihelion distance (AU)', 'Inclination (deg)')):
        xlo, x, xhi = np.nanquantile(values[xkey], [.025, .5, .975], axis=1)
        ylo, y, yhi = np.nanquantile(values[ykey], [.025, .5, .975], axis=1)
        for i in range(len(names)):
            ax.errorbar(x[i], y[i], xerr=[[x[i]-xlo[i]], [xhi[i]-x[i]]],
                        yerr=[[y[i]-ylo[i]], [yhi[i]-y[i]]], fmt='.',
                        color=colors[i], alpha=.65, elinewidth=.5)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
    fig.savefig(output / 'orbits.png', dpi=180)
    print(f'Read {len(names)} orbit ensembles; blue: cometary probability >= 0.5; orange: < 0.5.')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--event', default='tri_0012')
    parser.add_argument('--output', type=Path, default=Path('sanya-plots'))
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    name, data = read_event(args.root, args.event)
    plot_event(name, data, args.output)
    plot_orbits(args.root, args.output)
    print(f'Saved event curves, trajectory components, and orbit plots in {args.output.resolve()}')
    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
