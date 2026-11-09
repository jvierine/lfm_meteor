"""Build a current, checksummed reviewer release without changing scientific fits."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

import h5py
import numpy as np

from export_submission_l1 import state_vectors


CODE = Path(__file__).resolve().parent
RUN_COMMIT = '2328909207f68982b8aed4b19fb3046f61fe933c'
S = h5py.string_dtype()


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory_bytes(records, **attrs):
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as h:
        for index, name in enumerate(('path', 'sha256', 'bytes', 'origin')):
            values = [r[index] for r in records]
            h[name] = values if name == 'bytes' else np.asarray(values, dtype=S)
        h.attrs.update(**attrs)
    return buffer.getvalue()


class Archive:
    def __init__(self, path):
        self.tar = tarfile.open(path, 'w:gz', compresslevel=3)
        self.records, self.names = [], set()

    def add(self, name, blob, origin):
        if name in self.names:
            raise ValueError(f'Duplicate archive member: {name}')
        self.names.add(name)
        info = tarfile.TarInfo(name)
        info.size = len(blob)
        info.mode = 0o644
        self.tar.addfile(info, io.BytesIO(blob))
        self.records.append((name, hashlib.sha256(blob).hexdigest(), len(blob), origin))

    def file(self, name, path):
        if path.is_symlink():
            raise ValueError(f'External/symlink input: {path}')
        self.add(name, path.read_bytes(), str(path))

    def tree(self, source, prefix, allowed=('.h5', '.log', '.pdf', '.png', '.tex', '.svg')):
        for p in sorted(source.rglob('*')):
            if (p.is_file() and p.suffix in allowed and
                not any(x.startswith('.') or x.startswith('cache') or x == 'full_prediction'
                        for x in p.relative_to(source).parts)):
                self.file(prefix+'/'+p.relative_to(source).as_posix(), p)

    def close(self, manifest):
        self.add(manifest, inventory_bytes(self.records, version='2026-11-09'), 'generated member manifest')
        self.tar.close()


def git_snapshot(archive, repo, commit, prefix, working=False):
    data = subprocess.check_output(['git', 'archive', commit], cwd=repo)
    overrides = []
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        for member in source:
            if not member.isfile() or member.name.startswith(('results/', 'tmp/')):
                continue
            path = Path(member.name)
            if path.suffix not in ('.py', '.rs', '.toml', '.lock', '.md', '.txt', '.js', '.html', '.css', '.cjs', '.sh', '.metal'):
                continue
            blob = source.extractfile(member).read()
            if working and (repo/path).is_file():
                current = (repo/path).read_bytes()
                if blob != current:
                    overrides.append(member.name)
                    blob = current
            archive.add(prefix+'/'+member.name, blob, f'git:{repo.name}@{commit}; working={working}')
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging.resolve()
    output = staging/'upload'
    output.mkdir(exist_ok=False)
    baseline = CODE/'results/submission_staging/20261107-consolidated'
    current = CODE/'results/tristatic_calibrated_likelihood_20261109'
    kin = CODE/'results/tristatic_seven_parameter_catalogue_20261108'
    remote = staging/'input'
    with h5py.File(current/'mass_selection.h5') as h:
        events = h['events'].asstr()[:].tolist()
        included = dict(zip(events, h['dynamic_mass_included'][:]))
        statuses = dict(zip(events, h['status'].asstr()[:]))
    assert len(events) == 234 and sum(included.values()) == 135
    with h5py.File(kin/'orbits/orbits.h5') as h:
        orbit_events = h['event_id'].asstr()[:].tolist()
    assert len(orbit_events) == 216

    result = Archive(output/'sanya-results-20261109.tar.gz')
    original_inputs = {}
    with tarfile.open(baseline/'sanya-results.tar.gz') as source:
        for member in source:
            if not member.isfile():
                continue
            if member.name.startswith(('lookup/', 'support/original_inputs/')):
                blob = source.extractfile(member).read()
                result.add(member.name, blob, 'unchanged baseline supporting input')
                if member.name.startswith('support/original_inputs/'):
                    original_inputs[Path(member.name).stem] = blob
    # Canonical calibrated products, with their actual frozen inputs and flags.
    prefix = 'results/tristatic_calibrated_likelihood_20261109'
    result.tree(remote/'calibrated', prefix, allowed=('.h5', '.log'))
    present = {r[0] for r in result.records}
    for p in sorted(current.glob('*.h5')):
        name = prefix+'/'+p.name
        if name not in present:
            result.file(name, p)
    result.tree(current/'products', prefix+'/products')
    result.tree(current/'precision_audit', prefix+'/precision_audit_local')
    # Retain only current kinematic inputs, not superseded mass products.
    omitted = {'mass_distribution.h5', 'current_mass_influx.h5', 'rcs_figure.h5',
               'velocity_exponent_influx.h5'}
    for p in sorted((kin/'products').glob('*.h5')):
        if p.name not in omitted:
            result.file('results/tristatic_seven_parameter_catalogue_20261108/products/'+p.name, p)
    result.tree(kin/'products/delay_residuals', 'results/tristatic_seven_parameter_catalogue_20261108/products/delay_residuals')
    result.tree(kin/'orbits', 'orbits')
    result.tree(kin/'orbits', 'results/tristatic_seven_parameter_catalogue_20261108/orbits')

    for event in events:
        folder = remote/'catalog'/event
        if not folder.is_dir():
            raise ValueError(f'Missing current event: {event}')
        input_path = folder/'input.h5'
        with h5py.File(input_path) as h:
            center = h['center'][:]
            source_names = str(h.attrs['source_event_files']).split(';')
        assert sha(input_path) == sha(remote/'calibrated/inputs'/event/'input.h5')
        confidence = current/'confidence'/event/'confidence.h5'
        if included[event]:
            assert confidence.is_file()
            with h5py.File(confidence) as h:
                assert h.attrs['source_input_sha256'] == sha(input_path)
        for p in sorted(folder.glob('*.h5')):
            if p.name in ('dynamic_mass_posterior.h5', 'complete.h5'):
                continue
            blob = p.read_bytes()
            if p.name == 'plot.h5':
                buffer = io.BytesIO(blob)
                with h5py.File(buffer, 'r+') as h:
                    for key in ('initial_radius_m_quantiles', 'initial_mass_kg_quantiles'):
                        if key in h:
                            del h[key]
                    for key in ('best_initial_mass_kg', 'best_initial_radius_m'):
                        if key in h.attrs:
                            del h.attrs[key]
                    h.attrs.update(dynamic_mass_included=bool(included[event]),
                        dynamic_mass_status=statuses[event],
                        dynamic_mass_source='mass_confidence.h5 and calibrated mass_selection.h5 only',
                        trajectory_band_interpretation='Conditional posterior pointwise bands; not mass-confidence projections')
                blob = buffer.getvalue()
            elif p.name == 'physical_states.h5':
                if event not in original_inputs:
                    raise ValueError(f'Missing full-pulse frame input: {event}')
                with h5py.File(io.BytesIO(original_inputs[event])) as original:
                    frames = original['frames'][:]
                    times = original['time_ns'][:]
                    stations = original['station'][:]
                with h5py.File(folder/'fit.h5') as fit:
                    params = fit['params'][:]
                buffer = io.BytesIO(blob)
                with h5py.File(buffer, 'r+') as h:
                    index = {(int(t),int(s)):i for i,(t,s) in enumerate(zip(times,stations))}
                    indices = [index[int(t),int(s)] for t,s in zip(h['time_ns'][:],h['station'][:])]
                    _, _, velocity, error = state_vectors(center, params, frames[indices],
                        h['position_ecef_m'][:], h['speed_km_s'][:]*1000)
                    h['velocity_ecef_m_s'] = velocity
                    h['velocity_ecef_m_s'].attrs['units'] = 'm s-1'
                    h.attrs.update(velocity_type='Inferred nominal-time ECEF state, not an independent measurement',
                                   vector_reconstruction_transverse_error_m=error)
                blob = buffer.getvalue()
            result.add(f'L1/events/{event}/{p.name}', blob, str(p))
        if confidence.exists():
            result.file(f'L1/events/{event}/mass_confidence.h5', confidence)
    for event in orbit_events:
        p = remote/'trajectory_posteriors'/event/'posterior.h5'
        with h5py.File(p) as h:
            assert h.attrs['convergence_verified']
        result.file(f'uncertainty/trajectory/{event}/posterior.h5', p)
    result.close('release-inventory.h5')

    code = Archive(output/'sanya-code-20261109.tar.gz')
    with tarfile.open(baseline/'sanya-code.tar.gz') as source:
        for member in source:
            if member.isfile() and member.name.startswith(('source/fdtd/', 'source/rustplotlib/')):
                code.add(member.name, source.extractfile(member).read(), 'unchanged archived dependency snapshot')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
    overrides = git_snapshot(code, CODE, head, 'source/lfm_meteor', working=True)
    git_snapshot(code, CODE, RUN_COMMIT, 'source/calibrated-fit-run')
    for name in ('verify_current_release.py', 'README-current.txt'):
        code.file('source/reproduction/'+name, CODE/'reproduction'/name)
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as h:
        h.attrs.update(current_code_commit=head, mass_confidence_run_commit=RUN_COMMIT,
                       working_source_files_archived=True, draft_only=True)
        h['working_tree_overrides'] = np.asarray(overrides, dtype=S)
    code.add('source/provenance.h5', buffer.getvalue(), 'generated exact-source provenance')
    code.close('source/inventory.h5')
    (output/'README-20261109.txt').write_bytes((CODE/'reproduction/README-current.txt').read_bytes())
    (output/'read_sanya_zenodo.py').write_bytes((CODE/'read_sanya_zenodo.py').read_bytes())
    downloads = [baseline/'sanya-raw-data.tar.gz', output/'sanya-results-20261109.tar.gz',
                 output/'sanya-code-20261109.tar.gz', output/'README-20261109.txt', output/'read_sanya_zenodo.py']
    with h5py.File(output/'checksums-20261109.h5', 'x') as h:
        h['filename'] = np.asarray([p.name for p in downloads], dtype=S)
        h['sha256'] = np.asarray([sha(p) for p in downloads], dtype=S)
        h['md5'] = np.asarray([sha(p,'md5') for p in downloads], dtype=S)
        h['bytes'] = [p.stat().st_size for p in downloads]
        h.attrs.update(record_id='24037714', release_date='2026-11-09',
                       trajectory_events=234, dynamic_selected=135, rcs_selected=197,
                       paired_masses=115, verified_orbit_events=216,
                       raw_archive_unchanged=True, intended_state='Unpublished restricted reviewer draft')
    for p in sorted(output.iterdir()):
        print(p.name, p.stat().st_size, sha(p,'md5'), flush=True)


if __name__ == '__main__':
    main()
