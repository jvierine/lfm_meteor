"""Retain RCS panels a/b; use the mass-distribution sample exactly for c."""
from pathlib import Path
import h5py
import numpy as np

root = Path(__file__).resolve().parent
output = root / 'results/current_review_mass_comparison/rcs_panels.h5'
output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(root / 'results/sanya_rcs_from_snr.h5') as original, h5py.File(output, 'w') as out:
    pulses = out.create_group('pulses')
    for key in original:
        original.copy(key, pulses)
    source = root / 'results/current_dynamic_mass_distribution/figure6.h5'
    with h5py.File(source) as masses:
        group = out.create_group('mass_comparison')
        names = masses['event_id'].asstr()[:]
        rcs_names = masses['rcs_event_id'].asstr()[:]
        index = {name: i for i, name in enumerate(names)}
        assert len(index) == len(names) and len(set(rcs_names)) == len(rcs_names)
        selected = np.array([index[name] for name in rcs_names], dtype=int)
        assert np.all(masses['selected'][:][selected]), 'RCS sample contains rejected dynamic masses'
        group['event_id'] = np.asarray(rcs_names, dtype=h5py.string_dtype())
        for target, key in [('dynamic_mass_kg', 'initial_mass_kg'),
                            ('dynamic_lo95_kg', 'mass_95_lo_kg'),
                            ('dynamic_hi95_kg', 'mass_95_hi_kg')]:
            group[target] = masses[key][:][selected]
        for target, key in [('rcs_mass_kg', 'rcs_mass_kg'),
                            ('delta_mass_lo95_noise_only_kg', 'rcs_lo95_kg'),
                            ('delta_mass_hi95_noise_only_kg', 'rcs_hi95_kg')]:
            group[target] = masses[key][:]
        for key in group:
            if key != 'event_id':
                values = group[key][:]
                assert np.all(np.isfinite(values) & (values > 0)), key
        group.attrs.update(selection_source=str(source), selection=masses.attrs['selection'],
                           sample_count=len(rcs_names),
                           rcs_meaning=masses.attrs['rcs_meaning'])
    out.attrs['source_program'] = str(Path(__file__).resolve())
print(output)
