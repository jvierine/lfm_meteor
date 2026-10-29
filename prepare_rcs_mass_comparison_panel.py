"""Retain the paper RCS panels and attach the current matched mass snapshot."""
from pathlib import Path
import h5py

root = Path(__file__).resolve().parent
output = root / 'results/current_review_mass_comparison/rcs_panels.h5'
output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(root / 'results/sanya_rcs_from_snr.h5') as original, h5py.File(output, 'w') as out:
    pulses = out.create_group('pulses')
    for key in original:
        original.copy(key, pulses)
    with h5py.File(root / 'results/current_review_mass_comparison/scatter.h5') as masses:
        group = out.create_group('mass_comparison')
        for key in masses:
            masses.copy(key, group)
        for key, value in masses.attrs.items():
            group.attrs[key] = value
    out.attrs['source_program'] = str(Path(__file__).resolve())
print(output)
