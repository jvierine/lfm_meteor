"""Join independently released RCS pulses and mass comparisons without edits."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from prepare_current_kinematics import OUTPUT, sha, accepted_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mass_comparison',type=Path)
    args = parser.parse_args()
    pulses = OUTPUT.parent/'rcs_pulses.h5'
    target = OUTPUT.parent/'sanya_rcs_from_snr_article.h5'
    before = sha(args.mass_comparison)
    with h5py.File(pulses) as p, h5py.File(args.mass_comparison) as m, h5py.File(target,'w') as out:
        if not set(m['mass_comparison/event_id'].asstr()[:]).issubset(accepted_ids()):
            raise ValueError('Mass comparison includes an event outside the shared catalogue')
        p.copy('pulses',out)
        m.copy('mass_comparison',out)
        out.attrs.update(pulses_source=str(pulses),pulses_sha256=sha(pulses),
                         mass_comparison_source=str(args.mass_comparison.resolve()),
                         mass_comparison_sha256=before,source_program=__file__)
        for key,dataset in m['mass_comparison'].items():
            np.testing.assert_array_equal(dataset[()],out['mass_comparison/'+key][()])
    if sha(args.mass_comparison) != before:
        raise ValueError('Mass source changed during assembly')
    print(target,flush=True)


if __name__=='__main__':
    main()
