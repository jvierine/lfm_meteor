"""Assess the six-event uncertainty pilot without changing canonical products."""
import argparse
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.stats import binomtest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    phases=[('noise_regions',24),('scatter_regions',24),('weak_heldout_regions_v2',16)]
    with h5py.File(a.root/'assessment.h5','w') as out:
        out.attrs.update(pilot_only=True,canonical_products_unchanged=True,
            Bayesian_credible_intervals=False,nominal_joint_confidence=.95,
            mass_projection_conservative=True,physical_systematic_errors_included=False,
            experiments_noise_and_scatter_use_paired_random_seeds=True,
            heldout_weak_deceleration_seeds_independent=True,
            material_density_kg_m3=3000.,Gamma=1.,sigma_s2_m2=1e-8,
            source_code_repository='https://github.com/jvierine/lfm_meteor',
            source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        covered=joint=total=0
        for phase,n in phases:
            paths=sorted((a.root/phase).glob('sim_*/confidence.h5'))
            if len(paths)!=n:raise ValueError(f'{phase}: missing simulation results')
            mass_flags=[];joint_flags=[];checks=[]
            for path in paths:
                with h5py.File(path) as h:
                    mass_flags.append(bool(h.attrs['mass_projection_covered']))
                    joint_flags.append(bool(h.attrs['joint_truth_covered']))
                    checks.append(bool(h.attrs['bootstrap_numerical_checks_passed']))
            if not all(checks):raise ValueError(f'{phase}: unresolved numerical fits')
            g=out.create_group('validation/'+phase)
            g['case_id']=np.asarray([p.parent.name for p in paths],dtype=h5py.string_dtype())
            g['mass_covered']=mass_flags;g['joint_covered']=joint_flags
            ci=binomtest(sum(mass_flags),n).proportion_ci(method='exact')
            g.attrs.update(cases=n,mass_covered=sum(mass_flags),joint_covered=sum(joint_flags),
                mass_coverage_exact_lower95=ci.low,mass_coverage_exact_upper95=ci.high)
            covered+=sum(mass_flags);joint+=sum(joint_flags);total+=n
        out.attrs.update(simulation_cases=total,mass_cases_covered=covered,joint_cases_covered=joint)
        events=sorted((a.root/'real_regions_warm').glob('tri_*/confidence.h5'))
        if len(events)!=6:raise ValueError('Expected six complete real-event profiles')
        for path in events:
            with h5py.File(path) as h:
                lo,hi=h['mass_confidence_interval_kg'][:]
                best=4*np.pi/3*3000*10**(3*h['best_params'][6])
                if not lo<=best<=hi:raise ValueError('Best mass outside confidence projection')
                finite=lo>0 and np.isfinite(hi)
                width=np.log10(hi/lo) if finite else np.inf
                selected=finite and width<1.
                reason=('Finite confidence interval narrower than one order of magnitude' if selected
                    else 'Confidence interval spans at least one order of magnitude' if finite
                    else 'Upper or lower mass bound unresolved')
                g=out.create_group('events/'+path.parent.name)
                g['mass_confidence_interval_kg']=[lo,hi]
                g.attrs.update(best_mass_kg=best,interval_width_orders_of_magnitude=width,
                    suggested_mass_quality_selection=selected,selection_reason=reason,
                    mass_selection_only_not_trajectory_rejection=True,
                    source_input_sha256=h.attrs['source_input_sha256'],
                    profile_source_git_commit=h.attrs['source_git_commit'])
                print(path.parent.name,'bounds',lo,hi,'mass_eligible',selected,flush=True)
        print('SIMULATIONS',covered,'mass and',joint,'joint of',total,flush=True)


if __name__=='__main__':main()
