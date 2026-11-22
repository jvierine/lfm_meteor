"""Package current accepted mass extremes with their existing event plots."""
import argparse
from pathlib import Path
import shutil
import h5py
import numpy as np
from analyze import save


def scientific(value):
    exponent=int(np.floor(np.log10(value)))
    return r'%.3g\times10^{%d}' % (value/10**exponent,exponent)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_tristatic'))
    a=p.parse_args()
    with h5py.File(a.root/'catalogue_summary.h5') as h:
        names=[n.decode() for n in h['event_id'][:]]
    rows=[]
    for name in names:
        folder=a.root/'events'/name
        if not (folder/'physical.h5').exists():continue
        with h5py.File(folder/'physical.h5') as h:
            if not h.attrs.get('mass_resolved_profile',False):continue
            mass=float(h['initial_mass_kg'][()])
            interval=h['mass_local95_kg'][:]
        rows.append((mass,name,interval))
    rows.sort()
    selected=list(reversed(rows[-3:]))+rows[:10]
    assert len(selected)==13
    out=a.root/'extreme_mass_atlas';out.mkdir(exist_ok=True)
    text=[r'\documentclass[10pt]{article}',r'\usepackage[a4paper,margin=12mm]{geometry}',
          r'\usepackage{graphicx,amsmath}',
          r'\graphicspath{{memos/november2026/extreme_mass_atlas/}{}}',
          r'\pagestyle{plain}',r'\begin{document}']
    for i,(mass,name,interval) in enumerate(selected):
        label=('Largest %d'%(i+1)) if i<3 else ('Smallest %d'%(i-2))
        source=a.root/'figures/events'/name/'event.pdf'
        if not source.exists():raise FileNotFoundError(source)
        shutil.copy2(source,out/(name+'.pdf'))
        heading=r'\noindent{\Large\bfseries %s: %s}\par\medskip' % (label,name.replace('_',r'\_'))
        text.extend([heading,
            r'\noindent{\large Dynamic initial mass: $m_0=%s\ \mathrm{kg}$}\par' % scientific(mass),
            r'\noindent Conditional 95\%% interval: $[%s,\ %s]\ \mathrm{kg}$.\par\medskip' % tuple(scientific(v) for v in interval),
            r'\noindent Coloured measurements constrain the dynamic fit; grey measurements are excluded.\par\medskip',
            r'\noindent\includegraphics[width=\linewidth,height=0.84\textheight,keepaspectratio]{%s.pdf}' % name,
            r'\clearpage'])
        print(label,name,'mass_kg',mass,'interval',interval,flush=True)
    text.append(r'\end{document}')
    (out/'extreme_masses.tex').write_text('\n'.join(text)+'\n')
    save(out/'selection.h5',dict(event_id=np.array([r[1] for r in selected],dtype='S32'),
         initial_mass_kg=np.array([r[0] for r in selected]),
         interval_95_kg=np.array([r[2] for r in selected])),
         {'selection':'three largest, then ten smallest accepted dynamic masses; current catalogue'})


if __name__=='__main__':main()
