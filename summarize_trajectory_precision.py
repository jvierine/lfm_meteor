"""Summarize reported speed intervals for the residual-histogram event set."""
import argparse
from pathlib import Path
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('catalogue', type=Path)
p.add_argument('selection', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
with h5py.File(a.selection) as h:
    events = h['events'].asstr()[:]
    residual = h['modeled_path_m'][:]-h['seed_apparent_path_m'][:]
medians, methods = [], []
for event in events:
    with h5py.File(a.catalogue/event/'plot.h5') as h:
        _, indices = np.unique(h['speed_time_s'][:], return_index=True)
        halfwidth = 500*(h['speed_upper95_km_s'][:]-h['speed_lower95_km_s'][:])
        assert np.all(np.isfinite(halfwidth)) and np.all(halfwidth >= 0)
        medians.append(np.median(halfwidth[indices]))
        methods.append(str(h.attrs.get('uncertainty_method', h.attrs.get('uncertainty_status','unspecified'))))
q = np.quantile(medians,[.25,.5,.75])
with h5py.File(a.output,'w') as h:
    h.create_dataset('event_id',data=events,dtype=h5py.string_dtype())
    h.create_dataset('uncertainty_method',data=methods,dtype=h5py.string_dtype())
    h['event_median_speed_interval_halfwidth_m_s'] = medians
    h['event_halfwidth_quartiles_m_s'] = q
    h.attrs.update(n_events=len(events), n_measurements=len(residual),
        apparent_path_rms_m=np.sqrt(np.mean(residual**2)),
        definition='Half reported pointwise 95% speed interval width; median over unique times per event, then event quartiles. Not 1-sigma error or validated total uncertainty.',
        source_program=Path(__file__).read_text(),selection=str(a.selection),catalogue=str(a.catalogue))
print('Speed halfwidth quartiles (m/s):',q)
