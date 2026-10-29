import h5py
import numpy as np
import pytest
from voltage_review.server import Store


@pytest.fixture
def store(tmp_path):
    event=tmp_path/'data'/'tri_0003_1713794570204349518';event.mkdir(parents=True)
    with h5py.File(event/'plot.h5','w') as h:
        h['station']=[0,1];h['time_ns']=np.array([1713794570204349518,1713794570204349519],dtype=np.int64)
        h['time_s']=[0.,.01];h.attrs['beat_guard_pass']=False
    return Store(tmp_path/'data',tmp_path/'flags')


def payload(data):
    return dict(version=data['annotation']['version'],fingerprint=data['fingerprint'],excluded=[data['ids'][0]],bad=False,note='early pulse')


def test_persistence_exact_ns_and_source_unchanged(store):
    e='tri_0003_1713794570204349518';source=(store.path(e)/'plot.h5').read_bytes()
    d=store.data(e);assert d['time_ns']==['1713794570204349518','1713794570204349519']
    assert store.save(e,payload(d))['version']==1
    fresh=Store(store.root,store.flags);assert fresh.data(e)['annotation']['excluded']==[d['ids'][0]]
    assert (store.path(e)/'plot.h5').read_bytes()==source
    with h5py.File(store.flags/(e+'.h5')) as h:assert h['manual_reject_mask'][:].tolist()==[True,False]


def test_conflict_unknown_ids_and_history(store):
    e='tri_0003_1713794570204349518';d=store.data(e);p=payload(d);store.save(e,p)
    with pytest.raises(RuntimeError):store.save(e,p)
    p=payload(store.data(e));p['excluded']=['0:123']
    with pytest.raises(ValueError):store.save(e,p)
    p['excluded']=[];store.save(e,p)
    with h5py.File(store.flags/(e+'.h5')) as h:assert list(h['history'])==['1','2']


def test_data_change_and_traversal_rejected(store):
    e='tri_0003_1713794570204349518';p=payload(store.data(e));p['fingerprint']='wrong'
    with pytest.raises(RuntimeError):store.save(e,p)
    with pytest.raises(ValueError):store.data('../secret')


def test_model_bands_preserve_saved_measurement_fingerprint(store):
    e='tri_0003_1713794570204349518';d=store.data(e)
    with h5py.File(store.path(e)/'plot.h5','a') as h:
        for key in ('path_lower95_m','path_upper95_m','speed_lower95_km_s','speed_upper95_km_s'):
            h[key]=[1.,2.]
        h.attrs['uncertainty_status']='95% pointwise model interval'
    assert store.data(e)['fingerprint']==d['fingerprint']
    assert store.save(e,payload(d))['version']==1


def test_excluded_point_remains_selectable_after_refit(store):
    e='tri_0003_1713794570204349518'
    with h5py.File(store.path(e)/'plot.h5','a') as h:h['seed_apparent_path_m']=[100.,101.]
    original=store.data(e);store.save(e,payload(original))
    refits=store.root.parent/'refits';folder=refits/e;folder.mkdir(parents=True)
    with h5py.File(folder/'plot.h5','w') as h:
        h['station']=[1];h['time_ns']=np.array([1713794570204349519],dtype=np.int64)
        h['time_s']=[.01];h['seed_apparent_path_m']=[101.];h.attrs['beat_guard_pass']=True
    reviewed=Store(store.root,store.flags,refits);d=reviewed.data(e)
    assert d['omitted_measurements'][0]['id']==original['ids'][0]
    p=payload(d);p['excluded']=[]
    assert reviewed.save(e,p)['excluded']==[]
    with h5py.File(store.flags/(e+'.h5')) as h:assert len(h['manual_reject_mask'])==2
