from voltage_review.server import Store


def test_physical_catalogue_never_falls_back_to_wj(tmp_path):
    event='tri_0000_1713791855379348755'
    root=tmp_path/'raw';refits=tmp_path/'wj';latest=tmp_path/'physical'
    for parent in (root,refits):
        (parent/event).mkdir(parents=True)
        (parent/event/'plot.h5').touch()
    store=Store(root,tmp_path/'flags',refits=refits,latest=latest)
    assert store.path(event)==latest/event
    assert not (store.path(event)/'plot.h5').exists()


def test_physical_latest_has_priority_over_physical_previous(tmp_path):
    event='tri_0000_1713791855379348755'
    previous=tmp_path/'previous';latest=tmp_path/'latest'
    (previous/event).mkdir(parents=True)
    (previous/event/'plot.h5').touch()
    store=Store(tmp_path/'raw',tmp_path/'flags',latest=latest,previous=previous)
    assert store.path(event)==previous/event
    (latest/event).mkdir(parents=True)
    (latest/event/'plot.h5').touch()
    assert store.path(event)==latest/event
