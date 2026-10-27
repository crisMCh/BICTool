"""Tests `MainViewModel` (the workflow between the ViewModels) without any
widgets - synthetic volumes and results only."""

import sys
from types import SimpleNamespace

from bictool.presentation.viewmodels.main_viewmodel import MainViewModel
from bictool.presentation.viewmodels.roi_selection import RoiRect
from bictool.presentation.views.main_window import _default_tissue_settings

_WEIGHTS_CONFIG = {"segmentation": {"anatomy_weights_path": "C:/weights.hdf5"}}


def _volume(num_slices=3):
    return SimpleNamespace(num_slices=num_slices, study_id="S1", patient_id="P", pixel_spacing_mm=0.5)


def _vm(config=None):
    vm = MainViewModel(lambda job: None, config or {}, {"Brain": ["Brain Parenchyma"]}, lambda: None, _default_tissue_settings())
    opened, closed, segmented = [], [], []
    vm.series_opened.connect(lambda volume, primary: opened.append((volume, primary)))
    vm.study_closed.connect(lambda: closed.append(True))
    vm.slice_segmented.connect(segmented.append)
    return vm, opened, closed, segmented


def _fill_series_state(vm):
    """Results in every area, as after working on a series."""
    vm.segmentation.cache = {0: "masks"}
    vm.stroke.cache = {0: "lesion"}
    vm.brain_tissue.cache = {0: "tissue"}
    vm.roi.store.set_manual(0, "falx.roi1", RoiRect(10, 10, 30, 30))
    vm.roi.auto_on = True
    vm.qa.results = {(0, "falx"): "result"}
    vm.export.last_result = "result"


def _assert_series_state_empty(vm):
    assert vm.segmentation.cache == {}
    assert vm.stroke.cache == {}
    assert vm.brain_tissue.cache == {}
    assert vm.roi.effective(0) == {} and not vm.roi.auto_on and not vm.roi.confirmed
    assert vm.qa.results == {}
    assert vm.export.last_result is None


def test_opening_a_series_resets_every_area_and_tells_the_view():
    vm, opened, _closed, _segmented = _vm()
    vm.study.adopt_volume(_volume())
    _fill_series_state(vm)

    second = _volume(5)
    vm.study.adopt_volume(second, is_primary_series=False)

    _assert_series_state_empty(vm)
    assert opened[-1] == (second, False)


def test_returning_to_the_primary_series_restores_its_segmentation():
    vm, _opened, _closed, _segmented = _vm()
    vm.study.adopt_volume(_volume())
    vm.segmentation.cache = {0: "primary masks"}

    vm.study.adopt_volume(_volume(), is_primary_series=False)
    assert vm.segmentation.cache == {}
    vm.study.adopt_volume(_volume())

    assert vm.segmentation.cache == {0: "primary masks"}


def test_reloading_the_primary_series_drops_its_kept_segmentation():
    vm, _opened, _closed, _segmented = _vm()
    vm.study.adopt_volume(_volume())
    vm.segmentation.cache = {0: "primary masks"}
    vm.study.adopt_volume(_volume(), is_primary_series=False)

    vm.study.adopt_volume(_volume(), force_reload=True)

    assert vm.segmentation.cache == {}
    assert vm.segmentation.primary_cache is None


def test_closing_the_study_resets_everything_and_forgets_the_primary():
    vm, _opened, closed, _segmented = _vm()
    vm.study.adopt_volume(_volume())
    vm.segmentation.cache = {0: "primary masks"}
    vm.study.adopt_volume(_volume(), is_primary_series=False)
    _fill_series_state(vm)

    vm.study.close()

    _assert_series_state_empty(vm)
    assert vm.segmentation.primary_cache is None
    assert closed == [True]


def test_the_primary_series_is_segmented_automatically_when_a_model_is_configured(monkeypatch):
    vm, _opened, _closed, _segmented = _vm(_WEIGHTS_CONFIG)
    started = []
    monkeypatch.setattr(vm.segmentation, "segment_volume", lambda: started.append(True))
    monkeypatch.setitem(sys.modules, "bictool.domain.segmentation.head_ct_unet", SimpleNamespace())

    vm.study.adopt_volume(_volume(), is_primary_series=False)
    assert started == []  # another series: only on request
    vm.study.adopt_volume(_volume())
    assert started == [True]

    vm.auto_segment_primary = False
    vm.study.adopt_volume(_volume())
    assert started == [True]


def test_without_a_configured_model_nothing_is_segmented_automatically(monkeypatch):
    vm, _opened, _closed, _segmented = _vm()
    started = []
    monkeypatch.setattr(vm.segmentation, "segment_volume", lambda: started.append(True))

    vm.study.adopt_volume(_volume())

    assert started == []


def test_a_segmented_shown_slice_gets_its_wm_gm_recomputed_and_the_rois_follow(monkeypatch):
    vm, _opened, _closed, segmented = _vm()
    vm.study.adopt_volume(_volume())
    vm.brain_tissue.cache = {0: "fallback tissue", 1: "other tissue"}
    roi_calls = []
    monkeypatch.setattr(vm.roi, "on_slice_segmented", roi_calls.append)

    vm.segmentation.slice_segmented.emit(0)
    vm.segmentation.slice_segmented.emit(1)  # not the shown slice: its result stays

    assert vm.brain_tissue.cache == {1: "other tissue"}
    assert segmented == [0, 1] and roi_calls == [0, 1]


def test_changing_the_wm_gm_settings_re_places_the_white_matter_roi2s(monkeypatch):
    vm, _opened, _closed, _segmented = _vm()
    restarted = []
    monkeypatch.setattr(vm.roi, "restart_roi2", lambda: restarted.append(True))

    vm.brain_tissue.set_method(vm.brain_tissue.method)

    assert restarted == [True]


def test_dropping_the_qa_results_also_drops_the_export():
    vm, _opened, _closed, _segmented = _vm()
    vm.export.last_result = "result"

    vm.qa.reset()

    assert vm.export.last_result is None


def test_opening_another_study_forgets_the_kept_primary_segmentation():
    vm, _opened, _closed, _segmented = _vm()
    vm.study.adopt_volume(_volume())
    vm.segmentation.cache = {0: "primary masks"}
    vm.study.adopt_volume(_volume(), is_primary_series=False)

    vm.study.series_list_changed.emit(["new study axial"])

    assert vm.segmentation.primary_cache is None
