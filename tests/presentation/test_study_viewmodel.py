"""Tests `StudyViewModel` without any widgets - synthetic volumes only."""

from types import SimpleNamespace

import bictool.presentation.viewmodels.study_viewmodel as study_module
from bictool.presentation.viewmodels.study_viewmodel import NO_STUDY_TEXT, StudyViewModel


def _volume(num_slices=3, study_id="S1"):
    return SimpleNamespace(num_slices=num_slices, study_id=study_id, patient_id="P", pixel_spacing_mm=0.5)


def _candidate(label):
    return SimpleNamespace(label=label)


def _vm():
    submitted = []
    vm = StudyViewModel(submitted.append)
    loaded, status = [], []
    vm.volume_loaded.connect(lambda volume, primary, force, prev: loaded.append((volume, primary, force, prev)))
    vm.status_changed.connect(status.append)
    return vm, submitted, loaded, status


def test_open_scans_in_the_background_and_reports_loading():
    vm, submitted, _loaded, status = _vm()
    loading = []
    vm.loading_changed.connect(loading.append)

    vm.open_path("C:/study")

    assert len(submitted) == 1 and vm.active_load_worker is submitted[0]
    assert loading == [True] and status == ["Loading DICOM data…"]


def test_discovered_series_load_the_first_one_as_primary(monkeypatch):
    vm, _submitted, loaded, status = _vm()
    series = []
    vm.series_list_changed.connect(series.append)
    volumes = {"axial": _volume(40), "sagittal": _volume(20)}
    monkeypatch.setattr(study_module, "load_series", lambda candidate: volumes[candidate.label])

    vm.on_series_discovered([_candidate("axial"), _candidate("sagittal")])

    assert series == [["axial", "sagittal"]]
    assert loaded == [(volumes["axial"], True, False, False)]
    assert vm.volume is volumes["axial"] and vm.is_primary_series and vm.series_index == 0
    assert status[-1] == "Loaded: S1 (40 slice(s))"


def test_only_index_zero_is_primary_and_leaving_the_primary_is_reported(monkeypatch):
    vm, _submitted, loaded, _status = _vm()
    monkeypatch.setattr(study_module, "load_series", lambda candidate: _volume())
    vm.series_candidates = [_candidate("a"), _candidate("b")]

    vm.select_series(0)
    vm.select_series(1)
    vm.select_series(0)
    vm.reload_series()

    assert [(primary, force, prev) for _v, primary, force, prev in loaded] == [
        (True, False, False),   # first load
        (False, False, True),   # leaving the primary series
        (True, False, False),   # back to it
        (True, True, True),     # reload of the primary
    ]


def test_slices_are_clamped_and_reported():
    vm, _submitted, _loaded, _status = _vm()
    changed = []
    vm.slice_changed.connect(changed.append)
    vm.adopt_volume(_volume(3))

    vm.set_slice(2)
    vm.set_slice(10)
    vm.set_slice(-1)

    assert changed == [2, 2, 0] and vm.current_slice == 0


def test_close_forgets_the_study():
    vm, _submitted, _loaded, status = _vm()
    closed = []
    vm.closed.connect(lambda: closed.append(True))
    vm.adopt_volume(_volume())

    vm.close()

    assert vm.volume is None and vm.series_candidates == [] and closed == [True]
    assert status[-1] == NO_STUDY_TEXT


def test_load_errors_end_loading_and_are_reported():
    vm, _submitted, _loaded, status = _vm()
    errors, loading = [], []
    vm.load_failed.connect(errors.append)
    vm.loading_changed.connect(loading.append)
    vm.open_path("C:/broken")

    vm._on_load_error("no DICOM files")

    assert errors == ["no DICOM files"] and loading == [True, False] and status[-1] == NO_STUDY_TEXT
