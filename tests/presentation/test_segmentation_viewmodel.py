"""Tests `SegmentationViewModel` without widgets or TensorFlow."""

from types import SimpleNamespace

from PySide6.QtCore import QObject

from bictool.presentation.viewmodels.segmentation_viewmodel import DEFAULT_OPACITY, SegmentationViewModel

MAPPING = {"Ventricles": ["Ventricular System"]}


def _vm(config=None, num_slices=3):
    volume = SimpleNamespace(num_slices=num_slices, slice_paths=[], get_slice_hu=lambda i: None)
    vm = SegmentationViewModel(lambda job: None, config or {}, lambda: volume, MAPPING)
    events = {"status": [], "messages": [], "segmented": [], "busy": []}
    vm.status_changed.connect(events["status"].append)
    vm.message.connect(lambda level, title, text: events["messages"].append((level, title)))
    vm.slice_segmented.connect(events["segmented"].append)
    vm.busy_changed.connect(events["busy"].append)
    return vm, events


def test_a_cached_slice_is_reported_without_running_anything():
    vm, events = _vm()
    vm.cache[1] = object()

    vm.segment_slice(1)

    assert events["status"] == ["Done (cached)"] and events["segmented"] == [1]


def test_without_configured_weights_the_user_is_told():
    vm, events = _vm()

    vm.segment_slice(0)
    vm.segment_volume()

    assert events["messages"] == [("warning", "Segmentation model not configured")] * 2
    assert not vm.has_weights()


def test_reset_drops_results_jobs_and_choices_and_blocks_late_signals():
    vm, events = _vm()
    vm.cache[0] = object()
    vm.selected_names = set()
    vm.opacity = 0.2
    cancelled = []
    vm.active_volume_worker = SimpleNamespace(cancel=lambda: cancelled.append(True), signals=QObject())
    worker = vm.active_volume_worker

    vm.reset()

    assert vm.cache == {} and vm.selected_names == set(MAPPING) and vm.opacity == DEFAULT_OPACITY
    assert cancelled == [True] and worker.signals.signalsBlocked() and vm.active_volume_worker is None
    assert events["status"][-1] == "—" and events["busy"][-1] is False


def test_the_primary_series_results_are_kept_and_restored():
    vm, _events = _vm()
    primary = {0: object()}
    vm.cache = primary

    vm.backup_primary()
    vm.reset()  # another series is shown
    assert vm.cache == {}
    assert vm.restore_primary() and vm.cache is primary

    vm.forget_primary()
    assert not vm.restore_primary()


def test_volume_job_reports_progress_and_ends_cleanly():
    vm, events = _vm()
    vm.nifti_volume = SimpleNamespace(remap_result_to_dicom_space=lambda r: r)
    vm.active_volume_worker = SimpleNamespace(cancel=lambda: None)

    vm._on_volume_slice_finished(2, "mask")
    vm.on_volume_cancelled(1, 3)

    assert vm.cache[2] == "mask" and events["segmented"] == [2]
    assert events["status"][-1] == "Cancelled (1/3 segmented)" and vm.active_volume_worker is None
