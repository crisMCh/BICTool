"""Tests `StrokeViewModel` without widgets or TensorFlow."""

from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject

from bictool.presentation.viewmodels.stroke_viewmodel import StrokeViewModel


def _vm(config=None):
    volume = SimpleNamespace(num_slices=3, get_slice_hu=lambda i: np.zeros((4, 4)))
    vm = StrokeViewModel(lambda job: None, lambda: config or {}, lambda: volume)
    events = {"status": [], "messages": [], "checked": []}
    vm.status_changed.connect(events["status"].append)
    vm.message.connect(lambda level, title, text: events["messages"].append(title))
    vm.slice_checked.connect(lambda index, cached: events["checked"].append((index, cached)))
    return vm, events


def test_cached_slice_is_reported_and_its_mask_available():
    vm, events = _vm()
    lesion = np.ones((4, 4), dtype=bool)
    vm.cache[1] = SimpleNamespace(masks={"Lesion": lesion})

    vm.check_slice(1)

    assert events["status"] == ["Done (cached)"] and events["checked"] == [(1, True)]
    assert vm.lesion_mask(1) is lesion and vm.lesion_mask(0) is None


def test_without_weights_the_user_is_told():
    vm, events = _vm()

    vm.check_slice(0)

    assert events["messages"] == ["Stroke model not configured"] and not vm.has_weights()


def test_reset_drops_results_and_blocks_a_late_result():
    vm, events = _vm()
    vm.cache[0] = object()
    worker = SimpleNamespace(signals=QObject())
    vm.active_worker = worker

    vm.reset()

    assert vm.cache == {} and vm.active_worker is None and worker.signals.signalsBlocked()
    assert events["status"][-1] == "—"
