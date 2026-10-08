"""Tests `BrainTissueViewModel` without widgets - synthetic slices only."""

from types import SimpleNamespace

import numpy as np

import bictool.presentation.viewmodels.brain_tissue_viewmodel as module
from bictool.presentation.viewmodels.brain_tissue_viewmodel import (
    DEFAULT_METHOD,
    METHOD_FCM,
    METHOD_THRESHOLD,
    BrainTissueViewModel,
)

THRESHOLD = SimpleNamespace(
    air_hu_cutoff=-200.0, bone_hu_cutoff=300.0, csf_low=0.0, csf_high=15.0, white_matter_low=20.0,
    white_matter_high=30.0, gray_matter_low=30.0, gray_matter_high=45.0, smoothing_size=3, min_component_size=10,
)
FCM = SimpleNamespace(fuzziness=2.0, max_iter=50, spatial_weight=0.5, min_component_size=10)
ADAPTIVE = SimpleNamespace(block_size=31, plausible_low=15.0, plausible_high=45.0, min_component_size=10)


def _vm(segmentation_cache=None):
    image = np.full((8, 8), 30.0)
    volume = SimpleNamespace(num_slices=2, get_slice_hu=lambda i: image)
    vm = BrainTissueViewModel(
        lambda: volume, lambda: segmentation_cache or {}, THRESHOLD, FCM, ADAPTIVE
    )
    return vm, image


def _fake_classifier(monkeypatch, calls):
    def classify(slice_hu, brain_mask, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(masks={"White Matter": brain_mask})

    for name in ("classify_brain_tissue", "classify_brain_tissue_fcm", "classify_brain_tissue_adaptive"):
        monkeypatch.setattr(module, name, lambda *a, n=name, **k: (calls.append(n), classify(*a, **k))[1])


def test_results_are_computed_once_per_slice_and_carry_the_brain_mask(monkeypatch):
    calls = []
    _fake_classifier(monkeypatch, calls)
    vm, _image = _vm()

    first = vm.result(0)
    assert vm.result(0) is first
    assert calls.count("classify_brain_tissue_adaptive") == 1  # the default method
    assert "Brain Mask" in first.masks and DEFAULT_METHOD == vm.method


def test_changing_method_or_settings_drops_results_and_notifies(monkeypatch):
    calls = []
    _fake_classifier(monkeypatch, calls)
    vm, _image = _vm()
    notified = []
    vm.settings_changed.connect(lambda: notified.append(True))
    vm.result(0)

    vm.set_method(METHOD_FCM)
    assert vm.cache == {} and notified == [True]
    vm.result(0)
    assert "classify_brain_tissue_fcm" in calls

    vm.set_settings(METHOD_THRESHOLD, SimpleNamespace(air_hu_cutoff=-100.0, bone_hu_cutoff=200.0))
    assert vm.cache == {} and len(notified) == 2


def test_brain_mask_prefers_the_anatomy_segmentation_of_the_slice():
    segmented = np.zeros((8, 8), dtype=bool)
    segmented[2:6, 2:6] = True
    vm, image = _vm({0: SimpleNamespace(masks={"Brain Parenchyma": segmented})})

    np.testing.assert_array_equal(vm.brain_mask(0, image), segmented)
    # Slice 1 has no segmentation: HU skull-strip fallback (same shape, computed).
    assert vm.brain_mask(1, image).shape == image.shape


def test_reset_keeps_the_chosen_method():
    vm, _image = _vm()
    vm.set_method(METHOD_FCM)
    vm.cache[0] = object()

    vm.reset()

    assert vm.cache == {} and vm.method == METHOD_FCM
