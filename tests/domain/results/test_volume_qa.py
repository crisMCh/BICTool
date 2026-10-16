"""Tests `volume_qa.py` (auto ROIs + QA over a segmented volume) on
synthetic label maps and images - no patient data.
"""

from types import SimpleNamespace

import numpy as np

import bictool.domain.results.volume_qa as volume_qa
from bictool.domain.results.volume_qa import evaluate_volume_qa
from bictool.domain.roi_selection.structured_roi import DURAL_FOLDS, SEPTUM_PELLUCIDUM

_SHAPE = (160, 160)
_PIXEL_SPACING = 0.5


def _septum_slice() -> dict[str, np.ndarray]:
    ventricle = np.zeros(_SHAPE, dtype=bool)
    ventricle[40:120, 40:120] = True
    septum = np.zeros(_SHAPE, dtype=bool)
    septum[45:115, 79:81] = True
    ventricle &= ~septum
    parenchyma = np.zeros(_SHAPE, dtype=bool)
    parenchyma[10:150, 10:150] = True
    parenchyma &= ~(ventricle | septum)
    return {"Brain Parenchyma": parenchyma, "Ventricular System": ventricle, SEPTUM_PELLUCIDUM: septum}


def _image(masks) -> np.ndarray:
    image = np.full(_SHAPE, 30.0)
    image[masks["Ventricular System"]] = 5.0
    image[masks[SEPTUM_PELLUCIDUM]] = 35.0
    return image


def _tissue_classifier(calls):
    def classify(slice_hu, brain_mask, slice_index=None):
        calls.append(slice_hu.shape)
        return SimpleNamespace(masks={"White Matter": brain_mask.copy(), "Gray Matter": np.zeros(_SHAPE, dtype=bool)})

    return classify


def test_every_segmented_slice_gets_rois_and_qa_per_structure(monkeypatch):
    evaluated = []
    monkeypatch.setattr(
        volume_qa,
        "evaluate_slice_quality",
        lambda img1, img2, pix, svm_model=None, mtf_seeds=None, recipe=None: evaluated.append((img1.shape, img2.shape, mtf_seeds, recipe)) or "result",
    )
    masks_by_slice = {0: _septum_slice(), 2: _septum_slice()}
    calls = []

    out = evaluate_volume_qa(
        masks_by_slice, lambda i: _image(masks_by_slice[i]), 3, _PIXEL_SPACING, _tissue_classifier(calls), svm_model={}
    )

    septum = out[SEPTUM_PELLUCIDUM]
    assert sorted(septum.results) == [0, 2]
    roi1, roi2 = septum.rois[0]
    assert roi1.size == roi2.size
    assert evaluated[0][0] == (roi1.size, roi1.size)
    assert evaluated[0][2]  # MTF seeds passed through
    assert evaluated[0][3] == "physical"  # default QA recipe forwarded
    assert out[DURAL_FOLDS].results == {}  # no dural folds in these slices
    assert len(calls) == 2  # tissue classified at most once per slice


def test_a_failing_slice_is_recorded_and_the_rest_continue(monkeypatch):
    def evaluate(img1, img2, pix, svm_model=None, mtf_seeds=None, recipe=None):
        if not evaluate.failed:
            evaluate.failed = True
            raise ValueError("No edge columns survived bump removal")
        return "result"

    evaluate.failed = False
    monkeypatch.setattr(volume_qa, "evaluate_slice_quality", evaluate)
    masks_by_slice = {0: _septum_slice(), 1: _septum_slice()}

    out = evaluate_volume_qa(
        masks_by_slice, lambda i: _image(masks_by_slice[i]), 2, _PIXEL_SPACING, _tissue_classifier([]), svm_model={}
    )

    septum = out[SEPTUM_PELLUCIDUM]
    assert septum.errors == {0: "No edge columns survived bump removal"}
    assert list(septum.results) == [1]


def test_slices_with_too_little_brain_get_no_automatic_rois(monkeypatch):
    monkeypatch.setattr(volume_qa, "evaluate_slice_quality", lambda *a, **k: "result")
    # A septum between small ventricles and no parenchyma around: the structure
    # is found, but the brain is ~8 % of slice 0's (skull base / vertex-like).
    small = {name: np.zeros(_SHAPE, dtype=bool) for name in _septum_slice()}
    small["Ventricular System"][60:100, 60:100] = True
    small[SEPTUM_PELLUCIDUM][62:98, 79:81] = True
    small["Ventricular System"] &= ~small[SEPTUM_PELLUCIDUM]
    masks_by_slice = {0: _septum_slice(), 1: small}

    out = evaluate_volume_qa(
        masks_by_slice, lambda i: _image(masks_by_slice[i]), 2, _PIXEL_SPACING, _tissue_classifier([]), svm_model={}
    )

    septum = out[SEPTUM_PELLUCIDUM]
    assert sorted(septum.results) == [0]
    assert 1 not in septum.rois
    assert septum.skipped == [1]  # the septum was found there, but the slice has too little brain
