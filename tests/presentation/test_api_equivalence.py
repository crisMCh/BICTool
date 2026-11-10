"""The GUI and the public API run one pipeline: on the same synthetic series,
the GUI's ROI placement (`RoiViewModel` with the WM/GM ViewModel, background
jobs run synchronously) places exactly the ROIs `api.Study` places, and the
QA run measures them with the same numbers. Synthetic label maps only."""

from types import SimpleNamespace

import numpy as np

import bictool.application.pipeline as steps
from bictool.application.qa_settings import default_adaptive_settings, default_fcm_settings, default_threshold_settings
from bictool.api.study import Study
from bictool.domain.roi_selection.structured_roi import DURAL_FOLDS, SEPTUM_PELLUCIDUM
from bictool.presentation.viewmodels.brain_tissue_viewmodel import BrainTissueViewModel
from bictool.presentation.viewmodels.qa_run_viewmodel import QaRunViewModel
from bictool.presentation.viewmodels.roi_viewmodel import RoiViewModel

SHAPE = (160, 160)


def _slice(shift: int) -> dict[str, np.ndarray]:
    """Brain with ventricles split by a septum, and a falx line in front of them."""
    ventricle = np.zeros(SHAPE, dtype=bool)
    ventricle[60:120, 50 + shift:110 + shift] = True
    septum = np.zeros(SHAPE, dtype=bool)
    septum[65:115, 79 + shift:81 + shift] = True
    ventricle &= ~septum
    falx = np.zeros(SHAPE, dtype=bool)
    falx[12:50, 79 + shift:81 + shift] = True
    lobes = np.zeros(SHAPE, dtype=bool)
    lobes[8:152, 8:152] = True
    lobes &= ~(ventricle | septum | falx)
    return {"Frontal Lobe": lobes, "Ventricular System": ventricle, SEPTUM_PELLUCIDUM: septum, DURAL_FOLDS: falx}


def _image(masks, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full(SHAPE, 36.0) + rng.normal(0, 3, SHAPE)
    image[30:130, 30:130] -= 8.0  # a white-matter-like core inside grey matter
    image[masks["Ventricular System"]] = 6.0
    image[masks[SEPTUM_PELLUCIDUM] | masks[DURAL_FOLDS]] = 45.0
    return image


def test_the_gui_and_the_api_place_and_measure_the_same_rois(monkeypatch):
    masks_by_slice = {i: _slice(shift) for i, shift in enumerate((-3, 0, 4))}
    images = {i: _image(masks, i) for i, masks in masks_by_slice.items()}
    volume = SimpleNamespace(
        num_slices=3, pixel_spacing_mm=0.5, slice_thickness_mm=4.0, series_description="Head", slice_paths=[],
        get_slice_hu=lambda i: images[i],
    )
    segmentation = {i: SimpleNamespace(masks=masks) for i, masks in masks_by_slice.items()}
    run_now = lambda job: job.run()  # noqa: E731 - background jobs run synchronously

    tissue = BrainTissueViewModel(
        lambda: volume, lambda: segmentation, default_threshold_settings(), default_fcm_settings(),
        default_adaptive_settings(),
    )
    gui = RoiViewModel(run_now, lambda: volume, lambda: 0, lambda: segmentation, lambda: tissue.cache, tissue.classifier)
    gui.set_auto(True)

    study = Study(volume)
    study.set_segmentation(masks_by_slice)
    api_rois = study.place_rois()

    assert gui.store.slices() == api_rois.slices() == [0, 1, 2]
    for index in range(3):
        assert gui.effective(index) == api_rois.effective(index)
    assert {"falx.roi2", "septum.roi2"} <= set(api_rois.effective(1))  # both pairs, incl. a white-matter ROI2

    # Same measurements: the GUI's QA run and Study.measure call the same step.
    measured = []
    fake = SimpleNamespace(
        mtf_area=0.5, predicted_label=1, decision_value=0.0, recipe="physical", feature_vector=np.zeros(7),
        nps_f_av=0.2, nps_f_peak=0.1, contrast_roi1=0.1, contrast_roi2=0.02,
    )
    monkeypatch.setattr(
        steps, "evaluate_slice_quality",
        lambda img1, img2, pix, **k: measured.append((img1.sum(), img2.sum(), tuple(map(tuple, k["mtf_seeds"])))) or fake,
    )
    qa = QaRunViewModel(run_now, lambda: volume, gui.effective, lambda: segmentation)
    qa.run(gui.ready_jobs())
    gui_calls, measured[:] = list(measured), []
    study.measure(api_rois, head_ct_model=None)

    assert gui_calls and gui_calls == measured
