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


PLUGIN = '''
import numpy as np
from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(name="equal", version="1.0", requires_api=">=0.3,<1")

@plugins.hookimpl
def bictool_pair_metrics(pair):
    return [plugins.MetricValue("roi2_mean", float(pair.roi2_pixels.mean()))]

@plugins.hookimpl
def bictool_slice_metrics(ct_slice):
    wm = ct_slice.tissue()[plugins.WHITE_MATTER] & ct_slice.brain_mask()
    return [plugins.MetricValue("wm_mean", float(ct_slice.image[wm].mean())), plugins.RoiMarker("wm", wm)]

@plugins.hookimpl
def bictool_study_metrics(study):
    return [plugins.MetricValue("brain_px", float(sum(study.brain_mask(i).sum() for i in study.measured_slices)))]

@plugins.hookimpl
def bictool_artefact_hints(study):
    return [plugins.ArtefactHint("checked", study.measured_slices)]
'''


def test_the_gui_and_the_api_give_plugins_the_same_inputs(tmp_path, monkeypatch):
    """Plug-in values, markers and hints (incl. BICTool's brain and WM/GM masks) are the same in both."""
    from bictool.application import plugin_manager

    (tmp_path / "equal.py").write_text(PLUGIN, encoding="utf-8")
    plugin_manager.use(plugin_manager.load_plugins([tmp_path], entry_points=False))
    masks_by_slice = {i: _slice(shift) for i, shift in enumerate((-3, 0, 4))}
    images = {i: _image(masks, i) for i, masks in masks_by_slice.items()}
    volume = SimpleNamespace(
        num_slices=3, pixel_spacing_mm=0.5, slice_thickness_mm=4.0, series_description="Head", slice_paths=[],
        get_slice_hu=lambda i: images[i],
    )
    segmentation = {i: SimpleNamespace(masks=masks) for i, masks in masks_by_slice.items()}
    run_now = lambda job: job.run()  # noqa: E731
    fake = SimpleNamespace(
        mtf_area=0.5, predicted_label=1, decision_value=0.0, recipe="physical", feature_vector=np.zeros(7),
        nps_f_av=0.2, nps_f_peak=0.1, contrast_roi1=0.1, contrast_roi2=0.02,
    )
    monkeypatch.setattr(steps, "evaluate_slice_quality", lambda *a, **k: fake)
    tissue = BrainTissueViewModel(
        lambda: volume, lambda: segmentation, default_threshold_settings(), default_fcm_settings(),
        default_adaptive_settings(),
    )
    gui = RoiViewModel(run_now, lambda: volume, lambda: 0, lambda: segmentation, lambda: tissue.cache, tissue.classifier)
    gui.set_auto(True)
    qa = QaRunViewModel(run_now, lambda: volume, gui.effective, lambda: segmentation, tissue_settings=tissue.tissue_settings)
    qa.run(gui.ready_jobs())

    study = Study(volume)
    study.set_segmentation(masks_by_slice)
    result = study.measure(study.place_rois(), head_ct_model=None)

    assert qa.plugin_values == {(m.slice.index, m.pair): m.plugin_values for m in result.measurements}
    assert set(qa.slice_plugin_values) == {0, 1, 2}
    for index, values in qa.slice_plugin_values.items():
        (gui_values,), (api_values,) = values, result.slice_plugin_values[index]
        assert gui_values.values == api_values.values and gui_values.error is None
        np.testing.assert_array_equal(gui_values.markers[0].mask, api_values.markers[0].mask)
    assert qa.study_plugin_values == result.plugin_values and qa.plugin_hints == result.plugin_hints
    assert qa.plugin_tissue_method == "adaptive"


def test_the_programs_study_result_is_the_apis(monkeypatch):
    """What export-format plug-ins get from the program (`MainViewModel.study_result`) is
    what `api.Study.measure` returns for the same series."""
    from bictool.presentation.viewmodels.main_viewmodel import MainViewModel

    masks_by_slice = {i: _slice(shift) for i, shift in enumerate((-3, 0, 4))}
    images = {i: _image(masks, i) for i, masks in masks_by_slice.items()}
    volume = SimpleNamespace(
        num_slices=3, pixel_spacing_mm=0.5, slice_thickness_mm=4.0, series_description="Head", slice_paths=[],
        get_slice_hu=lambda i: images[i],
    )
    fake = SimpleNamespace(
        mtf_area=0.5, predicted_label=1, decision_value=0.0, recipe="physical", feature_vector=np.arange(7.0),
        nps_f_av=0.2, nps_f_peak=0.1, contrast_roi1=0.1, contrast_roi2=0.02,
    )
    monkeypatch.setattr(steps, "evaluate_slice_quality", lambda *a, **k: fake)
    vm = MainViewModel(
        lambda job: job.run(), {}, {"Brain": ["Brain Parenchyma"]}, lambda: None,
        (default_threshold_settings(), default_fcm_settings(), default_adaptive_settings()),
    )
    vm.study.volume = volume
    vm.segmentation.cache = {i: SimpleNamespace(masks=masks) for i, masks in masks_by_slice.items()}
    vm.segmentation.plausibility = steps.segmentation_check(masks_by_slice, volume.pixel_spacing_mm)
    assert vm.study_result() is None  # nothing measured yet
    vm.roi.set_auto(True)
    vm.qa.run(vm.roi.ready_jobs())

    study = Study(volume)
    study.set_segmentation(masks_by_slice)
    expected = study.measure(study.place_rois(), reviewed=True, head_ct_model=None)

    assert vm.study_result().to_dict() == expected.to_dict()
    vm.study.volume = None
