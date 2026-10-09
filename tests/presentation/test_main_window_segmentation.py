"""Tests for `main_window.py`'s Segmentation (FR-02) and Stroke Detection
wiring - checkbox-to-overlay wiring, the Segment Slice/Check for Lesion
trigger buttons' enable/disable lifecycle, per-slice-index caching, and the
lazy TensorFlow import's error handling.

Never imports the real `head_ct_unet`/`aisd_stroke_unet`/`SegmentationWorker`/
`StrokeWorker` modules (which transitively pull in TensorFlow, a real,
optional dependency this test file has no need to require - see those
modules' own test suites for coverage of the real domain logic against
fake model objects). Instead, fake modules are installed directly into
`sys.modules` (leaf
modules) or as attributes of the real `bictool.domain.segmentation` package
(submodules, so `from bictool.domain.segmentation import head_ct_unet`
resolves via `hasattr` without ever attempting a real import) before
exercising `main_window.py`'s lazy-import call sites - this both avoids the
TensorFlow cost and directly proves those call sites work with whatever
gets returned, decoupled from the real implementation.
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox

import bictool.domain.segmentation as segmentation_pkg
import bictool.presentation.viewmodels.roi_viewmodel as roi_viewmodel_module
import bictool.presentation.views.main_window as main_window_module
from bictool.integration.dicom_file_loader import DicomVolume
from bictool.presentation.viewmodels.roi_selection import RoiRect
from bictool.presentation.views.main_window import MainWindowView

_SLICE_SHAPE = (60, 60)

_ALL_ANATOMY_RAW_NAMES = [
    "Septum Pellucidum", "Ventricular System", "Dural Folds",
    "Caudate Nucleus", "Lentiform Nucleus", "Brain Parenchyma",
    "Subarachnoid Space", "Cerebellum", "Central Sulcus",
    "Internal Capsule", "Insular Cortex", "Frontal Lobe",
    "Parietal Lobe", "Occipital Lobe", "Temporal Lobe", "Thalamus",
]


def _fake_dicom_volume(num_slices: int = 1) -> DicomVolume:
    volume = DicomVolume(
        slice_paths=[f"fake_slice_{i}.dcm" for i in range(num_slices)],
        pixel_spacing_mm=0.5,
        slice_thickness_mm=1.0,
        slice_locations_mm=[float(i) for i in range(num_slices)],
        patient_id="ANON-0001",
        patient_name="ANON Patient",
        patient_sex="O",
        study_id="STU-0001",
        study_date="20260101",
        study_description="CT Schadel Spirale",
        series_description="Head 0.5 Vol",
        series_number="4",
        institution_name="Test Hospital",
        manufacturer_model="TestCorp Scanner X",
        acquisition_date="20260101",
        acquisition_time="10:45:00",
        kvp=120.0,
        tube_current_ma=170.0,
        transfer_syntax_name="Implicit VR Little Endian",
        modality="CT",
        ctdi_vol_mgy=None,
        dlp_mgy_cm=None,
        reconstruction=None,
    )
    volume._slice_cache = {i: np.zeros(_SLICE_SHAPE, dtype=np.float64) for i in range(num_slices)}
    return volume


def _load_fake_volume(window: MainWindowView, num_slices: int = 1, *, allow_auto_segment: bool = False) -> DicomVolume:
    """Loads a fake volume as the primary series (`is_primary_series` -
    `_on_volume_loaded`'s default). `MainWindowView` auto-starts a whole-
    volume segmentation for the primary series when weights are configured
    (`MainViewModel.auto_segment_primary`) - most of the tests using this helper are
    about explicit Segment Slice/Volume click behavior, not that auto-start
    feature (tested separately, see the "auto-segmentation on load" section
    further down), so it's suppressed here by default to avoid an
    unrequested background job interfering with those tests' assertions
    about worker dispatch/caching. Pass `allow_auto_segment=True` to let it
    run for real.
    """
    volume = _fake_dicom_volume(num_slices)
    if not allow_auto_segment:
        window.view_model.auto_segment_primary = False
    window._on_volume_loaded(volume)
    return volume


def _install_passthrough_nifti_volume(window: MainWindowView, num_slices: int = 1) -> None:
    """Pre-populates `window.view_model.segmentation.nifti_volume` with an identity
    passthrough (no real `dicom2nifti` conversion, no orientation
    transform) so `_ensure_nifti_volume_then` finds it already built and
    calls straight through - these tests exercise segmentation dispatch/
    caching/overlay wiring, not the real NIfTI-conversion pipeline itself
    (see `tests/domain/segmentation/test_nifti_alignment.py` and
    `tests/integration/test_dicom_to_nifti.py` for that, and the
    NIfTI-conversion-orchestration tests further down this file for
    `_ensure_nifti_volume_then` itself).
    """
    from bictool.domain.segmentation._nifti_alignment import AxisTransform, DicomNiftiVolume

    window.view_model.segmentation.nifti_volume = DicomNiftiVolume(
        nifti_hu=np.zeros((*_SLICE_SHAPE, num_slices)),
        _dicom_index_to_nifti_index={i: i for i in range(num_slices)},
        _transform=AxisTransform(transpose=False, flip0=False, flip1=False),
    )


def _fake_segmentation_result() -> SimpleNamespace:
    masks = {name: np.zeros(_SLICE_SHAPE, dtype=bool) for name in _ALL_ANATOMY_RAW_NAMES}
    return SimpleNamespace(masks=masks)


def _fake_stroke_result() -> SimpleNamespace:
    return SimpleNamespace(masks={"Lesion": np.zeros(_SLICE_SHAPE, dtype=bool)})


def _pump_scheduler(window: MainWindowView) -> None:
    window._scheduler.pool.waitForDone(5000)
    for _ in range(20):
        QApplication.processEvents()


def _dialog_checkboxes(window: MainWindowView) -> dict[str, QCheckBox]:
    """Opens (or reuses) the Select Segments dialog and returns its
    checkboxes keyed by display text - the checkboxes now live there, not
    on `window` directly (see `SegmentationSelectionDialog`).
    """
    window._on_select_segments_clicked()
    return {cb.text(): cb for cb in window._segment_selection_dialog.findChildren(QCheckBox)}


def _install_fake_worker_module(monkeypatch, module_path: str, worker_attr_name: str, *, result=None, error=None):
    """Installs a fake leaf module at `module_path` (e.g.
    `bictool.application.segmentation_worker`) whose `worker_attr_name`
    attribute is a `QRunnable` that, once run, immediately emits `finished`
    with `result` or `error` with `error` - mirroring the real
    `SegmentationWorker`/`StrokeWorker`'s signal shape without touching
    TensorFlow or a real model.
    """

    class _Signals(QObject):
        finished = Signal(object)
        error = Signal(str)

    class _FakeWorker(QRunnable):
        def __init__(self, model, slice_hu) -> None:
            super().__init__()
            self.model = model
            self.slice_hu = slice_hu
            self.signals = _Signals()

        def run(self) -> None:
            if error is not None:
                self.signals.error.emit(error)
            else:
                self.signals.finished.emit(result)

    fake_module = types.ModuleType(module_path)
    setattr(fake_module, worker_attr_name, _FakeWorker)
    monkeypatch.setitem(sys.modules, module_path, fake_module)
    return _FakeWorker


def _install_fake_domain_module(monkeypatch, submodule_name: str, *, load_weights_error: Exception | None = None):
    """Installs a fake submodule of the real `bictool.domain.segmentation`
    package (e.g. `head_ct_unet`) with `build_model()`/`load_weights()`
    stand-ins, as an attribute of the real package object - so
    `from bictool.domain.segmentation import head_ct_unet` resolves via
    `hasattr` without the import system ever attempting a real import.
    """
    fake_module = types.ModuleType(f"bictool.domain.segmentation.{submodule_name}")

    def build_model():
        return SimpleNamespace(name=f"fake-{submodule_name}-model")

    def load_weights(model, weights_path):
        if load_weights_error is not None:
            raise load_weights_error

    fake_module.build_model = build_model
    # Weights are loaded by `integration.model_files`, imported into the main window.
    loader = {"head_ct_unet": "load_anatomy_weights", "aisd_stroke_unet": "load_stroke_weights"}[submodule_name]
    import bictool.integration.model_files as model_files

    # Anatomy weights are loaded by the SegmentationViewModel (via model_files),
    # stroke weights still by the main window (its own import).
    monkeypatch.setattr(model_files, loader, load_weights)
    if hasattr(main_window_module, loader):
        monkeypatch.setattr(main_window_module, loader, load_weights)
    monkeypatch.setattr(segmentation_pkg, submodule_name, fake_module, raising=False)
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    return fake_module


def _install_fake_nifti_conversion_module(monkeypatch, *, result=None, error=None):
    """Installs a fake `bictool.application.nifti_conversion_worker` module
    whose `NiftiConversionWorker.run()` immediately emits `finished` with
    `result` or `error` with `error` - mirroring `_install_fake_worker_
    module`'s pattern but for `(slice_paths, get_slice_hu)` construction, to
    test `_ensure_nifti_volume_then` itself without a real `dicom2nifti`
    conversion (see `tests/integration/test_dicom_to_nifti.py` for that).
    """

    class _Signals(QObject):
        finished = Signal(object)
        error = Signal(str)

    class _FakeWorker(QRunnable):
        constructed: list = []

        def __init__(self, slice_paths, get_slice_hu) -> None:
            super().__init__()
            self.slice_paths = slice_paths
            self.get_slice_hu = get_slice_hu
            self.signals = _Signals()
            type(self).constructed.append((slice_paths, get_slice_hu))

        def run(self) -> None:
            if error is not None:
                self.signals.error.emit(error)
            else:
                self.signals.finished.emit(result)

    module = types.ModuleType("bictool.application.nifti_conversion_worker")
    module.NiftiConversionWorker = _FakeWorker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return _FakeWorker


def _install_fake_volume_worker_module(monkeypatch, *, error_on_index=None):
    """Installs a fake `bictool.application.volume_segmentation_worker`
    module whose `VolumeSegmentationWorker.run()` synchronously emits
    `slice_finished`+`progress` for every requested index (in order, each
    with its own `_fake_segmentation_result()`), then `finished` - or, if
    `error_on_index` is given, emits `error` on that index instead and stops
    there, mirroring the real worker's fail-and-stop behavior. Cancellation
    is exercised separately, at the handler level (see the cancel tests
    below) rather than raced against this synchronous fake.

    Returns the fake class so a test can inspect `constructed` (a list of
    `(model, get_slice_hu, slice_indices)` tuples, one per instantiation) to
    verify which indices `main_window.py` actually requested.
    """

    class Signals(QObject):
        slice_finished = Signal(int, object)
        progress = Signal(int, int)
        finished = Signal()
        cancelled = Signal(int, int)
        error = Signal(int, str)

    class FakeVolumeWorker(QRunnable):
        constructed: list = []

        def __init__(self, model, get_slice_hu, slice_indices) -> None:
            super().__init__()
            self.model = model
            self.get_slice_hu = get_slice_hu
            self.slice_indices = list(slice_indices)
            self.signals = Signals()
            type(self).constructed.append((model, get_slice_hu, self.slice_indices))

        def cancel(self) -> None:
            pass  # not exercised via this fake - see the cancel-handler tests

        def run(self) -> None:
            total = len(self.slice_indices)
            for completed, index in enumerate(self.slice_indices):
                if error_on_index is not None and index == error_on_index:
                    self.signals.error.emit(index, "boom")
                    return
                self.signals.slice_finished.emit(index, _fake_segmentation_result())
                self.signals.progress.emit(completed + 1, total)
            self.signals.finished.emit()

    module = types.ModuleType("bictool.application.volume_segmentation_worker")
    module.VolumeSegmentationWorker = FakeVolumeWorker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return FakeVolumeWorker


# ---- availability (buttons disabled/enabled based on configured weights) ----


def test_segment_slice_button_disabled_after_volume_load_when_anatomy_weights_not_configured(monkeypatch):
    monkeypatch.setattr(main_window_module, "load_config", lambda: {})
    window = MainWindowView()

    _load_fake_volume(window)

    assert window._segment_slice_action.isEnabled() is False
    assert "Segmentation model weights not configured" in window._segment_slice_action.toolTip()


def test_check_stroke_button_disabled_after_volume_load_when_stroke_weights_not_configured(monkeypatch):
    monkeypatch.setattr(main_window_module, "load_config", lambda: {})
    window = MainWindowView()

    _load_fake_volume(window)

    assert window._check_stroke_btn.isEnabled() is False
    assert "Stroke model weights not configured" in window._check_stroke_btn.toolTip()


def test_segment_slice_button_enabled_after_volume_load_when_anatomy_weights_configured(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    window = MainWindowView()

    _load_fake_volume(window)

    assert window._segment_slice_action.isEnabled() is True
    # No "why disabled" reason - only the help text.
    assert window._segment_slice_action.property("tipReason") == ""
    # Stroke weights weren't configured - the two are independent.
    assert window._check_stroke_btn.isEnabled() is False


def test_select_segments_dialog_shows_exactly_the_16_raw_structures():
    # No WP2 renaming/combining, no "Other structures" split, no Brainstem
    # placeholder - per explicit request, this dialog shows exactly the 16
    # raw structures `head_ct_unet` produces, nothing more.
    window = MainWindowView()
    checkboxes = _dialog_checkboxes(window)

    assert set(checkboxes) == set(main_window_module._SEGMENTATION_MAPPING)
    assert "Brainstem" not in checkboxes


def test_select_segments_button_disabled_until_a_volume_is_loaded_and_weights_configured(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    window = MainWindowView()
    assert window._select_segments_btn.isEnabled() is False
    assert "Open a study first" in window._select_segments_btn.toolTip()

    _load_fake_volume(window)

    assert window._select_segments_btn.isEnabled() is True


def test_select_segments_dialog_is_reused_not_recreated_on_repeated_clicks():
    window = MainWindowView()

    window._on_select_segments_clicked()
    first_dialog = window._segment_selection_dialog
    window._on_select_segments_clicked()

    assert window._segment_selection_dialog is first_dialog


# ---- anatomy segmentation: dispatch, apply, checkbox/overlay wiring ----


def test_segment_slice_dispatches_worker_and_enables_checkboxes(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)

    window._segment_slice_action.trigger()
    _pump_scheduler(window)

    assert window._segmentation_status_label.text() == "Done"
    assert window._segment_slice_action.isEnabled() is True
    for name, cb in _dialog_checkboxes(window).items():
        if name == "Brainstem":
            continue
        assert cb.isEnabled() is True
    assert 0 in window.view_model.segmentation.cache


def test_segment_slice_enables_the_brain_mask_checkbox_for_the_segmented_slice(monkeypatch):
    # "Brain Mask" (Tools -> Brain Tissue box) reflects the CNN-segmentation-
    # derived skull-strip (see `main_window._brain_mask_for_current_slice`)
    # and is disabled until real segmentation data exists for the currently
    # displayed slice - never silently enabled just because a study loaded
    # (see `tests/presentation/test_main_window_brain_tissue.py` for the
    # "no weights configured"/"not yet segmented" disabled cases).
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    assert window._brain_mask_cb.isEnabled() is False

    window._segment_slice_action.trigger()
    _pump_scheduler(window)

    assert window._brain_mask_cb.isEnabled() is True


def test_segment_slice_second_click_uses_cache_and_never_reimports(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    assert window._segmentation_status_label.text() == "Done"

    # Simulate the fake modules becoming unavailable - if the cache path
    # re-imported them, this would raise instead of hitting the cache.
    monkeypatch.delitem(sys.modules, "bictool.application.segmentation_worker", raising=False)
    monkeypatch.delattr(segmentation_pkg, "head_ct_unet", raising=False)

    window._segment_slice_action.trigger()

    assert window._segmentation_status_label.text() == "Done (cached)"


def test_segmentation_checkbox_toggle_sets_and_clears_overlay(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    window._show_segments_cb.setChecked(True)  # master switch defaults off (2026-09-24)
    cb = _dialog_checkboxes(window)["Ventricular System"]

    cb.setChecked(True)
    assert "seg_Ventricular System" in window._roi_viewport._overlay_items
    assert "Ventricular System" in window.view_model.segmentation.selected_names

    cb.setChecked(False)
    assert "seg_Ventricular System" not in window._roi_viewport._overlay_items
    assert "Ventricular System" not in window.view_model.segmentation.selected_names


def test_show_segments_checkbox_hides_and_restores_selected_overlays_without_touching_selection(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    window._show_segments_cb.setChecked(True)  # master switch defaults off (2026-09-24)
    cb = _dialog_checkboxes(window)["Ventricular System"]
    cb.setChecked(True)
    assert "seg_Ventricular System" in window._roi_viewport._overlay_items

    window._show_segments_cb.setChecked(False)

    assert "seg_Ventricular System" not in window._roi_viewport._overlay_items
    assert "Ventricular System" in window.view_model.segmentation.selected_names  # selection itself is untouched

    window._show_segments_cb.setChecked(True)

    assert "seg_Ventricular System" in window._roi_viewport._overlay_items


def test_show_segments_checkbox_starts_disabled_until_weights_configured(monkeypatch):
    monkeypatch.setattr(main_window_module, "load_config", lambda: {})
    window = MainWindowView()

    _load_fake_volume(window)

    assert window._show_segments_cb.isEnabled() is False
    assert window._show_segments_cb.isChecked() is False  # default is "off" (2026-09-24), even while disabled


def test_opacity_slider_change_reapplies_currently_visible_overlays_at_the_new_opacity(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    window._show_segments_cb.setChecked(True)  # master switch defaults off (2026-09-24)
    # All 16 are selected by default - deselect down to just one, so this
    # test can assert on a single, predictable re-application.
    for name, cb in _dialog_checkboxes(window).items():
        if name != "Ventricular System":
            cb.setChecked(False)
    assert window.view_model.segmentation.opacity == 0.6  # default, unchanged so far
    calls = []
    real_set_overlay = window._roi_viewport.set_overlay
    window._roi_viewport.set_overlay = lambda *a, **k: (calls.append(k.get("opacity")), real_set_overlay(*a, **k))[1]

    window._segment_selection_dialog._opacity_slider.setValue(25)

    assert window.view_model.segmentation.opacity == 0.25
    assert calls == [0.25]  # only the one currently-selected structure was re-applied, at the new opacity


def test_opacity_slider_change_does_nothing_while_segments_are_hidden(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    _dialog_checkboxes(window)["Ventricular System"].setChecked(True)
    window._show_segments_cb.setChecked(False)

    window._segment_selection_dialog._opacity_slider.setValue(25)

    assert window.view_model.segmentation.opacity == 0.25  # the value itself is still recorded
    assert "seg_Ventricular System" not in window._roi_viewport._overlay_items  # but nothing is redrawn


def test_segment_slice_button_text_is_select_segments():
    window = MainWindowView()
    assert window._select_segments_btn.text() == "Select Segments…"


def test_segmentation_error_is_shown_and_button_is_reenabled(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", error="boom"
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)

    window._segment_slice_action.trigger()
    _pump_scheduler(window)

    assert window._segmentation_status_label.text() == "Error"
    assert "boom" in window._segmentation_status_label.toolTip()
    assert window._segment_slice_action.isEnabled() is True


def test_segment_slice_shows_error_dialog_when_tensorflow_not_installed(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    # sys.modules[name] = None is how Python itself signals "this import
    # previously failed and is blocked" - the standard way to simulate an
    # ImportError for an otherwise-installed module.
    monkeypatch.setitem(sys.modules, "bictool.application.segmentation_worker", None)
    calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: calls.append(args))
    window = MainWindowView()
    _load_fake_volume(window)

    window._segment_slice_action.trigger()

    assert calls
    assert calls[0][1] == "TensorFlow not installed"
    assert window.view_model.segmentation.active_slice_worker is None


def test_segment_slice_shows_error_dialog_when_weights_file_missing(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/does/not/exist.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet", load_weights_error=FileNotFoundError("no such file"))
    _install_fake_worker_module(monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker")
    calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: calls.append(args))
    window = MainWindowView()
    _load_fake_volume(window)

    window._segment_slice_action.trigger()

    assert calls
    assert calls[0][1] == "Segmentation weights not found"
    assert window.view_model.segmentation.active_slice_worker is None


# ---- stroke detection: dispatch, apply, checkbox/overlay wiring ----


def test_stroke_box_is_hidden_but_still_fully_wired():
    # Hidden per explicit request (2026-09-24 - "don't need it yet"), not
    # removed - every widget/handler underneath it must still exist and
    # work exactly as before, only its visibility changes, so it can be
    # shown again later with no further changes.
    window = MainWindowView()

    assert window._stroke_box.isHidden() is True
    assert window._check_stroke_btn is not None
    assert window._stroke_lesion_cb is not None


def test_check_stroke_dispatches_worker_and_enables_checkbox(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"stroke_weights_path": "/fake/stroke.weights.h5"}}
    )
    _install_fake_domain_module(monkeypatch, "aisd_stroke_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.stroke_worker", "StrokeWorker", result=_fake_stroke_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)

    window._check_stroke_btn.click()
    _pump_scheduler(window)

    assert window._stroke_status_label.text() == "Done"
    assert window._check_stroke_btn.isEnabled() is True
    assert window._stroke_lesion_cb.isEnabled() is True
    assert 0 in window.view_model.stroke.cache


def test_stroke_checkbox_toggle_sets_and_clears_overlay(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"stroke_weights_path": "/fake/stroke.weights.h5"}}
    )
    _install_fake_domain_module(monkeypatch, "aisd_stroke_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.stroke_worker", "StrokeWorker", result=_fake_stroke_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    window._check_stroke_btn.click()
    _pump_scheduler(window)

    window._stroke_lesion_cb.setChecked(True)
    assert "stroke_lesion" in window._roi_viewport._overlay_items

    window._stroke_lesion_cb.setChecked(False)
    assert "stroke_lesion" not in window._roi_viewport._overlay_items


def test_stroke_error_is_shown_and_button_is_reenabled(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"stroke_weights_path": "/fake/stroke.weights.h5"}}
    )
    _install_fake_domain_module(monkeypatch, "aisd_stroke_unet")
    _install_fake_worker_module(monkeypatch, "bictool.application.stroke_worker", "StrokeWorker", error="lesion check failed")
    window = MainWindowView()
    _load_fake_volume(window)

    window._check_stroke_btn.click()
    _pump_scheduler(window)

    assert window._stroke_status_label.text() == "Error"
    assert "lesion check failed" in window._stroke_status_label.toolTip()
    assert window._check_stroke_btn.isEnabled() is True


def test_check_stroke_shows_error_dialog_when_tensorflow_not_installed(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"stroke_weights_path": "/fake/stroke.weights.h5"}}
    )
    monkeypatch.setitem(sys.modules, "bictool.application.stroke_worker", None)
    calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: calls.append(args))
    window = MainWindowView()
    _load_fake_volume(window)

    window._check_stroke_btn.click()

    assert calls
    assert calls[0][1] == "TensorFlow not installed"
    assert window.view_model.stroke.active_worker is None


# ---- lifecycle: close study / new volume / slice change ----


def test_close_study_resets_segmentation_and_stroke_display(monkeypatch):
    monkeypatch.setattr(
        main_window_module,
        "load_config",
        lambda: {
            "segmentation": {
                "anatomy_weights_path": "/fake/weights.hdf5",
                "stroke_weights_path": "/fake/stroke.weights.h5",
            }
        },
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_domain_module(monkeypatch, "aisd_stroke_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    _install_fake_worker_module(
        monkeypatch, "bictool.application.stroke_worker", "StrokeWorker", result=_fake_stroke_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    window._check_stroke_btn.click()
    _pump_scheduler(window)
    dialog_checkboxes = _dialog_checkboxes(window)
    dialog_checkboxes["Ventricular System"].setChecked(False)  # so the reset below has something to actually restore

    window._on_close_study_clicked()

    assert window.view_model.segmentation.cache == {}
    assert window.view_model.stroke.cache == {}
    assert window.view_model.segmentation.selected_names == set(main_window_module._SEGMENTATION_MAPPING)  # back to "all selected"
    assert window._segment_slice_action.isEnabled() is False
    assert "Open a study first" in window._segment_slice_action.toolTip()
    assert window._check_stroke_btn.isEnabled() is False
    assert "Open a study first" in window._check_stroke_btn.toolTip()
    assert window._select_segments_btn.isEnabled() is False
    assert "Open a study first" in window._select_segments_btn.toolTip()
    for cb in dialog_checkboxes.values():
        assert cb.isEnabled() is False
        assert cb.isChecked() is True  # all selected is the default, restored on reset
    assert window._stroke_lesion_cb.isEnabled() is False
    assert window._stroke_lesion_cb.isChecked() is False


def test_opening_a_new_study_clears_a_stale_primary_series_segmentation_cache(monkeypatch):
    # A fresh `_on_series_discovered` (a new study opened, even without an
    # explicit "Close Study" first) must not let a previous, unrelated
    # study's cached primary-series segmentation resurface under the new
    # study's own primary series - see `_on_volume_loaded`'s restore-on-
    # return-to-primary logic, which this reset must run ahead of.
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    assert 0 in window.view_model.segmentation.cache

    from bictool.integration.dicom_file_loader import SeriesCandidate

    candidate = SeriesCandidate(
        label="new study series", sorted_paths=["b.dcm"], slice_locations_mm=[0.0], _reference_header=SimpleNamespace()
    )
    monkeypatch.setattr(main_window_module, "load_series", lambda c: _fake_dicom_volume(1))
    window.view_model.auto_segment_primary = False  # not this test's concern

    window._on_series_discovered([candidate])  # simulates opening a brand new study

    assert window.view_model.segmentation.cache == {}
    assert window._segmentation_status_label.text() == "—"
    assert window.view_model.segmentation.primary_cache is None


def test_returning_to_the_primary_series_without_a_fresh_discovery_restores_the_cache(monkeypatch):
    # Distinct from the study-discovery case above: switching to another
    # series and back to the primary one (per explicit 2026-09-24 request)
    # restores its segmentation instead of restarting it from scratch.
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    assert 0 in window.view_model.segmentation.cache

    # allow_auto_segment stays False (default) - this test is only about
    # `_on_volume_loaded`'s restore logic, not the automatic segmentation's
    # own dispatch (covered separately above).
    _load_fake_volume(window)

    assert 0 in window.view_model.segmentation.cache


def test_explicitly_reloading_the_primary_series_discards_the_stale_cache(monkeypatch):
    # An explicit "Reload Series" re-reads from disk - the previous cache
    # can't be trusted (the file(s) may have changed), so it must NOT be
    # silently restored the way a plain series switch-and-back is.
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    window._segment_slice_action.trigger()
    _pump_scheduler(window)
    assert 0 in window.view_model.segmentation.cache

    window.view_model.auto_segment_primary = False  # not this test's concern
    window._on_volume_loaded(_fake_dicom_volume(1), is_primary_series=True, force_reload=True)

    assert window.view_model.segmentation.cache == {}
    assert window.view_model.segmentation.primary_cache is None


def test_segmentation_selection_persists_across_slice_change_and_reapplies_when_cached(monkeypatch):
    # The structure selection is whole-volume-scoped by explicit request -
    # it must survive navigating to a slice with no cached data yet (only
    # the overlay itself disappears, since `RoiViewport.set_image` clears
    # every overlay on slice change) and reapply automatically, with no
    # re-toggling, once back on an already-segmented slice.
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    _install_passthrough_nifti_volume(window, num_slices=3)
    window._segment_slice_action.trigger()  # segments slice 0 only
    _pump_scheduler(window)
    window._show_segments_cb.setChecked(True)  # master switch defaults off (2026-09-24)
    ventricles_cb = _dialog_checkboxes(window)["Ventricular System"]
    ventricles_cb.setChecked(True)
    assert "seg_Ventricular System" in window._roi_viewport._overlay_items
    assert window._segmentation_status_label.text() == "Done"

    window._slice_slider.setValue(1)  # slice 1 has no cached result yet
    assert "seg_Ventricular System" not in window._roi_viewport._overlay_items
    assert ventricles_cb.isChecked() is True  # selection itself is untouched
    assert "Ventricular System" in window.view_model.segmentation.selected_names
    assert window._segmentation_status_label.text() == "Done"  # browsing never changes the task status

    window._slice_slider.setValue(0)  # back to the already-segmented slice
    assert window._segmentation_status_label.text() == "Done"
    assert "seg_Ventricular System" in window._roi_viewport._overlay_items


# ---- anatomy segmentation: whole-volume batch (background, cancellable) ----


def test_segment_volume_disables_buttons_and_shows_cancel_immediately(monkeypatch):
    # These widget-state changes happen synchronously inside the click
    # handler, before the worker is even submitted to the thread pool - so
    # they must already be true right after click() returns, with no need
    # to pump the scheduler first.
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_volume_worker_module(monkeypatch)
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    _install_passthrough_nifti_volume(window, num_slices=3)

    window._segment_volume_action.trigger()

    assert window._segment_slice_action.isEnabled() is False
    assert window._segment_volume_action.isEnabled() is False
    assert window._cancel_segment_volume_btn.isHidden() is False
    _pump_scheduler(window)  # let the fake worker finish so it doesn't leak into the next test


def test_segment_volume_segments_every_slice_and_caches_each_result(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_volume_worker_module(monkeypatch)
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    _install_passthrough_nifti_volume(window, num_slices=3)

    window._segment_volume_action.trigger()
    _pump_scheduler(window)

    assert set(window.view_model.segmentation.cache) == {0, 1, 2}
    assert window._segmentation_status_label.text() == "Done (volume)"
    assert window._segment_slice_action.isEnabled() is True
    assert window._segment_volume_action.isEnabled() is True
    assert window._cancel_segment_volume_btn.isHidden() is True
    # The currently displayed slice (0) is live-updated as its own result
    # lands mid-batch - checkboxes should already be enabled.
    for name, cb in _dialog_checkboxes(window).items():
        if name == "Brainstem":
            continue
        assert cb.isEnabled() is True


def test_segment_volume_live_updates_overlay_for_the_currently_displayed_slice(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_volume_worker_module(monkeypatch)
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    _install_passthrough_nifti_volume(window, num_slices=3)
    window._show_segments_cb.setChecked(True)  # master switch defaults off (2026-09-24)
    cb = _dialog_checkboxes(window)["Ventricular System"]
    cb.setChecked(True)  # checked ahead of time, even while disabled - toggled again once enabled below

    window._segment_volume_action.trigger()
    _pump_scheduler(window)

    # Slice 0's result landed mid-batch with the box already checked - the
    # overlay should have been applied without needing to re-toggle it.
    assert "seg_Ventricular System" in window._roi_viewport._overlay_items


def test_segment_volume_skips_slices_already_in_the_cache(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    fake_worker_cls = _install_fake_volume_worker_module(monkeypatch)
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    _install_passthrough_nifti_volume(window, num_slices=3)
    window.view_model.segmentation.cache[1] = _fake_segmentation_result()  # slice 1 already segmented earlier

    window._segment_volume_action.trigger()
    _pump_scheduler(window)

    assert len(fake_worker_cls.constructed) == 1
    _model, _get_slice_hu, requested_indices = fake_worker_cls.constructed[0]
    assert requested_indices == [0, 2]


def test_segment_volume_is_a_noop_when_every_slice_is_already_cached(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    fake_worker_cls = _install_fake_volume_worker_module(monkeypatch)
    window = MainWindowView()
    _load_fake_volume(window, num_slices=2)
    window.view_model.segmentation.cache[0] = _fake_segmentation_result()
    window.view_model.segmentation.cache[1] = _fake_segmentation_result()

    window._segment_volume_action.trigger()

    assert fake_worker_cls.constructed == []
    assert window._segmentation_status_label.text() == "Done (all slices already segmented)"


def test_segment_volume_error_shows_which_slice_failed(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_volume_worker_module(monkeypatch, error_on_index=1)
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    _install_passthrough_nifti_volume(window, num_slices=3)

    window._segment_volume_action.trigger()
    _pump_scheduler(window)

    assert window._segmentation_status_label.text() == "Error (slice 2)"
    assert "boom" in window._segmentation_status_label.toolTip()
    assert window._segment_slice_action.isEnabled() is True
    assert window._segment_volume_action.isEnabled() is True
    assert window._cancel_segment_volume_btn.isHidden() is True
    # Slice 0 completed before the error on slice 1 - its result is kept.
    assert 0 in window.view_model.segmentation.cache
    assert 1 not in window.view_model.segmentation.cache


def test_cancel_segment_volume_button_calls_cancel_on_the_active_worker(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    window = MainWindowView()
    _load_fake_volume(window)
    cancel_calls = []
    window.view_model.segmentation.active_volume_worker = SimpleNamespace(cancel=lambda: cancel_calls.append(True))

    window._on_cancel_segment_volume_clicked()

    assert cancel_calls == [True]
    assert window._segmentation_status_label.text() == "Cancelling…"


def test_cancel_segment_volume_button_is_a_noop_without_an_active_worker():
    window = MainWindowView()
    assert window.view_model.segmentation.active_volume_worker is None

    window._on_cancel_segment_volume_clicked()  # must not raise


def test_volume_segmentation_cancelled_resets_ui_state(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    window = MainWindowView()
    _load_fake_volume(window, num_slices=10)
    window._segment_slice_action.setEnabled(False)
    window._segment_volume_action.setEnabled(False)
    window._cancel_segment_volume_btn.setVisible(True)
    window.view_model.segmentation.active_volume_worker = SimpleNamespace(cancel=lambda: None)

    window._on_volume_segmentation_cancelled(3, 10)

    assert window.view_model.segmentation.active_volume_worker is None
    assert window._segment_slice_action.isEnabled() is True
    assert window._segment_volume_action.isEnabled() is True
    assert window._cancel_segment_volume_btn.isHidden() is True
    assert window._segmentation_status_label.text() == "Cancelled (3/10 segmented)"


def test_close_study_cancels_an_in_progress_volume_segmentation_and_ignores_late_signals(monkeypatch):
    # A real volume batch is a multi-minute background job - closing the
    # study (or loading a new volume) while one is in flight must not let a
    # slice that finishes afterward write into what's by then a stale/
    # different cache.
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    window = MainWindowView()
    _load_fake_volume(window, num_slices=5)
    cancel_calls = []
    worker = SimpleNamespace(cancel=lambda: cancel_calls.append(True), signals=QObject())
    window.view_model.segmentation.active_volume_worker = worker

    window._on_close_study_clicked()

    assert cancel_calls == [True]
    assert window.view_model.segmentation.active_volume_worker is None
    # A late signal delivery after this point would be silently dropped by
    # blockSignals(True) on the real worker - not re-testable with this
    # plain QObject stand-in, but `signals.blockSignals(True)` was called
    # without raising, which is what `_reset_segmentation_display` relies on.
    assert worker.signals.signalsBlocked() is True


# ---- NIfTI-conversion orchestration (`_ensure_nifti_volume_then`) ----
# Orientation-transform discovery/matching is covered in
# `tests/domain/segmentation/test_nifti_alignment.py`; the real `dicom2nifti`
# conversion + file dedup in `tests/integration/test_dicom_to_nifti.py` -
# these tests are only about `main_window.py`'s own once-per-study
# caching/dispatch around it.


def test_ensure_nifti_volume_then_calls_on_ready_immediately_when_already_built():
    window = MainWindowView()
    _load_fake_volume(window)
    window.view_model.segmentation.nifti_volume = object()  # already built - no conversion should be attempted
    calls = []

    window._ensure_nifti_volume_then(lambda: calls.append(True))

    assert calls == [True]


def test_ensure_nifti_volume_then_converts_once_then_calls_on_ready(monkeypatch):
    sentinel_volume = object()
    fake_worker_cls = _install_fake_nifti_conversion_module(monkeypatch, result=sentinel_volume)
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    calls = []

    window._ensure_nifti_volume_then(lambda: calls.append(True))
    _pump_scheduler(window)

    assert calls == [True]
    assert window.view_model.segmentation.nifti_volume is sentinel_volume
    assert len(fake_worker_cls.constructed) == 1
    slice_paths, get_slice_hu = fake_worker_cls.constructed[0]
    assert slice_paths == window.view_model.study.volume.slice_paths
    assert get_slice_hu == window.view_model.study.volume.get_slice_hu


def test_ensure_nifti_volume_then_disables_buttons_during_conversion_and_reenables_after(monkeypatch):
    _install_fake_nifti_conversion_module(monkeypatch, result=object())
    window = MainWindowView()
    _load_fake_volume(window)

    window._ensure_nifti_volume_then(lambda: None)

    assert window._segment_slice_action.isEnabled() is False
    assert window._segment_volume_action.isEnabled() is False
    _pump_scheduler(window)
    assert window._segment_slice_action.isEnabled() is True
    assert window._segment_volume_action.isEnabled() is True


def test_ensure_nifti_volume_then_second_call_reuses_the_cached_volume(monkeypatch):
    fake_worker_cls = _install_fake_nifti_conversion_module(monkeypatch, result=object())
    window = MainWindowView()
    _load_fake_volume(window)
    calls = []
    window._ensure_nifti_volume_then(lambda: calls.append(1))
    _pump_scheduler(window)

    window._ensure_nifti_volume_then(lambda: calls.append(2))

    assert calls == [1, 2]
    assert len(fake_worker_cls.constructed) == 1  # not converted a second time


def test_ensure_nifti_volume_then_error_shows_message_and_reenables_buttons_without_calling_on_ready(monkeypatch):
    _install_fake_nifti_conversion_module(monkeypatch, error="bad orientation")
    window = MainWindowView()
    _load_fake_volume(window)
    calls = []

    window._ensure_nifti_volume_then(lambda: calls.append(True))
    _pump_scheduler(window)

    assert calls == []
    assert window.view_model.segmentation.nifti_volume is None
    assert window._segmentation_status_label.text() == "Error"
    assert "bad orientation" in window._segmentation_status_label.toolTip()
    assert window._segment_slice_action.isEnabled() is True
    assert window._segment_volume_action.isEnabled() is True


def test_do_segment_slice_shows_not_segmentable_message_for_an_excluded_slice(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker")
    window = MainWindowView()
    _load_fake_volume(window)
    # Already "built" (skips the conversion worker entirely), but this
    # particular slice has no NIfTI counterpart - e.g. a scout/localizer
    # sharing the series' geometry, see `integration.dicom_to_nifti`.
    window.view_model.segmentation.nifti_volume = SimpleNamespace(is_segmentable=lambda i: False)

    window._segment_slice_action.trigger()

    assert window._segmentation_status_label.text() == "Not segmentable"
    assert window.view_model.segmentation.active_slice_worker is None


# ---- auto-segmentation of the primary series on load ----


def test_loading_the_primary_series_auto_starts_volume_segmentation_when_weights_configured(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    from bictool.domain.segmentation._nifti_alignment import AxisTransform, DicomNiftiVolume

    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_volume_worker_module(monkeypatch)
    _install_fake_nifti_conversion_module(
        monkeypatch,
        result=DicomNiftiVolume(
            nifti_hu=np.zeros((*_SLICE_SHAPE, 3)),
            _dicom_index_to_nifti_index={0: 0, 1: 1, 2: 2},
            _transform=AxisTransform(transpose=False, flip0=False, flip1=False),
        ),
    )
    window = MainWindowView()

    _load_fake_volume(window, num_slices=3, allow_auto_segment=True)
    _pump_scheduler(window)

    # No Segment Slice/Volume click anywhere above - this all happened on its own.
    assert set(window.view_model.segmentation.cache) == {0, 1, 2}
    assert window._segmentation_status_label.text() == "Done (volume)"


def test_loading_the_primary_series_does_not_auto_segment_when_weights_not_configured(monkeypatch):
    monkeypatch.setattr(main_window_module, "load_config", lambda: {})
    fake_worker_cls = _install_fake_volume_worker_module(monkeypatch)

    window = MainWindowView()
    _load_fake_volume(window, allow_auto_segment=True)

    assert window.view_model.segmentation.cache == {}
    assert fake_worker_cls.constructed == []


def test_loading_the_primary_series_does_not_auto_segment_when_tensorflow_not_installed(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    # See `test_segment_slice_shows_error_dialog_when_tensorflow_not_installed`
    # for why `sys.modules[name] = None` simulates a blocked import.
    monkeypatch.setitem(sys.modules, "bictool.domain.segmentation.head_ct_unet", None)
    monkeypatch.delattr(segmentation_pkg, "head_ct_unet", raising=False)
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: critical_calls.append(args))

    window = MainWindowView()
    _load_fake_volume(window, allow_auto_segment=True)

    assert window.view_model.segmentation.cache == {}
    # Auto-start is a convenience, not a user action - it must stay silent
    # rather than popping up "TensorFlow not installed" on every study open.
    assert critical_calls == []


def test_loading_a_non_primary_series_does_not_auto_segment(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    fake_worker_cls = _install_fake_volume_worker_module(monkeypatch)
    window = MainWindowView()

    window._on_volume_loaded(_fake_dicom_volume(2), is_primary_series=False)

    assert window.view_model.segmentation.cache == {}
    assert fake_worker_cls.constructed == []


def test_segmentation_colors_are_all_distinct_across_all_16_structures():
    # Regression test for 2026-09-25's report ("many of them have the same
    # or very similar color") - the old 7-color palette repeated every 7
    # structures; this just checks there are no exact duplicates left.
    colors = list(main_window_module._SEGMENTATION_COLORS.values())
    assert len(colors) == 16
    assert len(set(colors)) == 16


def test_segmentation_colors_of_consecutive_structures_are_well_separated_in_hue():
    # Regression test for the follow-up report ("similar colored labels are
    # close to each other") - `_SEGMENTATION_MAPPING`'s dict order clusters
    # anatomically-neighboring structures together (the four lobes; Caudate/
    # Lentiform Nucleus/Insular Cortex/Internal Capsule around the basal
    # ganglia), so it's not enough for the 16 colors to be merely distinct -
    # consecutive entries specifically must land far apart on the hue wheel.
    import colorsys

    def _hue_degrees(hex_color: str) -> float:
        r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        h, _l, _s = colorsys.rgb_to_hls(r, g, b)
        return h * 360

    colors = list(main_window_module._SEGMENTATION_COLORS.values())
    hues = [_hue_degrees(c) for c in colors]

    for a, b in zip(hues, hues[1:]):
        diff = abs(a - b) % 360
        circular_diff = min(diff, 360 - diff)
        assert circular_diff >= 45, f"consecutive structures only {circular_diff:.1f}° apart in hue"


# ---- Auto ROIs (ROI Selection box) ----

_AUTO_SHAPE = (120, 120)  # roomier than _SLICE_SHAPE: ROI1 and ROI2 side by side


def _septum_result() -> SimpleNamespace:
    """A septum line inside a roomy ventricle, parenchyma around it - at the
    fake volume's 0.5 mm spacing: a 22 px ROI1 on the septum, a 22 px
    ventricle ROI2 next to it.
    """
    masks = {name: np.zeros(_AUTO_SHAPE, dtype=bool) for name in _ALL_ANATOMY_RAW_NAMES}
    masks["Brain Parenchyma"][5:115, 5:115] = True
    masks["Ventricular System"][20:100, 20:100] = True
    masks["Septum Pellucidum"][30:90, 59:61] = True
    masks["Ventricular System"] &= ~masks["Septum Pellucidum"]
    masks["Brain Parenchyma"] &= ~(masks["Ventricular System"] | masks["Septum Pellucidum"])
    return SimpleNamespace(masks=masks)


def _falx_result() -> SimpleNamespace:
    """A straight midline falx line in parenchyma - ROI2 must come from white matter."""
    masks = {name: np.zeros(_AUTO_SHAPE, dtype=bool) for name in _ALL_ANATOMY_RAW_NAMES}
    masks["Frontal Lobe"][5:115, 5:115] = True
    masks["Dural Folds"][15:105, 59:61] = True
    masks["Frontal Lobe"] &= ~masks["Dural Folds"]
    return SimpleNamespace(masks=masks)


def _empty_result() -> SimpleNamespace:
    return SimpleNamespace(masks={name: np.zeros(_AUTO_SHAPE, dtype=bool) for name in _ALL_ANATOMY_RAW_NAMES})


def _window_with_segmented_slices(results_by_slice: dict, num_slices: int = 3) -> MainWindowView:
    window = MainWindowView()
    volume = _fake_dicom_volume(num_slices)
    volume._slice_cache = {i: np.zeros(_AUTO_SHAPE, dtype=np.float64) for i in range(num_slices)}
    window.view_model.auto_segment_primary = False
    window._on_volume_loaded(volume)
    window.view_model.segmentation.cache = dict(results_by_slice)
    window._update_auto_rois_availability()
    return window


def _fake_all_white_matter_classifier(window: MainWindowView, monkeypatch) -> None:
    """Every brain pixel is white matter - stands in for the real (slow) WM/GM methods."""

    def classifier():
        def classify(slice_hu, brain_mask):
            masks = {"CSF": np.zeros_like(brain_mask), "White Matter": brain_mask.copy(), "Gray Matter": np.zeros_like(brain_mask)}
            masks["Brain Mask"] = brain_mask
            return SimpleNamespace(masks=masks)

        return classify

    monkeypatch.setattr(window.view_model.brain_tissue, "classifier", classifier)


def _slot_rect(window: MainWindowView, key: str) -> RoiRect | None:
    return window._roi_viewport.roi(key)


def _touching(a: RoiRect, b: RoiRect) -> bool:
    return a.x0 <= b.x1 and b.x0 <= a.x1 and a.y0 <= b.y1 and b.y0 <= a.y1


def _both_result() -> SimpleNamespace:
    """Falx above the ventricles, septum between them - both pairs on one slice."""
    masks = {name: np.zeros(_AUTO_SHAPE, dtype=bool) for name in _ALL_ANATOMY_RAW_NAMES}
    masks["Frontal Lobe"][2:118, 2:118] = True
    masks["Ventricular System"][50:112, 20:100] = True
    masks["Septum Pellucidum"][55:107, 59:61] = True
    masks["Dural Folds"][4:46, 59:61] = True
    masks["Ventricular System"] &= ~masks["Septum Pellucidum"]
    masks["Frontal Lobe"] &= ~(masks["Ventricular System"] | masks["Septum Pellucidum"] | masks["Dural Folds"])
    return SimpleNamespace(masks=masks)


def _both_pairs_window(monkeypatch, results_by_slice=None, num_slices=1) -> MainWindowView:
    window = _window_with_segmented_slices(results_by_slice or {0: _both_result()}, num_slices=num_slices)
    _fake_all_white_matter_classifier(window, monkeypatch)
    window._auto_rois_btn.setChecked(True)
    _pump_scheduler(window)  # white-matter ROI2s land from the background WM/GM worker
    return window


def test_auto_rois_disabled_until_segmentation_exists():
    window = MainWindowView()
    _load_fake_volume(window)

    assert window._auto_rois_btn.isEnabled() is False


def test_auto_rois_place_falx_and_septum_pairs_side_by_side(monkeypatch):
    window = _both_pairs_window(monkeypatch)

    rois = window._roi_rects
    assert set(rois) == {"falx.roi1", "falx.roi2", "septum.roi1", "septum.roi2"}
    for pair in ("falx", "septum"):
        assert rois[f"{pair}.roi1"].width == rois[f"{pair}.roi2"].width
        assert not _touching(rois[f"{pair}.roi1"], rois[f"{pair}.roi2"])
    assert window.view_model.roi.roi2_regions["septum"][0] == "Ventricular System"
    assert window.view_model.roi.roi2_regions["falx"][0] == "White Matter"
    # Both pairs are drawn on the image at once.
    assert all(_slot_rect(window, key) == rect for key, rect in rois.items())
    status = window.view_model.roi.status_text
    assert "Falx: ROI2 in white matter" in status and "Septum: ROI2 in ventricle" in status


def test_auto_rois_are_computed_for_every_segmented_slice():
    window = _window_with_segmented_slices({0: _septum_result(), 1: _empty_result(), 2: _septum_result()})

    window._auto_rois_btn.setChecked(True)

    assert window._auto_roi1_slices("septum") == [0, 2]
    assert window.view_model.roi.store.complete_pairs(0) == ["septum"]
    assert window.view_model.roi.store.complete_pairs(1) == []


def test_auto_rois_follow_slice_navigation():
    window = _window_with_segmented_slices({0: _septum_result(), 1: _empty_result(), 2: _septum_result()})
    window._auto_rois_btn.setChecked(True)

    window._show_slice(1)
    assert _slot_rect(window, "septum.roi1") is None and _slot_rect(window, "septum.roi2") is None
    assert "Septum: not on this slice" in window.view_model.roi.status_text
    assert "1, 3" in window.view_model.roi.status_text  # 1-based slice numbers

    window._show_slice(2)
    assert _slot_rect(window, "septum.roi1") is not None and _slot_rect(window, "septum.roi2") is not None


def test_auto_rois_fill_the_roi_table_without_set_clicks():
    window = _window_with_segmented_slices({0: _septum_result()}, num_slices=1)

    window._auto_rois_btn.setChecked(True)

    roi1, roi2 = _slot_rect(window, "septum.roi1"), _slot_rect(window, "septum.roi2")
    assert window._roi_rects == {"septum.roi1": roi1, "septum.roi2": roi2}
    assert window._roi_position_items["septum.roi1"].text() == f"{roi1.x0}–{roi1.x1}, {roi1.y0}–{roi1.y1}"
    assert window._roi_size_items["septum.roi2"].text() == f"{roi2.width}×{roi2.height} px"
    assert window._roi_position_items["falx.roi1"].text() == "not set"
    assert window._confirm_rois_cb.isEnabled() is True  # still needs the user's confirmation


def test_a_drawn_rectangle_can_be_saved_into_any_of_the_four_slots(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    auto_septum_roi2 = window._roi_rects["septum.roi2"]
    hand_drawn = RoiRect(5, 70, 27, 92)

    window._roi_viewport.set_draft(hand_drawn)
    window._save_draft_as("septum.roi1")

    assert window._roi_rects["septum.roi1"] == hand_drawn
    assert window.view_model.roi.store.is_manual(0, "septum.roi1")
    # The septum ROI2 was re-placed next to the newly saved ROI1.
    assert window._roi_rects["septum.roi2"] != auto_septum_roi2
    assert not _touching(hand_drawn, window._roi_rects["septum.roi2"])
    assert window._restore_auto_btn.isEnabled() is True


def test_delete_key_removes_one_roi_and_restore_brings_the_automatic_ones_back(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    automatic = dict(window._roi_rects)

    window._roi_viewport.select_roi("falx.roi2")
    assert window._delete_roi_btn.isEnabled() is True
    window._roi_viewport.roi_delete_requested.emit("falx.roi2")  # what the Delete key emits

    assert "falx.roi2" not in window._roi_rects
    assert _slot_rect(window, "falx.roi2") is None
    assert window._roi_position_items["falx.roi2"].text() == "not set"
    assert window.view_model.roi.store.complete_pairs(0) == ["septum"]  # falx pair no longer measurable here

    window._restore_auto_btn.click()

    assert window._roi_rects == automatic
    assert window._restore_auto_btn.isEnabled() is False


def test_a_deleted_roi_stays_deleted_when_navigating(monkeypatch):
    window = _both_pairs_window(monkeypatch, {0: _both_result(), 1: _both_result()}, num_slices=2)
    window._roi_viewport.select_roi("septum.roi1")
    window._delete_roi_btn.click()

    window._show_slice(1)
    assert "septum.roi1" in window._roi_rects  # other slice untouched
    window._show_slice(0)

    assert "septum.roi1" not in window._roi_rects


def test_confirmed_rois_can_not_be_moved_deleted_or_changed(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    window._roi_viewport.select_roi("falx.roi1")

    window._confirm_rois_cb.setChecked(True)

    assert window._roi_viewport._locked is True
    assert window._auto_rois_btn.isEnabled() is False
    assert window._delete_roi_btn.isEnabled() is False
    window._roi_viewport.roi_delete_requested.emit("falx.roi1")
    assert "falx.roi1" in window._roi_rects

    window._confirm_rois_cb.setChecked(False)

    assert window._roi_viewport._locked is False
    assert window._auto_rois_btn.isEnabled() is True


def test_moving_a_roi1_re_places_only_its_own_pairs_roi2(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    septum_before = (window._roi_rects["septum.roi1"], window._roi_rects["septum.roi2"])
    roi1 = window._roi_rects["falx.roi1"]
    moved = RoiRect(roi1.x0, roi1.y0 + 4, roi1.x1, roi1.y1 + 4)

    window._roi_viewport.set_roi("falx.roi1", moved)
    window._roi_viewport.draft_moved.emit("falx.roi1", moved)

    assert window._roi_rects["falx.roi1"] == moved
    assert not _touching(moved, window._roi_rects["falx.roi2"])
    assert (window._roi_rects["septum.roi1"], window._roi_rects["septum.roi2"]) == septum_before


def test_a_hand_moved_roi2_is_kept_when_roi1_moves():
    window = _window_with_segmented_slices({0: _septum_result()}, num_slices=1)
    window._auto_rois_btn.setChecked(True)
    roi2 = _slot_rect(window, "septum.roi2")
    moved_roi2 = RoiRect(roi2.x0 + 2, roi2.y0, roi2.x1 + 2, roi2.y1)
    # What a real drag does: the viewport moves the ROI, then reports it.
    window._roi_viewport.set_roi("septum.roi2", moved_roi2)
    window._roi_viewport.draft_moved.emit("septum.roi2", moved_roi2)

    roi1 = _slot_rect(window, "septum.roi1")
    moved_roi1 = RoiRect(roi1.x0, roi1.y0 + 4, roi1.x1, roi1.y1 + 4)
    window._roi_viewport.set_roi("septum.roi1", moved_roi1)
    window._roi_viewport.draft_moved.emit("septum.roi1", moved_roi1)

    assert _slot_rect(window, "septum.roi2") == moved_roi2


def test_moved_rois_are_remembered_per_slice():
    window = _window_with_segmented_slices({0: _septum_result(), 1: _septum_result()}, num_slices=2)
    window._auto_rois_btn.setChecked(True)
    roi2 = _slot_rect(window, "septum.roi2")
    moved_roi2 = RoiRect(roi2.x0 + 2, roi2.y0, roi2.x1 + 2, roi2.y1)
    window._roi_viewport.draft_moved.emit("septum.roi2", moved_roi2)

    window._show_slice(1)
    window._show_slice(0)

    assert _slot_rect(window, "septum.roi2") == moved_roi2


def test_white_matter_roi2_waits_for_the_background_wm_gm_classification(monkeypatch):
    window = _window_with_segmented_slices({0: _falx_result()}, num_slices=1)
    _fake_all_white_matter_classifier(window, monkeypatch)

    window._auto_rois_btn.setChecked(True)
    assert _slot_rect(window, "falx.roi1") is not None
    assert _slot_rect(window, "falx.roi2") is None  # waiting for WM/GM
    _pump_scheduler(window)

    assert window.view_model.roi.roi2_regions["falx"][0] == "White Matter"
    assert _slot_rect(window, "falx.roi2") is not None
    assert not _touching(_slot_rect(window, "falx.roi1"), _slot_rect(window, "falx.roi2"))


def test_changing_the_wm_gm_method_re_places_white_matter_roi2_and_keeps_the_ventricle_one(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    septum_roi2 = window._roi_rects["septum.roi2"]

    window._on_brain_tissue_method_selected("HU Thresholding")  # clears the WM/GM cache
    assert "falx.roi2" not in window._roi_rects  # pending again...
    assert window._roi_rects["septum.roi2"] == septum_roi2
    _pump_scheduler(window)

    assert window.view_model.roi.roi2_regions["falx"][0] == "White Matter"  # ...and re-placed


def test_unchecking_auto_rois_clears_the_automatic_ones_but_keeps_manual_rois():
    window = _window_with_segmented_slices({0: _septum_result()}, num_slices=1)
    window._auto_rois_btn.setChecked(True)
    window._roi_viewport.set_draft(RoiRect(5, 5, 27, 27))
    window._save_draft_as("falx.roi1")

    window._auto_rois_btn.setChecked(False)

    assert set(window._roi_rects) == {"falx.roi1"}
    assert _slot_rect(window, "septum.roi1") is None


def test_segment_slice_enables_auto_rois(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    _install_fake_domain_module(monkeypatch, "head_ct_unet")
    _install_fake_worker_module(
        monkeypatch, "bictool.application.segmentation_worker", "SegmentationWorker", result=_fake_segmentation_result()
    )
    window = MainWindowView()
    _load_fake_volume(window)
    _install_passthrough_nifti_volume(window)
    assert window._auto_rois_btn.isEnabled() is False

    window._segment_slice_action.trigger()
    _pump_scheduler(window)

    assert window._auto_rois_btn.isEnabled() is True


def test_close_study_resets_auto_rois():
    window = _window_with_segmented_slices({0: _septum_result()}, num_slices=1)
    window._auto_rois_btn.setChecked(True)

    window._on_close_study_clicked()

    assert window._auto_rois_btn.isChecked() is False
    assert window._auto_rois_btn.isEnabled() is False
    assert window.view_model.roi.store.slices() == []
    assert window._roi_rects == {}


def _fake_result(label=1, recipe="physical", features=None):
    return SimpleNamespace(
            mtf_area=0.5, mtf_xaxis=np.array([0.0, 0.5, 1.0]), mtf=np.array([[1.0], [0.6], [0.2]]),
            nps_roi1=SimpleNamespace(var_nps=2.0, var_measured=3.0, f_avg=np.array([0.0, 0.5]),
                                     nps_measured_radagv=np.array([1.0, 0.5])),
            nps_roi2=SimpleNamespace(var_nps=1.0, f_avg=np.array([0.0, 0.5]), nps_measured_radagv=np.array([1.0, 0.5])),
            ratio=SimpleNamespace(ratio_lf_hf=10.0, f_int=np.array([0.1, 0.8]), ratio_nps_mtf=np.array([1.0, 0.0])),
            contrast_roi1=0.3, contrast_roi2=0.1, predicted_label=label, decision_value=0.4 if label else -0.4,
            feature_vector=np.zeros(7) if features is None else np.asarray(features, dtype=float), recipe=recipe,
            nps_f_av=0.2, nps_f_peak=0.1,
        )


def _fake_qa(monkeypatch, label_by_call=None, recipe="physical"):
    import bictool.application.worker as worker_module

    calls = iter(label_by_call or [])

    def evaluate(*args, **kwargs):
        return _fake_result(next(calls, 1), recipe)

    monkeypatch.setattr(worker_module, "evaluate_slice_quality", evaluate)


def _run_all(window):
    window._confirm_rois_cb.setChecked(True)
    window._run_all_btn.click()
    for _ in range(100):
        _pump_scheduler(window)
        if window.view_model.qa.jobs is None:
            break


def test_run_all_measures_both_pairs_and_the_results_page_switches_between_them(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    _fake_qa(monkeypatch, label_by_call=[1, 0], recipe="octave")  # legacy: falx acceptable, septum not

    _run_all(window)

    assert set(window.view_model.qa.results) == {(0, "falx"), (0, "septum")}
    assert window._qa_progress_label.text() == "QA done: 2 ROI pair(s)"
    assert window._results_pair_buttons["falx"].isChecked()
    assert window._result_verdict_label.text() == "Acceptable"

    window._results_pair_buttons["septum"].click()

    assert window._result_verdict_label.text() == "Unacceptable"
    assert "Septum" in window._results_slice_label.text()
    lines = window._study_verdict_label.text().splitlines()
    assert lines[0].startswith("Falx: Acceptable") and lines[1].startswith("Septum: Unacceptable")
    assert [window._study_table.item(row, 1).text() for row in range(2)] == ["Falx", "Septum"]
    assert window._study_provisional_label.isHidden()


def test_physical_recipe_without_enough_falx_slices_explains_the_missing_verdict(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    _fake_qa(monkeypatch)

    _run_all(window)

    assert window._study_verdict_label.text() == "Study verdict: not available"
    assert "this study has 1" in window._study_verdict_basics_label.text()
    assert not window._study_provisional_label.isHidden()
    assert window._study_table.horizontalHeaderItem(5).text() == "Resembles (hint)"
    assert window._result_verdict_label.text() == ""  # falx, but no study verdict
    window._results_pair_buttons["septum"].click()
    assert "not used for the verdict" in window._result_verdict_label.text()


def test_physical_recipe_shows_the_study_verdict_with_interval_and_slice_scores(monkeypatch):
    window = _both_pairs_window(monkeypatch)
    # Typical good-study values: mtf_area, nps_homo, nps_struct, var_struct, ratio_lf_hf, slope, loc_max.
    good = [0.51, 20.0, 100.0, 70.0, 11.0, -1.0, 0.1]
    window.view_model.qa.results = {
        (i, "falx"): _fake_result(features=np.array(good) * (1 + 0.01 * (i % 3))) for i in range(8)
    }
    window.view_model.qa.results[(3, "falx")].feature_vector[1] = 80.0  # one very noisy slice
    window.view_model.qa.results[(4, "septum")] = _fake_result(features=good)

    window.view_model.results.refresh()

    headline = window._study_verdict_label.text()
    assert headline.startswith("Study verdict: ") and "likely acceptable" in headline
    basics = window._study_verdict_basics_label.text()
    assert "Artefacts (noise across slices):" in basics and "Unusually noisy slices: 4." in basics
    details = window._study_verdict_details_label.text()
    assert "from 8 falx slices" in details and "model head-ct-1" in details
    assert "noise above 3x the study median: 4" in details  # slice numbers are 1-based
    assert "not used for the verdict" in details  # septum note
    rows = {
        (window._study_table.item(r, 0).text(), window._study_table.item(r, 1).text()): r
        for r in range(window._study_table.rowCount())
    }
    falx_row, septum_row = rows[("4", "Falx")], rows[("5", "Septum")]
    assert window._study_table.item(falx_row, 5).text() in ("acceptable", "not acceptable", "unclear")
    assert "of 1" in window._study_table.item(falx_row, 5).toolTip()
    assert window._study_table.item(falx_row, 6).text() == "noise outlier"
    assert window._study_table.item(septum_row, 5).text() == "—"


def test_study_export_lists_both_pairs(monkeypatch, tmp_path):
    import csv

    window = _both_pairs_window(monkeypatch)
    _fake_qa(monkeypatch)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    _run_all(window)

    window._perform_study_export({"csv"}, tmp_path)

    slices_csv = next(tmp_path.glob("*_study_slices.csv"))
    with open(slices_csv, newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert rows[0][:2] == ["Slice", "Structure"]
    assert sorted(row[1] for row in rows[1:]) == ["Falx", "Septum"]


def test_auto_rois_skip_slices_with_too_little_brain(monkeypatch):
    # Two identical slices; the eligibility rule (patched) says only slice 0 has enough brain.
    monkeypatch.setattr(roi_viewmodel_module, "eligible_slices", lambda masks: {0} & set(masks))
    window = _both_pairs_window(monkeypatch, {0: _both_result(), 1: _both_result()}, num_slices=2)

    assert window._auto_roi1_slices("falx") == [0]
    assert window._auto_roi1_slices("septum") == [0]
    assert "No automatic ROIs on 1 slice(s) with < 20% of the largest brain area: 2." in (
        window.view_model.roi.status_text
    )


def test_a_newly_segmented_larger_slice_can_make_earlier_slices_ineligible(monkeypatch):
    # Patched rule: only the highest segmented slice is "large enough".
    monkeypatch.setattr(roi_viewmodel_module, "eligible_slices", lambda masks: {max(masks)} if masks else set())
    window = _both_pairs_window(monkeypatch, {0: _both_result()}, num_slices=2)
    assert window._auto_roi1_slices("falx") == [0]

    window.view_model.segmentation.cache[1] = _both_result()
    window.view_model.segmentation.slice_segmented.emit(1)
    _pump_scheduler(window)

    assert window._auto_roi1_slices("falx") == [1]
    assert window.view_model.roi.store.auto(0, "falx.roi2") is None


def test_manual_rois_are_allowed_on_ineligible_slices(monkeypatch):
    monkeypatch.setattr(roi_viewmodel_module, "eligible_slices", lambda masks: set())
    window = _both_pairs_window(monkeypatch)

    window.view_model.roi.store.set_manual(0, "falx.roi1", RoiRect(10, 10, 30, 30))
    window.view_model.roi.store.set_manual(0, "falx.roi2", RoiRect(50, 10, 70, 30))

    assert window._auto_roi1_slices("falx") == []
    assert window.view_model.roi.store.complete_pairs(0) == ["falx"]


def test_scrolling_never_hides_a_running_task_in_the_status_bar():
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    window.view_model.segmentation.cache[0] = _fake_segmentation_result()
    # A volume segmentation in progress (as `_on_volume_segmentation_progress` reports it).
    window._segment_slice_action.setEnabled(False)
    window._segment_volume_action.setEnabled(False)
    window._segmentation_status_label.setText("Segmenting volume… 3/10")
    bar = window._status_panel.progress_bar()
    assert not bar.isHidden()

    for index in (1, 0, 2, 0):  # browse over segmented and unsegmented slices
        window._slice_slider.setValue(index)
        assert window.view_model.study.current_slice == index
        assert not bar.isHidden() and bar.value() == 3
        assert window._segmentation_status_label.text() == "Segmenting volume… 3/10"
        assert not window._segment_slice_action.isEnabled() and not window._segment_volume_action.isEnabled()


def test_scrolling_or_zooming_never_cancels_the_loading_bar():
    window = MainWindowView()
    _load_fake_volume(window, num_slices=3)
    window._study_status_label.setText("Loading DICOM data… (12 files scanned)")  # another series is loading

    window._slice_slider.setValue(1)
    window._update_viewer_overlay()

    assert window._study_activity.is_busy() and not window._status_panel.progress_bar().isHidden()
