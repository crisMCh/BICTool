"""Tests for `main_window.py`'s HU-threshold-only "Brain Tissue" box
(white/gray matter checkboxes) - checkbox-to-overlay wiring, per-slice-index
caching, and the disabled-until-loaded lifecycle.

White Matter/Gray Matter have no ML model and no config to gate on, so most
of these tests never need to install fake TensorFlow modules or touch
`load_config`. "Brain Mask" is the exception (see
`main_window._brain_mask_for_current_slice`/`_update_brain_mask_checkbox_
availability`): it prefers a CNN-segmentation-derived result and is disabled
until one exists for the current slice - the "segmentation completes it"
case needs the fake-segmentation-module machinery, so it's covered in
`test_main_window_segmentation.py` instead
(`test_segment_slice_enables_the_brain_mask_checkbox_for_the_segmented_slice`);
this file only covers the two disabled cases, which need no fake modules.
"""

import numpy as np

import bictool.presentation.viewmodels.brain_tissue_viewmodel as brain_tissue_vm_module
import bictool.presentation.views.main_window as main_window_module
from bictool.integration.dicom_file_loader import DicomVolume
from bictool.presentation.views.main_window import MainWindowView
from bictool.presentation.viewmodels.brain_tissue_viewmodel import METHOD_THRESHOLD

_SLICE_SHAPE = (60, 60)


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
    # A slice built like `test_brain_tissue.py`'s own synthetic head slice -
    # a brain-HU disk wrapped in a skull-HU ring - so the wiring under test
    # actually produces a non-empty overlay, not just an all-False mask.
    slices = {}
    for i in range(num_slices):
        slice_hu = np.full(_SLICE_SHAPE, -1000.0, dtype=np.float64)
        cy, cx, radius = _SLICE_SHAPE[0] // 2, _SLICE_SHAPE[1] // 2, _SLICE_SHAPE[0] // 3
        y, x = np.ogrid[: _SLICE_SHAPE[0], : _SLICE_SHAPE[1]]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        slice_hu[dist < radius] = 28.0  # white-matter-band HU throughout the disk
        slices[i] = slice_hu
    volume._slice_cache = slices
    return volume


def _load_fake_volume(window: MainWindowView, num_slices: int = 1) -> DicomVolume:
    volume = _fake_dicom_volume(num_slices)
    window.view_model.auto_segment_primary = False
    window._on_volume_loaded(volume)
    return volume


def test_brain_tissue_checkboxes_disabled_before_a_study_is_loaded():
    window = MainWindowView()

    assert window._csf_cb.isEnabled() is False
    assert window._white_matter_cb.isEnabled() is False
    assert window._gray_matter_cb.isEnabled() is False
    assert window._brain_mask_cb.isEnabled() is False


def test_csf_white_gray_matter_checkboxes_enabled_once_a_study_is_loaded():
    window = MainWindowView()
    _load_fake_volume(window)

    assert window._csf_cb.isEnabled() is True
    assert window._white_matter_cb.isEnabled() is True
    assert window._gray_matter_cb.isEnabled() is True


def test_brain_mask_checkbox_disabled_without_weights_configured(monkeypatch):
    # Explicitly no weights configured - never rely on the real machine's
    # own `~/.bictool/config.json` being absent, since a dev machine that
    # has configured real segmentation weights for other work would make
    # this test's premise false.
    monkeypatch.setattr(main_window_module, "load_config", lambda: {})
    window = MainWindowView()
    _load_fake_volume(window)

    assert window._brain_mask_cb.isEnabled() is False
    assert "Segmentation model weights not configured" in window._brain_mask_cb.toolTip()


def test_brain_mask_checkbox_disabled_when_slice_not_yet_segmented(monkeypatch):
    monkeypatch.setattr(
        main_window_module, "load_config", lambda: {"segmentation": {"anatomy_weights_path": "/fake/weights.hdf5"}}
    )
    window = MainWindowView()
    _load_fake_volume(window)  # automatic segmentation off, so nothing actually segments

    assert window._brain_mask_cb.isEnabled() is False
    assert "Waiting for anatomy segmentation on this slice…" in window._brain_mask_cb.toolTip()


def test_brain_mask_checkbox_toggle_sets_and_clears_overlay():
    window = MainWindowView()
    _load_fake_volume(window)

    window._brain_mask_cb.setChecked(True)
    assert "tissue_Brain Mask" in window._roi_viewport._overlay_items

    window._brain_mask_cb.setChecked(False)
    assert "tissue_Brain Mask" not in window._roi_viewport._overlay_items


def test_csf_checkbox_toggle_sets_and_clears_overlay():
    window = MainWindowView()
    _load_fake_volume(window)

    window._csf_cb.setChecked(True)
    assert "tissue_CSF" in window._roi_viewport._overlay_items

    window._csf_cb.setChecked(False)
    assert "tissue_CSF" not in window._roi_viewport._overlay_items


def test_white_matter_checkbox_toggle_sets_and_clears_overlay():
    window = MainWindowView()
    _load_fake_volume(window)

    window._white_matter_cb.setChecked(True)
    assert "tissue_White Matter" in window._roi_viewport._overlay_items

    window._white_matter_cb.setChecked(False)
    assert "tissue_White Matter" not in window._roi_viewport._overlay_items


def test_gray_matter_checkbox_toggle_sets_and_clears_overlay():
    window = MainWindowView()
    _load_fake_volume(window)

    window._gray_matter_cb.setChecked(True)
    assert "tissue_Gray Matter" in window._roi_viewport._overlay_items

    window._gray_matter_cb.setChecked(False)
    assert "tissue_Gray Matter" not in window._roi_viewport._overlay_items


def test_result_is_cached_per_slice_and_not_recomputed(monkeypatch):
    window = MainWindowView()
    _load_fake_volume(window)
    window._on_brain_tissue_method_selected("HU Thresholding")

    call_count = 0
    real_classify = brain_tissue_vm_module.classify_brain_tissue

    def _counting_classify(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(brain_tissue_vm_module, "classify_brain_tissue", _counting_classify)

    window._white_matter_cb.setChecked(True)
    window._white_matter_cb.setChecked(False)
    window._white_matter_cb.setChecked(True)

    assert call_count == 1
    assert 0 in window.view_model.brain_tissue.cache


def test_checked_boxes_persist_and_reapply_across_slice_navigation():
    window = MainWindowView()
    _load_fake_volume(window, num_slices=2)

    window._white_matter_cb.setChecked(True)
    assert "tissue_White Matter" in window._roi_viewport._overlay_items

    window._show_slice(1)

    assert "tissue_White Matter" in window._roi_viewport._overlay_items
    assert 1 in window.view_model.brain_tissue.cache


def test_closing_study_resets_and_disables_brain_tissue_box():
    window = MainWindowView()
    _load_fake_volume(window)
    window._white_matter_cb.setChecked(True)

    window._on_close_study_clicked()

    assert window._white_matter_cb.isEnabled() is False
    assert window._white_matter_cb.isChecked() is False
    assert window.view_model.brain_tissue.cache == {}


def test_settings_dialog_accept_updates_settings_and_recomputes(monkeypatch):
    from PySide6.QtWidgets import QDialog

    from bictool.presentation.views.brain_tissue_settings_dialog import BrainTissueSettings

    new_settings = BrainTissueSettings(
        air_hu_cutoff=-400.0,
        bone_hu_cutoff=90.0,
        csf_low=-5.0,
        csf_high=18.0,
        white_matter_low=18.0,
        white_matter_high=30.0,
        gray_matter_low=30.0,
        gray_matter_high=42.0,
        smoothing_size=5,
        min_component_size=40,
    )

    class _FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def settings(self):
            return new_settings

    monkeypatch.setattr(main_window_module, "BrainTissueSettingsDialog", _FakeDialog)

    window = MainWindowView()
    _load_fake_volume(window)
    window._white_matter_cb.setChecked(True)
    assert 0 in window.view_model.brain_tissue.cache

    window._on_brain_tissue_settings_clicked()

    assert window.view_model.brain_tissue.settings[METHOD_THRESHOLD] == new_settings
    # Checkbox was still checked, so the slice was recomputed under the new
    # settings rather than left stale.
    assert 0 in window.view_model.brain_tissue.cache


def test_settings_dialog_cancel_leaves_settings_unchanged(monkeypatch):
    from PySide6.QtWidgets import QDialog

    class _FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(main_window_module, "BrainTissueSettingsDialog", _FakeDialog)

    window = MainWindowView()
    original_settings = window.view_model.brain_tissue.settings[METHOD_THRESHOLD]

    window._on_brain_tissue_settings_clicked()

    assert window.view_model.brain_tissue.settings[METHOD_THRESHOLD] is original_settings


def test_ensure_result_passes_configured_settings_through(monkeypatch):
    captured = {}
    real_classify = brain_tissue_vm_module.classify_brain_tissue

    def _capturing_classify(slice_hu, brain_mask, **kwargs):
        captured.update(kwargs)
        return real_classify(slice_hu, brain_mask, **kwargs)

    monkeypatch.setattr(brain_tissue_vm_module, "classify_brain_tissue", _capturing_classify)

    window = MainWindowView()
    _load_fake_volume(window)
    window._on_brain_tissue_method_selected("HU Thresholding")
    window.view_model.brain_tissue.settings[METHOD_THRESHOLD].smoothing_size = 7
    window.view_model.brain_tissue.settings[METHOD_THRESHOLD].min_component_size = 99
    window.view_model.brain_tissue.settings[METHOD_THRESHOLD].white_matter_low = 15.0
    window.view_model.brain_tissue.settings[METHOD_THRESHOLD].white_matter_high = 25.0

    window._ensure_brain_tissue_result_for_current_slice()

    assert captured["smoothing_size"] == 7
    assert captured["min_component_size"] == 99
    assert captured["white_matter_range"] == (15.0, 25.0)


def test_method_dropdown_offers_the_methods_and_the_prototype_and_defaults_to_adaptive():
    window = MainWindowView()
    combo = window._brain_tissue_method_combo

    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Fuzzy C-Means", "HU Thresholding", "Adaptive Thresholding", "Anchored (prototype)"
    ]
    assert window.view_model.brain_tissue.method == "Adaptive Thresholding"
    assert combo.currentText() == "Adaptive Thresholding"


def test_method_dropdown_is_always_enabled_no_study_loaded_gate():
    # Unlike the checkboxes, choosing a method is a plain preference (like
    # its own settings dialog) - no study needs to be loaded to set it up.
    window = MainWindowView()
    assert window._brain_tissue_method_combo.isEnabled() is True
    assert window._brain_tissue_settings_btn.isEnabled() is True


def test_choosing_a_method_in_the_dropdown_switches_the_classifier():
    window = MainWindowView()

    window._brain_tissue_method_combo.setCurrentText("Adaptive Thresholding")

    assert window.view_model.brain_tissue.method == "Adaptive Thresholding"


def test_settings_button_opens_the_selected_methods_dialog(monkeypatch):
    window = MainWindowView()
    opened = []
    monkeypatch.setattr(window, "_on_fcm_settings_clicked", lambda: opened.append("fcm"))
    monkeypatch.setattr(window, "_on_adaptive_settings_clicked", lambda: opened.append("adaptive"))
    monkeypatch.setattr(window, "_on_brain_tissue_settings_clicked", lambda: opened.append("threshold"))

    window._brain_tissue_settings_btn.click()
    window._brain_tissue_method_combo.setCurrentText("HU Thresholding")
    window._brain_tissue_settings_btn.click()
    window._brain_tissue_method_combo.setCurrentText("Fuzzy C-Means")
    window._brain_tissue_settings_btn.click()

    assert opened == ["adaptive", "threshold", "fcm"]


def test_switching_method_clears_cache_and_dispatches_to_the_right_function(monkeypatch):
    calls = []
    monkeypatch.setattr(
        brain_tissue_vm_module, "classify_brain_tissue_fcm", lambda *a, **k: calls.append("fcm") or _empty_result()
    )
    monkeypatch.setattr(
        brain_tissue_vm_module, "classify_brain_tissue_adaptive", lambda *a, **k: calls.append("adaptive") or _empty_result()
    )

    window = MainWindowView()
    _load_fake_volume(window)
    window._white_matter_cb.setChecked(True)
    assert calls == ["adaptive"]  # default method is Adaptive Thresholding

    # Checkbox is already checked, so switching methods clears the stale
    # cache AND immediately recomputes under the new method - same
    # "apply now" behavior as a settings-dialog change.
    window._on_brain_tissue_method_selected("Fuzzy C-Means")
    assert calls == ["adaptive", "fcm"]
    assert 0 in window.view_model.brain_tissue.cache

    window._on_brain_tissue_method_selected("HU Thresholding")
    assert calls == ["adaptive", "fcm"]  # real (unmocked) threshold method ran instead, not one of the fakes above
    assert 0 in window.view_model.brain_tissue.cache  # still recomputed and cached, just via the real function


def _empty_result():
    from bictool.domain.segmentation.brain_tissue import BrainTissueResult

    shape = _SLICE_SHAPE
    return BrainTissueResult(
        masks={"White Matter": np.zeros(shape, dtype=bool), "Gray Matter": np.zeros(shape, dtype=bool)}
    )


def test_the_prototype_has_no_settings_button():
    window = MainWindowView()
    assert window._brain_tissue_settings_btn.isEnabled()
    window._brain_tissue_method_combo.setCurrentText("Anchored (prototype)")
    assert not window._brain_tissue_settings_btn.isEnabled()
    window._brain_tissue_method_combo.setCurrentText("Adaptive Thresholding")
    assert window._brain_tissue_settings_btn.isEnabled()
