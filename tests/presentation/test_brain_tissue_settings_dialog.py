"""Tests for `BrainTissueSettingsDialog` - a plain form (no state of its
own beyond the widgets), matching `test_export_dialog.py`'s style.
"""

from bictool.presentation.views.brain_tissue_settings_dialog import BrainTissueSettings, BrainTissueSettingsDialog

_DEFAULTS = BrainTissueSettings(
    air_hu_cutoff=-500.0,
    bone_hu_cutoff=100.0,
    white_matter_low=20.0,
    white_matter_high=32.0,
    gray_matter_low=32.0,
    gray_matter_high=45.0,
    smoothing_size=3,
    min_component_size=20,
)


def test_dialog_starts_prefilled_with_the_current_settings():
    current = BrainTissueSettings(
        air_hu_cutoff=-400.0,
        bone_hu_cutoff=90.0,
        white_matter_low=18.0,
        white_matter_high=30.0,
        gray_matter_low=30.0,
        gray_matter_high=42.0,
        smoothing_size=5,
        min_component_size=40,
    )
    dialog = BrainTissueSettingsDialog(current, _DEFAULTS)

    assert dialog.settings() == current


def test_restore_defaults_resets_all_fields():
    current = BrainTissueSettings(
        air_hu_cutoff=-400.0,
        bone_hu_cutoff=90.0,
        white_matter_low=18.0,
        white_matter_high=30.0,
        gray_matter_low=30.0,
        gray_matter_high=42.0,
        smoothing_size=5,
        min_component_size=40,
    )
    dialog = BrainTissueSettingsDialog(current, _DEFAULTS)

    dialog._on_restore_defaults_clicked()

    assert dialog.settings() == _DEFAULTS


def test_editing_a_field_is_reflected_in_settings():
    dialog = BrainTissueSettingsDialog(_DEFAULTS, _DEFAULTS)

    dialog._min_component_spin.setValue(100)
    dialog._smoothing_spin.setValue(7)

    result = dialog.settings()
    assert result.min_component_size == 100
    assert result.smoothing_size == 7
