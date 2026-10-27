"""Tests for `AdaptiveSettingsDialog` - a plain form (no state of its own
beyond the widgets), matching `test_export_dialog.py`'s style.
"""

from bictool.presentation.views.adaptive_settings_dialog import AdaptiveSettings, AdaptiveSettingsDialog

_DEFAULTS = AdaptiveSettings(block_size=31, plausible_low=15.0, plausible_high=50.0, min_component_size=20)


def test_dialog_starts_prefilled_with_the_current_settings():
    current = AdaptiveSettings(block_size=51, plausible_low=10.0, plausible_high=45.0, min_component_size=40)
    dialog = AdaptiveSettingsDialog(current, _DEFAULTS)

    assert dialog.settings() == current


def test_restore_defaults_resets_all_fields():
    current = AdaptiveSettings(block_size=51, plausible_low=10.0, plausible_high=45.0, min_component_size=40)
    dialog = AdaptiveSettingsDialog(current, _DEFAULTS)

    dialog._on_restore_defaults_clicked()

    assert dialog.settings() == _DEFAULTS


def test_editing_a_field_is_reflected_in_settings():
    dialog = AdaptiveSettingsDialog(_DEFAULTS, _DEFAULTS)

    dialog._block_size_spin.setValue(61)
    dialog._plausible_low_spin.setValue(12.0)

    result = dialog.settings()
    assert result.block_size == 61
    assert result.plausible_low == 12.0


def test_smoothing_and_study_range_round_trip_and_restore():
    defaults = AdaptiveSettings(31, 15.0, 50.0, 20, smoothing_mm=1.0, use_study_range=True)
    current = AdaptiveSettings(31, 15.0, 50.0, 20, smoothing_mm=0.0, use_study_range=False)
    dialog = AdaptiveSettingsDialog(current, defaults)
    assert dialog.settings() == current

    dialog._smoothing_spin.setValue(2.5)
    dialog._study_range_check.setChecked(True)
    assert (dialog.settings().smoothing_mm, dialog.settings().use_study_range) == (2.5, True)

    dialog._on_restore_defaults_clicked()
    assert dialog.settings() == defaults
