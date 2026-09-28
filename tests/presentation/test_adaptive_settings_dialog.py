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
