"""Tests for `FcmSettingsDialog` - a plain form (no state of its own beyond
the widgets), matching `test_export_dialog.py`'s style.
"""

from bictool.presentation.views.fcm_settings_dialog import FcmSettings, FcmSettingsDialog

_DEFAULTS = FcmSettings(fuzziness=2.0, max_iter=50, spatial_weight=0.3, min_component_size=20)


def test_dialog_starts_prefilled_with_the_current_settings():
    current = FcmSettings(fuzziness=1.8, max_iter=30, spatial_weight=0.5, min_component_size=40)
    dialog = FcmSettingsDialog(current, _DEFAULTS)

    assert dialog.settings() == current


def test_restore_defaults_resets_all_fields():
    current = FcmSettings(fuzziness=1.8, max_iter=30, spatial_weight=0.5, min_component_size=40)
    dialog = FcmSettingsDialog(current, _DEFAULTS)

    dialog._on_restore_defaults_clicked()

    assert dialog.settings() == _DEFAULTS


def test_editing_a_field_is_reflected_in_settings():
    dialog = FcmSettingsDialog(_DEFAULTS, _DEFAULTS)

    dialog._max_iter_spin.setValue(80)
    dialog._spatial_weight_spin.setValue(0.6)

    result = dialog.settings()
    assert result.max_iter == 80
    assert result.spatial_weight == 0.6
