"""Tests for `CustomWindowDialog` - a plain form (no state of its own
beyond the widgets), matching `test_export_dialog.py`'s style.
"""

from bictool.presentation.views.custom_window_dialog import CustomWindowDialog


def test_dialog_starts_prefilled_with_the_current_window():
    dialog = CustomWindowDialog(current_width=90.0, current_center=35.0)

    assert dialog.window() == (90.0, 35.0)


def test_editing_the_fields_is_reflected_in_window():
    dialog = CustomWindowDialog(current_width=90.0, current_center=35.0)

    dialog._width_spin.setValue(200.0)
    dialog._center_spin.setValue(50.0)

    assert dialog.window() == (200.0, 50.0)
