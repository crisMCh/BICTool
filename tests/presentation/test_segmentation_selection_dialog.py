from PySide6.QtWidgets import QCheckBox, QPushButton

from bictool.presentation.views.segmentation_selection_dialog import SegmentationSelectionDialog

_MAPPING = {
    "Septum Pellucidum": ["Septum Pellucidum"],
    "Ventricular System": ["Ventricular System"],
    "Cerebellum": ["Cerebellum"],
    "Thalamus": ["Thalamus"],
}
_COLORS = {
    "Septum Pellucidum": "#5fbf83",
    "Ventricular System": "#e0b34c",
    "Cerebellum": "#9b7ede",
    "Thalamus": "#e08a4c",
}


def _checkboxes_by_text(dialog: SegmentationSelectionDialog) -> dict[str, QCheckBox]:
    return {cb.text(): cb for cb in dialog.findChildren(QCheckBox)}


def _button(dialog: SegmentationSelectionDialog, text: str) -> QPushButton:
    return {b.text(): b for b in dialog.findChildren(QPushButton)}[text]


def test_a_checkbox_is_created_for_every_mapping_entry_disabled_but_checked_by_default():
    # All structures are selected by default (per explicit request) - this
    # dialog is for narrowing down what's shown, not opting in from
    # nothing; disabled until real segmentation data exists, same as before.
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    checkboxes = _checkboxes_by_text(dialog)
    assert set(checkboxes) == set(_MAPPING)
    for cb in checkboxes.values():
        assert cb.isEnabled() is False
        assert cb.isChecked() is True


def test_set_enabled_all_enables_every_checkbox():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)

    dialog.set_enabled_all(True)

    for cb in _checkboxes_by_text(dialog).values():
        assert cb.isEnabled() is True


def test_set_enabled_all_false_disables_every_checkbox_again():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    dialog.set_enabled_all(True)

    dialog.set_enabled_all(False)

    for cb in _checkboxes_by_text(dialog).values():
        assert cb.isEnabled() is False


def test_toggling_a_checkbox_emits_selection_changed_with_display_name_and_state():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    dialog.set_enabled_all(True)
    emitted = []
    dialog.selection_changed.connect(lambda name, checked: emitted.append((name, checked)))

    # Already checked by default - toggle off then back on to exercise both.
    _checkboxes_by_text(dialog)["Ventricular System"].setChecked(False)
    _checkboxes_by_text(dialog)["Ventricular System"].setChecked(True)

    assert emitted == [("Ventricular System", False), ("Ventricular System", True)]


def test_reset_rechecks_and_disables_every_checkbox_without_emitting():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    dialog.set_enabled_all(True)
    checkboxes = _checkboxes_by_text(dialog)
    checkboxes["Septum Pellucidum"].setChecked(False)
    checkboxes["Cerebellum"].setChecked(False)
    emitted = []
    dialog.selection_changed.connect(lambda name, checked: emitted.append((name, checked)))

    dialog.reset()

    for cb in checkboxes.values():
        assert cb.isChecked() is True  # back to the default - all selected
        assert cb.isEnabled() is False
    assert emitted == []  # reset is a silent state clear, not a user action


def test_select_all_checks_every_checkbox_including_disabled_ones():
    # Works even before any slice has real data (checkboxes still
    # disabled) - Qt still applies setChecked/emits toggled on a disabled
    # widget, only user clicks are blocked - so this pre-selects everything
    # for whenever Segment Volume actually produces data.
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    _button(dialog, "Deselect All").click()  # start from unchecked, since checked is now the default
    emitted = []
    dialog.selection_changed.connect(lambda name, checked: emitted.append((name, checked)))

    _button(dialog, "Select All").click()

    checkboxes = _checkboxes_by_text(dialog)
    for cb in checkboxes.values():
        assert cb.isChecked() is True
    assert set(emitted) == {(name, True) for name in _MAPPING}


def test_deselect_all_unchecks_every_checkbox():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    dialog.set_enabled_all(True)
    _button(dialog, "Select All").click()
    emitted = []
    dialog.selection_changed.connect(lambda name, checked: emitted.append((name, checked)))

    _button(dialog, "Deselect All").click()

    checkboxes = _checkboxes_by_text(dialog)
    for cb in checkboxes.values():
        assert cb.isChecked() is False
    assert set(emitted) == {(name, False) for name in _MAPPING}


def test_select_all_twice_does_not_double_emit_for_already_checked_boxes():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    _button(dialog, "Select All").click()  # already checked by default, but harmless
    emitted = []
    dialog.selection_changed.connect(lambda name, checked: emitted.append((name, checked)))

    _button(dialog, "Select All").click()  # nothing actually changes state this time

    assert emitted == []


def test_window_title_is_select_segments():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    assert dialog.windowTitle() == "Select Segments"


def test_opacity_slider_defaults_to_60_percent():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    assert dialog._opacity_slider.value() == 60
    assert dialog._opacity_value_label.text() == "60%"


def test_moving_the_opacity_slider_emits_opacity_changed_as_a_0_to_1_fraction():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    emitted = []
    dialog.opacity_changed.connect(emitted.append)

    dialog._opacity_slider.setValue(25)

    assert emitted == [0.25]
    assert dialog._opacity_value_label.text() == "25%"


def test_reset_restores_the_default_opacity_without_emitting():
    dialog = SegmentationSelectionDialog(_MAPPING, _COLORS)
    dialog._opacity_slider.setValue(10)
    emitted = []
    dialog.opacity_changed.connect(emitted.append)

    dialog.reset()

    assert dialog._opacity_slider.value() == 60
    assert dialog._opacity_value_label.text() == "60%"
    assert emitted == []
