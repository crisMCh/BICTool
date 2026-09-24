"""Non-modal window listing the anatomy segmentation (FR-02) checkboxes -
split out of the main window's sidebar (per explicit user request) so
structure selection doesn't compete for space there.

Shows exactly the 16 raw structures `head_ct_unet` produces (per explicit
follow-up request - "I want to see only the following labels" - superseding
an earlier version of this dialog that curated/combined them into WP2's
own structure names behind a primary/"Other structures" split); no
Brainstem placeholder either, since it isn't one of these 16 (see
`head_ct_unet`'s docstring for that gap - still real, just no longer shown
here as a disabled checkbox).

This dialog owns only the checkbox/slider widgets - `MainWindowView` owns
the actual selection state (`_selected_segmentation_names`, all 16
structures selected by default - this dialog is for narrowing that down,
not opting in from nothing), overlay visibility (`_show_segments_cb`, in
the main sidebar - a quick on/off switch separate from this dialog, so the
user doesn't need to open it just to hide/show what's already selected),
opacity (`_segmentation_opacity`), and the per-slice result cache, since
that state must persist across this dialog being closed/reopened and
across slice navigation (a whole-volume-scoped selection, not a per-slice
or per-dialog-session one). This is a pure view over that state: toggling
a checkbox or moving the slider here only emits a signal; `MainWindowView`
decides what to do about it (apply/clear an overlay, re-render at a new
opacity). Non-modal by construction (a `QDialog`, but `MainWindowView` must
call `.show()`, not `.exec()`) so the user can keep it open while
navigating slices in the main window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Matches `RoiViewport.set_overlay`'s own default opacity (0.6) - the
# slider's starting position reproduces today's existing look exactly,
# rather than silently changing it just by introducing the slider.
_DEFAULT_OPACITY_PERCENT = 60


class SegmentationSelectionDialog(QDialog):
    """Checkbox list (plus an overlay-opacity slider) for choosing which
    segmented structures to overlay, and how transparent they look.

    `mapping` is `main_window.py`'s `_SEGMENTATION_MAPPING` (only its keys -
    the structure names - are used here; the raw-class lookup itself is
    `MainWindowView`'s concern via `decode_masks_for_checkboxes`). `colors`
    maps each name to the hex color its checkbox/overlay uses, keyed the
    same way as `main_window.py`'s module-level `_SEGMENTATION_COLORS` so
    the two always agree.
    """

    selection_changed = Signal(str, bool)  # display_name, checked
    opacity_changed = Signal(float)  # 0.0-1.0

    def __init__(
        self,
        mapping: dict[str, list[str]],
        colors: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Segments")

        root = QHBoxLayout(self)

        checklist = QVBoxLayout()
        # Select/deselect-all work regardless of a checkbox's enabled state
        # (Qt still applies `setChecked` - and emits `toggled` - on a
        # disabled widget; only user *clicks* are blocked), so this is safe
        # to use even before any slice has real data, pre-selecting
        # everything for whenever Segment Volume actually produces it.
        button_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._on_select_all_clicked)
        button_row.addWidget(select_all_btn)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._on_deselect_all_clicked)
        button_row.addWidget(deselect_all_btn)
        checklist.addLayout(button_row)

        self._checkboxes: dict[str, QCheckBox] = {}
        for display_name in mapping:
            cb = self._make_checkbox(display_name, colors[display_name])
            checklist.addWidget(cb)
            self._checkboxes[display_name] = cb
        root.addLayout(checklist, 1)

        # A vertical slider (per explicit request) reads naturally as a
        # transparency "level" - low at the bottom, opaque at the top -
        # and its narrow footprint sits comfortably beside the checkbox
        # column rather than needing its own row.
        opacity_column = QVBoxLayout()
        opacity_column.addWidget(QLabel("Opacity"))
        self._opacity_slider = QSlider(Qt.Orientation.Vertical)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(_DEFAULT_OPACITY_PERCENT)
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        opacity_column.addWidget(self._opacity_slider, 1, Qt.AlignmentFlag.AlignHCenter)
        self._opacity_value_label = QLabel(f"{_DEFAULT_OPACITY_PERCENT}%")
        self._opacity_value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        opacity_column.addWidget(self._opacity_value_label)
        root.addLayout(opacity_column)

    def _make_checkbox(self, display_name: str, color: str) -> QCheckBox:
        cb = QCheckBox(display_name)
        cb.setEnabled(False)  # enabled once real data exists - see set_enabled_all
        cb.setChecked(True)  # all 16 selected by default - set before connecting, so this doesn't itself emit
        cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
        cb.toggled.connect(lambda checked, name=display_name: self.selection_changed.emit(name, checked))
        return cb

    def _on_select_all_clicked(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _on_deselect_all_clicked(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def _on_opacity_slider_changed(self, value: int) -> None:
        self._opacity_value_label.setText(f"{value}%")
        self.opacity_changed.emit(value / 100.0)

    def set_enabled_all(self, enabled: bool) -> None:
        """Enables/disables every checkbox - called once real segmentation
        data exists for at least one slice of the current volume.
        """
        for cb in self._checkboxes.values():
            cb.setEnabled(enabled)

    def reset(self) -> None:
        """Re-checks (all 16 selected is the default) and disables every
        checkbox, and resets opacity back to its default - called on close
        study / new volume load, since the previous volume's selection no
        longer applies to a different, not-yet-segmented volume.
        """
        for cb in self._checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.setEnabled(False)
            cb.blockSignals(False)
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(_DEFAULT_OPACITY_PERCENT)
        self._opacity_slider.blockSignals(False)
        self._opacity_value_label.setText(f"{_DEFAULT_OPACITY_PERCENT}%")
