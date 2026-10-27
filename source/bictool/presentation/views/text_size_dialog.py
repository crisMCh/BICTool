"""Settings > Text size...: the size of all text in BICTool, in percent of the original sizes (80-200 %).

Same pattern as the other settings dialogs: a spin box, "Restore Defaults", Apply/Cancel, a `.value()` getter.
`main_window` saves the value; it applies after a restart (`text_scale`). A preview line shows the chosen size.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from bictool.presentation.views.text_scale import DEFAULT_TEXT_SCALE, MAX_TEXT_SCALE, MIN_TEXT_SCALE, px

# Same palette as `main_window.stylesheet()` - duplicated rather than imported
# to avoid a circular import, matching `export_dialog.py`'s own precedent.
BG = "#11151a"
SURFACE_RAISED = "#20262d"
BORDER = "#2b323a"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"
ACCENT = "#49c8de"
_PREVIEW_BASE_PX = 12  # the main text size at 100 %


def _stylesheet() -> str:
    return f"""
QDialog {{ background: {BG}; color: {TEXT}; font-size: {px(12)}px; }}
QLabel {{ color: {TEXT}; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; font-size: {px(11)}px; }}
QSpinBox {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 3px 6px; color: {TEXT};
}}
QPushButton {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 11px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton#textSizeAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


class TextSizeDialog(QDialog):
    """Modal form with one percentage spin box. Doesn't apply anything itself - `main_window` reads `value()`
    (a factor, e.g. 1.25) after `exec()` returns `Accepted`."""

    def __init__(self, current: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Text Size")
        self.setStyleSheet(_stylesheet())
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        note = QLabel("Size of all text in BICTool. The new size applies after BICTool is restarted.")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self._percent_spin = QSpinBox()
        self._percent_spin.setRange(round(MIN_TEXT_SCALE * 100), round(MAX_TEXT_SCALE * 100))
        self._percent_spin.setSingleStep(5)
        self._percent_spin.setSuffix(" %")
        self._percent_spin.setValue(round(current * 100))
        self._percent_spin.valueChanged.connect(self._update_preview)
        form.addRow("Text size", self._percent_spin)
        layout.addLayout(form)

        self._preview = QLabel("Preview: Study verdict - MTF 0.42 mm⁻¹")
        layout.addWidget(self._preview)
        self._update_preview()

        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(lambda: self._percent_spin.setValue(round(DEFAULT_TEXT_SCALE * 100)))
        layout.addWidget(restore_btn)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        accept_btn = button_box.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        accept_btn.setObjectName("textSizeAcceptBtn")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _update_preview(self) -> None:
        size = max(1, round(_PREVIEW_BASE_PX * self._percent_spin.value() / 100))
        self._preview.setStyleSheet(f"color: {TEXT}; font-size: {size}px;")

    def value(self) -> float:
        return self._percent_spin.value() / 100
