"""Manual window width/level (WW/WL) dialog - lets the user set an exact
value pair directly, rather than only picking from
`viewmodels.windowing.WINDOW_PRESETS` (the toolbar dropdown).
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QVBoxLayout

# Same palette as `main_window.STYLESHEET` - duplicated rather than imported
# to avoid a circular import, matching `export_dialog.py`'s own precedent.
BG = "#11151a"
SURFACE_RAISED = "#20262d"
BORDER = "#2b323a"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"
ACCENT = "#49c8de"

_STYLESHEET = f"""
QDialog {{ background: {BG}; color: {TEXT}; font-size: 12px; }}
QLabel {{ color: {TEXT}; }}
QDoubleSpinBox {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 3px 6px; color: {TEXT};
}}
QPushButton {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 11px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton#customWindowAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


class CustomWindowDialog(QDialog):
    """Modal form: Width/Level spin boxes, prefilled with the viewport's
    current window (`RoiViewport.current_window()`) - `main_window` reads
    `window()` after `exec()` returns `Accepted` and calls
    `RoiViewport.set_window` itself (this dialog owns no viewport state).
    """

    def __init__(self, current_width: float, current_center: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom Window")
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(8)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(1.0, 10000.0)  # width must stay positive - zero/negative has no meaning
        self._width_spin.setDecimals(1)
        self._width_spin.setSuffix(" HU")
        self._width_spin.setValue(current_width)
        form.addRow("Window Width (WW)", self._width_spin)

        self._center_spin = QDoubleSpinBox()
        self._center_spin.setRange(-3000.0, 5000.0)
        self._center_spin.setDecimals(1)
        self._center_spin.setSuffix(" HU")
        self._center_spin.setValue(current_center)
        form.addRow("Window Level (WL)", self._center_spin)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        accept_btn = button_box.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        accept_btn.setObjectName("customWindowAcceptBtn")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def window(self) -> tuple[float, float]:
        return self._width_spin.value(), self._center_spin.value()
