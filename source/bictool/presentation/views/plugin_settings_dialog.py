"""Settings dialog for a plug-in's WM/GM method (`plugins.TissueMethod.settings`),
built from the fields of its settings dataclass: a number box per `int` or
`float` field, a tick box per `bool` field; other fields keep their value.
"""

from __future__ import annotations

import dataclasses

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from bictool.presentation.views.text_scale import px

# Same palette as `main_window.stylesheet()` - duplicated rather than imported
# to avoid a circular import, matching `export_dialog.py`'s own precedent.
BG = "#11151a"
SURFACE_RAISED = "#20262d"
BORDER = "#2b323a"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"
ACCENT = "#49c8de"


def _stylesheet() -> str:
    return f"""
QDialog {{ background: {BG}; color: {TEXT}; font-size: {px(12)}px; }}
QLabel {{ color: {TEXT}; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; font-size: {px(11)}px; }}
QDoubleSpinBox, QSpinBox {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 3px 6px; color: {TEXT};
}}
QPushButton {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 11px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton#pluginAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


def _field_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


class PluginSettingsDialog(QDialog):
    """Modal form over a plug-in method's settings dataclass, plus "Restore
    Defaults". Doesn't apply anything itself - `main_window` reads `settings()`
    after `exec()` returns `Accepted`."""

    def __init__(self, title: str, current, defaults, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} Settings")
        self.setStyleSheet(_stylesheet())
        self.setMinimumWidth(360)
        self._current, self._defaults = current, defaults
        self._inputs: dict[str, QSpinBox | QDoubleSpinBox | QCheckBox] = {}

        layout = QVBoxLayout(self)
        note = QLabel("A WM/GM method from a plug-in. Changing these clears any already-computed result.")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setSpacing(8)
        for f in dataclasses.fields(current):
            value = getattr(current, f.name)
            if isinstance(value, bool):
                box = QCheckBox()
                box.setChecked(value)
            elif isinstance(value, int):
                box = QSpinBox()
                box.setRange(-1_000_000, 1_000_000)
                box.setValue(value)
            elif isinstance(value, float):
                box = QDoubleSpinBox()
                box.setRange(-1e6, 1e6)
                box.setDecimals(3)
                box.setValue(value)
            else:
                continue  # not editable here - keeps its value
            self._inputs[f.name] = box
            form.addRow(_field_label(f.name), box)
        layout.addLayout(form)

        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self._on_restore_defaults_clicked)
        layout.addWidget(restore_btn)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        accept_btn = button_box.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        accept_btn.setObjectName("pluginAcceptBtn")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _set(self, name: str, value) -> None:
        box = self._inputs[name]
        box.setChecked(value) if isinstance(box, QCheckBox) else box.setValue(value)

    def _on_restore_defaults_clicked(self) -> None:
        for name in self._inputs:
            self._set(name, getattr(self._defaults, name))

    def _on_accept(self) -> None:
        try:
            self.settings()  # the plug-in's own checks (__post_init__)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, self.windowTitle(), f"These settings are not valid:\n{exc}")
            return
        self.accept()

    def settings(self):
        values = {
            name: box.isChecked() if isinstance(box, QCheckBox) else box.value() for name, box in self._inputs.items()
        }
        return dataclasses.replace(self._current, **values)
