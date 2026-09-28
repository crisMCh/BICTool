"""Settings dialog for `domain.segmentation.brain_tissue_adaptive`'s
locally-adaptive HU threshold white/gray matter classification - every
tunable parameter that module exposes.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

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
QLabel[role="muted"] {{ color: {TEXT_MUTED}; font-size: 11px; }}
QDoubleSpinBox, QSpinBox {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 3px 6px; color: {TEXT};
}}
QPushButton {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 11px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton#adaptiveAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


@dataclass
class AdaptiveSettings:
    block_size: int
    plausible_low: float
    plausible_high: float
    min_component_size: int


class AdaptiveSettingsDialog(QDialog):
    """Modal form: one spin box per `AdaptiveSettings` field, plus a
    "Restore Defaults" button. Doesn't apply anything itself - `main_window`
    reads `settings()` after `exec()` returns `Accepted`.
    """

    def __init__(self, current: AdaptiveSettings, defaults: AdaptiveSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adaptive Threshold Settings")
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(360)
        self._defaults = defaults

        layout = QVBoxLayout(self)

        note = QLabel(
            "Locally-adaptive HU threshold white/gray matter classification "
            "(Tools → WM/GM Classification). Changing these clears any "
            "already-computed result."
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setSpacing(8)

        self._block_size_spin = QSpinBox()
        self._block_size_spin.setRange(3, 199)
        self._block_size_spin.setSingleStep(2)  # local-mean window size - odd values only
        self._block_size_spin.setValue(current.block_size)
        form.addRow("Local window size (px)", self._block_size_spin)

        self._plausible_low_spin = QDoubleSpinBox()
        self._plausible_low_spin.setRange(-3000.0, 3000.0)
        self._plausible_low_spin.setDecimals(1)
        self._plausible_low_spin.setSuffix(" HU")
        self._plausible_low_spin.setValue(current.plausible_low)
        form.addRow("Plausible range: low", self._plausible_low_spin)

        self._plausible_high_spin = QDoubleSpinBox()
        self._plausible_high_spin.setRange(-3000.0, 3000.0)
        self._plausible_high_spin.setDecimals(1)
        self._plausible_high_spin.setSuffix(" HU")
        self._plausible_high_spin.setValue(current.plausible_high)
        form.addRow("Plausible range: high", self._plausible_high_spin)

        self._min_component_spin = QSpinBox()
        self._min_component_spin.setRange(0, 5000)
        self._min_component_spin.setValue(current.min_component_size)
        form.addRow("Minimum region size (px)", self._min_component_spin)

        layout.addLayout(form)

        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self._on_restore_defaults_clicked)
        layout.addWidget(restore_btn)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        accept_btn = button_box.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        accept_btn.setObjectName("adaptiveAcceptBtn")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_restore_defaults_clicked(self) -> None:
        self._block_size_spin.setValue(self._defaults.block_size)
        self._plausible_low_spin.setValue(self._defaults.plausible_low)
        self._plausible_high_spin.setValue(self._defaults.plausible_high)
        self._min_component_spin.setValue(self._defaults.min_component_size)

    def settings(self) -> AdaptiveSettings:
        return AdaptiveSettings(
            block_size=self._block_size_spin.value(),
            plausible_low=self._plausible_low_spin.value(),
            plausible_high=self._plausible_high_spin.value(),
            min_component_size=self._min_component_spin.value(),
        )
