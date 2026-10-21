"""Settings dialog for `domain.segmentation.brain_tissue_adaptive`'s
locally-adaptive HU threshold white/gray matter classification - every
tunable parameter that module exposes.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
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
QPushButton#adaptiveAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


@dataclass
class AdaptiveSettings:
    block_size: int
    plausible_low: float           # fixed range - used when no study range is available (or it's off)
    plausible_high: float
    min_component_size: int
    smoothing_mm: float = 0.0      # Gaussian sigma before classifying (app default 1 mm)
    use_study_range: bool = False  # plausible range from the study's basal-ganglia references (app default on)


class AdaptiveSettingsDialog(QDialog):
    """Modal form: one spin box per `AdaptiveSettings` field, plus a
    "Restore Defaults" button. Doesn't apply anything itself - `main_window`
    reads `settings()` after `exec()` returns `Accepted`.
    """

    def __init__(self, current: AdaptiveSettings, defaults: AdaptiveSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adaptive Threshold Settings")
        self.setStyleSheet(_stylesheet())
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

        self._smoothing_spin = QDoubleSpinBox()
        self._smoothing_spin.setRange(0.0, 5.0)
        self._smoothing_spin.setDecimals(1)
        self._smoothing_spin.setSingleStep(0.5)
        self._smoothing_spin.setSuffix(" mm")
        self._smoothing_spin.setValue(current.smoothing_mm)
        self._smoothing_spin.setToolTip("Light smoothing before classifying - less speckle. 0 = off.")
        form.addRow("Smoothing", self._smoothing_spin)

        self._study_range_check = QCheckBox("Use the study's own range (from the basal ganglia)")
        self._study_range_check.setChecked(current.use_study_range)
        self._study_range_check.setToolTip(
            "Brain-tissue HU range from this study's caudate/lentiform (gray), internal capsule (white) and "
            "ventricles (CSF) instead of the fixed range above - HU levels differ between scanners. "
            "Falls back to the fixed range until the basal ganglia are segmented."
        )
        form.addRow("", self._study_range_check)

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
        self._smoothing_spin.setValue(self._defaults.smoothing_mm)
        self._study_range_check.setChecked(self._defaults.use_study_range)

    def settings(self) -> AdaptiveSettings:
        return AdaptiveSettings(
            block_size=self._block_size_spin.value(),
            plausible_low=self._plausible_low_spin.value(),
            plausible_high=self._plausible_high_spin.value(),
            min_component_size=self._min_component_spin.value(),
            smoothing_mm=self._smoothing_spin.value(),
            use_study_range=self._study_range_check.isChecked(),
        )
