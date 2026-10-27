"""Settings dialog for `domain.segmentation.brain_tissue`'s HU-threshold
white/gray matter classification - every tunable parameter that module
exposes (skull-strip HU cutoffs, WM/GM HU bands, smoothing kernel size,
minimum component size) gets one spin box here, so retuning against real
scanner data doesn't require a code change.
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
QPushButton#settingsAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


@dataclass
class BrainTissueSettings:
    """Every tunable parameter `brain_tissue.extract_brain_mask`/
    `classify_brain_tissue` accept, bundled together so `main_window.py` can
    pass one object around instead of six loose fields.
    """

    air_hu_cutoff: float
    bone_hu_cutoff: float
    csf_low: float
    csf_high: float
    white_matter_low: float
    white_matter_high: float
    gray_matter_low: float
    gray_matter_high: float
    smoothing_size: int
    min_component_size: int


class BrainTissueSettingsDialog(QDialog):
    """Modal form: one spin box per `BrainTissueSettings` field, plus a
    "Restore Defaults" button. Doesn't apply anything itself - `main_window`
    reads `settings()` after `exec()` returns `Accepted`.
    """

    def __init__(self, current: BrainTissueSettings, defaults: BrainTissueSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Brain Tissue Settings")
        self.setStyleSheet(_stylesheet())
        self.setMinimumWidth(360)
        self._defaults = defaults

        layout = QVBoxLayout(self)

        note = QLabel(
            "HU-threshold white/gray matter classification (see the Brain Tissue "
            "sidebar box). Changing these clears any already-computed result."
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setSpacing(8)

        self._air_cutoff_spin = self._make_hu_spin(-3000, 3000, current.air_hu_cutoff)
        form.addRow("Skull-strip: air HU cutoff (below)", self._air_cutoff_spin)

        self._bone_cutoff_spin = self._make_hu_spin(-3000, 3000, current.bone_hu_cutoff)
        form.addRow("Skull-strip: bone HU cutoff (above)", self._bone_cutoff_spin)

        self._csf_low_spin = self._make_hu_spin(-3000, 3000, current.csf_low)
        form.addRow("CSF: HU low", self._csf_low_spin)

        self._csf_high_spin = self._make_hu_spin(-3000, 3000, current.csf_high)
        form.addRow("CSF: HU high", self._csf_high_spin)

        self._white_low_spin = self._make_hu_spin(-3000, 3000, current.white_matter_low)
        form.addRow("White matter: HU low", self._white_low_spin)

        self._white_high_spin = self._make_hu_spin(-3000, 3000, current.white_matter_high)
        form.addRow("White matter: HU high", self._white_high_spin)

        self._gray_low_spin = self._make_hu_spin(-3000, 3000, current.gray_matter_low)
        form.addRow("Gray matter: HU low", self._gray_low_spin)

        self._gray_high_spin = self._make_hu_spin(-3000, 3000, current.gray_matter_high)
        form.addRow("Gray matter: HU high", self._gray_high_spin)

        self._smoothing_spin = QSpinBox()
        self._smoothing_spin.setRange(1, 15)
        self._smoothing_spin.setSingleStep(2)  # median filter kernel size - odd values only
        self._smoothing_spin.setValue(current.smoothing_size)
        form.addRow("Smoothing (median filter size, px)", self._smoothing_spin)

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
        accept_btn.setObjectName("settingsAcceptBtn")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _make_hu_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.setSuffix(" HU")
        spin.setValue(value)
        return spin

    def _on_restore_defaults_clicked(self) -> None:
        self._air_cutoff_spin.setValue(self._defaults.air_hu_cutoff)
        self._bone_cutoff_spin.setValue(self._defaults.bone_hu_cutoff)
        self._csf_low_spin.setValue(self._defaults.csf_low)
        self._csf_high_spin.setValue(self._defaults.csf_high)
        self._white_low_spin.setValue(self._defaults.white_matter_low)
        self._white_high_spin.setValue(self._defaults.white_matter_high)
        self._gray_low_spin.setValue(self._defaults.gray_matter_low)
        self._gray_high_spin.setValue(self._defaults.gray_matter_high)
        self._smoothing_spin.setValue(self._defaults.smoothing_size)
        self._min_component_spin.setValue(self._defaults.min_component_size)

    def settings(self) -> BrainTissueSettings:
        return BrainTissueSettings(
            air_hu_cutoff=self._air_cutoff_spin.value(),
            bone_hu_cutoff=self._bone_cutoff_spin.value(),
            csf_low=self._csf_low_spin.value(),
            csf_high=self._csf_high_spin.value(),
            white_matter_low=self._white_low_spin.value(),
            white_matter_high=self._white_high_spin.value(),
            gray_matter_low=self._gray_low_spin.value(),
            gray_matter_high=self._gray_high_spin.value(),
            smoothing_size=self._smoothing_spin.value(),
            min_component_size=self._min_component_spin.value(),
        )
