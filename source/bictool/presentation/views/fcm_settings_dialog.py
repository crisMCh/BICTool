"""Settings dialog for `domain.segmentation.brain_tissue_fcm`'s spatial
Fuzzy C-Means white/gray matter classification - every tunable parameter
that module exposes except `n_clusters` (fixed at 3 - the algorithm assumes
exactly three real HU peaks: CSF/other, White Matter, Gray Matter; changing
it would break the ascending-sort class assignment, not just retune a
threshold).
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
QPushButton#fcmAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


@dataclass
class FcmSettings:
    fuzziness: float
    max_iter: int
    spatial_weight: float
    min_component_size: int


class FcmSettingsDialog(QDialog):
    """Modal form: one spin box per `FcmSettings` field, plus a "Restore
    Defaults" button. Doesn't apply anything itself - `main_window` reads
    `settings()` after `exec()` returns `Accepted`.
    """

    def __init__(self, current: FcmSettings, defaults: FcmSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fuzzy C-Means Settings")
        self.setStyleSheet(_stylesheet())
        self.setMinimumWidth(360)
        self._defaults = defaults

        layout = QVBoxLayout(self)

        note = QLabel(
            "Spatial Fuzzy C-Means white/gray matter classification (Tools → "
            "WM/GM Classification). Changing these clears any already-computed result."
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setSpacing(8)

        self._fuzziness_spin = QDoubleSpinBox()
        self._fuzziness_spin.setRange(1.1, 5.0)
        self._fuzziness_spin.setSingleStep(0.1)
        self._fuzziness_spin.setDecimals(1)
        self._fuzziness_spin.setValue(current.fuzziness)
        form.addRow("Fuzziness (m)", self._fuzziness_spin)

        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, 500)
        self._max_iter_spin.setValue(current.max_iter)
        form.addRow("Max iterations", self._max_iter_spin)

        self._spatial_weight_spin = QDoubleSpinBox()
        self._spatial_weight_spin.setRange(0.0, 1.0)
        self._spatial_weight_spin.setSingleStep(0.05)
        self._spatial_weight_spin.setDecimals(2)
        self._spatial_weight_spin.setValue(current.spatial_weight)
        form.addRow("Spatial blending weight", self._spatial_weight_spin)

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
        accept_btn.setObjectName("fcmAcceptBtn")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_restore_defaults_clicked(self) -> None:
        self._fuzziness_spin.setValue(self._defaults.fuzziness)
        self._max_iter_spin.setValue(self._defaults.max_iter)
        self._spatial_weight_spin.setValue(self._defaults.spatial_weight)
        self._min_component_spin.setValue(self._defaults.min_component_size)

    def settings(self) -> FcmSettings:
        return FcmSettings(
            fuzziness=self._fuzziness_spin.value(),
            max_iter=self._max_iter_spin.value(),
            spatial_weight=self._spatial_weight_spin.value(),
            min_component_size=self._min_component_spin.value(),
        )
