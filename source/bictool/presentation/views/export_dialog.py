"""Export dialog (FR-07): lets the user pick which pieces of the currently
computed result to write out, and where.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

# Same palette as `main_window.STYLESHEET` - duplicated rather than imported
# to avoid a circular import (main_window constructs this dialog), matching
# the precedent already set by `roi_viewport.py`'s own independent color
# constants.
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
QCheckBox {{ color: {TEXT}; padding: 2px 0; }}
QLineEdit {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 8px; color: {TEXT};
}}
QPushButton {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 11px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; }}
QPushButton#exportAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
QPushButton#exportAcceptBtn:disabled {{ background: {SURFACE_RAISED}; color: {TEXT_MUTED}; }}
"""


class ExportDialog(QDialog):
    """Lets the user choose which result pieces to export - metrics as an
    Excel workbook, metrics as ASCII/CSV, charts as JPG images, any
    combination of at least one - and a destination folder.

    Doesn't perform the export itself: `main_window` reads `selected_items()`
    /`destination()` after `exec()` returns `Accepted` and writes the files,
    since chart-image export needs the live `QChartView`s this dialog has no
    reason to know about (keeps this dialog a plain, testable-without-a-
    result "what do you want" prompt).
    """

    def __init__(self, *, default_dir: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("What to export"))
        self._excel_cb = QCheckBox("Metrics — Excel workbook (.xlsx)")
        self._excel_cb.setChecked(True)
        self._csv_cb = QCheckBox("Metrics — ASCII / CSV (.csv)")
        self._jpg_cb = QCheckBox("Charts — JPG images (MTF, NPS, PS, PS/MTF)")
        for cb in (self._excel_cb, self._csv_cb, self._jpg_cb):
            cb.toggled.connect(self._update_accept_enabled)
            layout.addWidget(cb)

        layout.addSpacing(6)
        layout.addWidget(QLabel("Destination folder"))
        dest_row = QHBoxLayout()
        self._dest_edit = QLineEdit(default_dir)
        self._dest_edit.textChanged.connect(self._update_accept_enabled)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_clicked)
        dest_row.addWidget(self._dest_edit, 1)
        dest_row.addWidget(browse_btn)
        layout.addLayout(dest_row)

        note = QLabel("Exports contain only de-identified fields (no patient name/ID) — NFR-06, GDPR/BDSG.")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._accept_btn = button_box.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        self._accept_btn.setObjectName("exportAcceptBtn")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._update_accept_enabled()

    def _on_browse_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose export destination", self._dest_edit.text())
        if directory:
            self._dest_edit.setText(directory)

    def _update_accept_enabled(self) -> None:
        any_selected = self._excel_cb.isChecked() or self._csv_cb.isChecked() or self._jpg_cb.isChecked()
        self._accept_btn.setEnabled(any_selected and bool(self._dest_edit.text().strip()))

    def selected_items(self) -> set[str]:
        items: set[str] = set()
        if self._excel_cb.isChecked():
            items.add("excel")
        if self._csv_cb.isChecked():
            items.add("csv")
        if self._jpg_cb.isChecked():
            items.add("jpg")
        return items

    def destination(self) -> Path:
        return Path(self._dest_edit.text().strip())
