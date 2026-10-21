"""DICOM tag browser dialog (View > DICOM Tags...) - a raw, unfiltered dump
of the currently displayed slice's header, for anyone who needs to see
exactly what's in the file rather than this app's own curated summary (the
corner overlay - see `main_window._update_viewer_overlay`).
"""

from __future__ import annotations

import pydicom
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from bictool.presentation.views.text_scale import px

BG = "#11151a"
SURFACE = "#181d23"
SURFACE_RAISED = "#20262d"
BORDER = "#2b323a"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"


def _stylesheet() -> str:
    return f"""
QDialog {{ background: {BG}; color: {TEXT}; font-size: {px(12)}px; }}
QLineEdit {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 8px; color: {TEXT};
}}
QTableWidget {{
    background: {SURFACE}; border: 1px solid {BORDER}; color: {TEXT};
    font-family: "IBM Plex Mono", Consolas, monospace; font-size: {px(11)}px;
}}
QHeaderView::section {{
    background: {SURFACE}; color: {TEXT_MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 4px; font-weight: 600; font-size: {px(10)}px;
}}
"""


class DicomTagsDialog(QDialog):
    """Read-only browser for every top-level DICOM element in one file's
    header - tag, VR, keyword, and value, in tag order, with a live text
    filter. Pixel data is never read or dumped as text (see `_format_value`)
    even if present in `dataset` - only its byte length is shown.
    """

    def __init__(self, dataset: pydicom.Dataset, *, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(_stylesheet())
        self.resize(640, 560)

        layout = QVBoxLayout(self)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter by tag, keyword, or value…")
        self._filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter_edit)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Tag", "VR", "Keyword", "Value"])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)

        self._populate(dataset)

    def _populate(self, dataset: pydicom.Dataset) -> None:
        elements = list(dataset)  # top-level, tag order - matches most tag-browser tools' default view
        self._table.setRowCount(len(elements))
        for row, element in enumerate(elements):
            values = (str(element.tag), element.VR or "", element.keyword or "", _format_value(element))
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(TEXT))
                self._table.setItem(row, col, item)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self._table.rowCount()):
            if not needle:
                self._table.setRowHidden(row, False)
                continue
            haystack = " ".join(self._table.item(row, col).text().lower() for col in range(self._table.columnCount()))
            self._table.setRowHidden(row, needle not in haystack)


def _format_value(element: pydicom.DataElement) -> str:
    if element.VR == "SQ":
        return f"<sequence, {len(element.value)} item(s)>"
    if element.keyword == "PixelData":
        length = len(element.value) if element.value is not None else 0
        return f"<binary, {length} byte(s)>"
    try:
        return str(element.value)
    except Exception:
        return "<unreadable>"
