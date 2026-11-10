"""Settings > Plug-ins: the plug-ins found at start, their status, and tick
boxes to switch them on or off (applies after a restart). A view over
`PluginsViewModel`; `main_window` stores the result.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from bictool.presentation.viewmodels.plugins_viewmodel import PluginsViewModel
from bictool.presentation.views.text_scale import px

# Same palette as `main_window.stylesheet()` - duplicated rather than imported
# to avoid a circular import, matching `export_dialog.py`'s own precedent.
BG = "#11151a"
SURFACE = "#181d23"
SURFACE_RAISED = "#20262d"
BORDER = "#2b323a"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"
ACCENT = "#49c8de"

COLUMNS = ["On", "Plug-in", "Version", "Status", "Details", "From"]


def _stylesheet() -> str:
    return f"""
QDialog {{ background: {BG}; color: {TEXT}; font-size: {px(12)}px; }}
QLabel {{ color: {TEXT}; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; font-size: {px(11)}px; }}
QTableWidget {{ background: {SURFACE}; color: {TEXT}; gridline-color: {BORDER}; border: 1px solid {BORDER}; }}
QHeaderView::section {{ background: {SURFACE_RAISED}; color: {TEXT_MUTED}; border: none; padding: 4px; }}
QPushButton {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 11px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton#pluginsAcceptBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
"""


class PluginsDialog(QDialog):
    def __init__(self, view_model: PluginsViewModel, parent=None) -> None:
        super().__init__(parent)
        self._vm = view_model
        self.setWindowTitle("Plug-ins")
        self.setStyleSheet(_stylesheet())
        self.resize(px(760), px(380))

        layout = QVBoxLayout(self)
        intro = QLabel(
            f"{view_model.summary()} Plug-ins add values or WM/GM methods; their values are descriptive and "
            "never change the study verdict. Only use plug-ins from people you trust - they run with "
            "BICTool's access to the images. Changes apply when BICTool is restarted."
        )
        intro.setWordWrap(True)
        intro.setProperty("role", "muted")
        layout.addWidget(intro)

        rows = view_model.rows()
        self._table = QTableWidget(len(rows), len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for r, row in enumerate(rows):
            on = QTableWidgetItem()
            flags = Qt.ItemFlag.ItemIsEnabled | (Qt.ItemFlag.ItemIsUserCheckable if row.switchable else Qt.ItemFlag.NoItemFlags)
            on.setFlags(flags)
            on.setCheckState(Qt.CheckState.Checked if row.enabled else Qt.CheckState.Unchecked)
            on.setData(Qt.ItemDataRole.UserRole, row.name)
            self._table.setItem(r, 0, on)
            for c, text in enumerate((row.name, row.version, row.status, row.details, row.source), start=1):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self._table.setItem(r, c, item)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, 1)

        folder_row = QHBoxLayout()
        folder = view_model.folder
        self._folder_label = QLabel(f"Plug-in folder: {folder}" if folder else "No plug-in folder")
        self._folder_label.setProperty("role", "muted")
        self._folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        folder_row.addWidget(self._folder_label, 1)
        open_btn = QPushButton("Open Folder")
        open_btn.setEnabled(folder is not None)
        open_btn.clicked.connect(self._on_open_folder_clicked)
        folder_row.addWidget(open_btn)
        layout.addLayout(folder_row)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        accept_btn = button_box.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        accept_btn.setObjectName("pluginsAcceptBtn")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self._vm.set_enabled(item.data(Qt.ItemDataRole.UserRole), item.checkState() == Qt.CheckState.Checked)

    def _on_open_folder_clicked(self) -> None:
        folder = self._vm.folder
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
