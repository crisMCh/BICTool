"""A titled sidebar section whose content can be collapsed by clicking its
header - so finished workflow steps take one line instead of a whole box."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QFrame):
    """Header button (▾/▸ + title) above a content widget. `content_layout`
    is where the section's controls go."""

    toggled = Signal(bool)  # expanded?

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("collapsibleSection")
        self._title = title
        self._header = QToolButton()
        self._header.setObjectName("sectionHeader")
        self._header.setCheckable(True)
        self._header.setChecked(True)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.ArrowType.DownArrow)
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._header.setText(title)
        self._header.toggled.connect(self._on_header_toggled)

        self._content = QWidget()
        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(8, 2, 8, 6)
        self.content_layout.setSpacing(4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._content)

    def title(self) -> str:
        return self._title

    def setTitle(self, title: str) -> None:  # noqa: N802 - mirrors QGroupBox
        self._title = title
        self._header.setText(title)

    def is_expanded(self) -> bool:
        return self._header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._header.setChecked(expanded)

    def header(self) -> QToolButton:
        return self._header

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt override
        # The header and the empty parts of the content show the section's help.
        super().setToolTip(text)
        self._header.setToolTip(f"{text}\n\nClick to collapse or expand." if text else "")

    def _on_header_toggled(self, expanded: bool) -> None:
        self._header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content.setVisible(expanded)
        self.toggled.emit(expanded)
