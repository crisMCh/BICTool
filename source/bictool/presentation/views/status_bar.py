"""The window's bottom bar for long-running work (DICOM loading, segmentation,
ROI placement, QA run) - the usual desktop pattern:

* idle: the study message on the left ("Loaded: ..."), and each task's last
  state small on the right ("Segmentation: Done", "QA done: 33 ROI pair(s)");
* while a task runs: one progress bar across the whole bar, replacing the
  study message, with the task's text in front and Cancel (where the task
  can be stopped) at the end. With several tasks running, the one started
  last is shown.

Each task is an `ActivityItem` whose label the existing workflow code writes
plain status text into; the item derives its state from that text, so the
code reporting progress does not need to know about progress bars:

* empty or "—" = nothing to show;
* text matching the item's `busy` pattern = running;
* an "n/m" in the text drives the bar's value, otherwise it is indeterminate.
"""

from __future__ import annotations

import itertools
import re

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

_COUNT = re.compile(r"(\d+)\s*/\s*(\d+)")
_IDLE_TEXTS = ("", "—")
_ORDER = itertools.count(1)


class _ActivityLabel(QLabel):
    def __init__(self, item: ActivityItem) -> None:
        super().__init__()
        self._item = item
        self.setProperty("role", "muted")

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        super().setText(text)
        self._item._on_text(text)


class ActivityItem(QWidget):
    """One task. As a widget it is the task's idle display ("prefix label");
    while it runs, `StatusPanel` shows it in the full-width progress bar."""

    def __init__(self, panel: StatusPanel, prefix: str, busy: str) -> None:
        super().__init__()
        self._panel = panel
        self._busy_pattern = re.compile(busy)
        self.prefix = prefix
        self.busy = False
        self.started = 0  # order in which tasks became busy
        self.count: tuple[int, int] | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)
        if prefix:
            prefix_label = QLabel(prefix)
            prefix_label.setStyleSheet("font-weight: 600;")
            layout.addWidget(prefix_label)
        self.label = _ActivityLabel(self)
        layout.addWidget(self.label)
        self.setVisible(False)

    def text(self) -> str:
        return self.label.text()

    def is_busy(self) -> bool:
        return self.busy

    def add_cancel(self, on_click) -> QPushButton:
        """A small Cancel button at the end of the progress bar, hidden until
        the workflow shows it (only while the task can be stopped)."""
        button = QPushButton("Cancel")
        button.setObjectName("statusCancelBtn")
        button.setVisible(False)
        button.clicked.connect(on_click)
        self._panel.add_cancel_button(button)
        return button

    def _on_text(self, text: str) -> None:
        busy = bool(self._busy_pattern.search(text))
        if busy and not self.busy:
            self.started = next(_ORDER)
        self.busy = busy
        match = _COUNT.search(text) if busy else None
        self.count = (int(match.group(1)), int(match.group(2))) if match else None
        self._panel.refresh()


class StatusPanel(QWidget):
    """The bottom bar's content: study message / full-width progress bar on
    the left, the tasks' idle states on the right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[ActivityItem] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(8)

        self.message = ActivityItem(self, "", busy=r"^Loading")  # the study message ("Loaded: ...")
        self._items.append(self.message)
        layout.addWidget(self.message, 1)

        self._task_label = QLabel()
        self._task_label.setStyleSheet("font-weight: 600;")
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(12)
        layout.addWidget(self._task_label)
        layout.addWidget(self._bar, 1)
        self._cancel_layout = QHBoxLayout()
        self._cancel_layout.setSpacing(4)
        layout.addLayout(self._cancel_layout)

        self._idle_layout = QHBoxLayout()
        self._idle_layout.setSpacing(0)
        layout.addLayout(self._idle_layout)
        self.refresh()

    def add_item(self, prefix: str, busy: str) -> ActivityItem:
        item = ActivityItem(self, prefix, busy)
        self._items.append(item)
        self._idle_layout.addWidget(item)
        self.refresh()
        return item

    def add_cancel_button(self, button: QPushButton) -> None:
        self._cancel_layout.addWidget(button)

    def active_item(self) -> ActivityItem | None:
        busy = [item for item in self._items if item.busy]
        return max(busy, key=lambda item: item.started) if busy else None

    def progress_bar(self) -> QProgressBar:
        return self._bar

    def refresh(self) -> None:
        active = self.active_item()
        running = active is not None
        self._task_label.setVisible(running)
        self._bar.setVisible(running)
        if running:
            self._task_label.setText(f"{active.prefix} {active.text()}".strip())
            if active.count:
                done, total = active.count
                self._bar.setRange(0, max(total, 1))
                self._bar.setValue(min(done, total))
            else:
                self._bar.setRange(0, 0)  # indeterminate ("busy") animation
        # The study message gives way to the bar; tasks show their idle state.
        self.message.setVisible(not running and self.message.text() not in _IDLE_TEXTS)
        for item in self._items[1:]:
            item.setVisible(not item.busy and item.text() not in _IDLE_TEXTS)
