"""The help window (F1, "? Help"): the user manual and the technical
reference, with a contents tree, full-text search and back/forward.

A separate, non-modal window, so it can stay open next to BICTool. The
pages are HTML built by docs/manual/build_manual.py; state and search live
in `viewmodels/help_viewmodel.py`. The pages themselves are shown light (like
a printed page) inside the dark window frame.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bictool.presentation.viewmodels.help_viewmodel import LANGUAGES, HelpViewModel, Topic
from bictool.presentation.views.text_scale import px

# Same palette as `main_window.stylesheet()` - duplicated rather than imported
# to avoid a circular import (main_window constructs this window), as in the
# other views.
BG = "#11151a"
SURFACE = "#171c21"
SURFACE_RAISED = "#20262d"
BORDER = "#2b323a"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"
ACCENT = "#49c8de"
PAGE_BG = "#ffffff"
PAGE_TEXT = "#1d2329"

_ROLE_TOPIC = Qt.ItemDataRole.UserRole


def _stylesheet() -> str:
    return f"""
QWidget#helpWindow {{ background: {BG}; color: {TEXT}; font-size: {px(12)}px; }}
QLabel {{ color: {TEXT}; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; }}
QPushButton, QComboBox {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px 10px; color: {TEXT};
}}
QPushButton:hover, QComboBox:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; }}
QComboBox QAbstractItemView {{ background: {SURFACE_RAISED}; color: {TEXT}; selection-background-color: {ACCENT}; }}
QLineEdit {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 6px; color: {TEXT}; }}
QTreeWidget, QListWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; color: {TEXT}; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {ACCENT}; color: #08181c; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: {SURFACE}; color: {TEXT_MUTED}; padding: 5px 12px; border: 1px solid {BORDER}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTextBrowser {{ background: {PAGE_BG}; color: {PAGE_TEXT}; border: 1px solid {BORDER}; }}
"""


class HelpWindow(QWidget):
    def __init__(self, view_model: HelpViewModel, parent: QWidget | None = None) -> None:
        # A top-level window of its own (not a child widget), also when a parent is given.
        super().__init__(parent, Qt.WindowType.Window)
        self.view_model = view_model
        self.setObjectName("helpWindow")
        self.setWindowTitle("BICTool Help")
        self.setStyleSheet(_stylesheet())
        self.resize(px(1100), px(760))

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self._back_btn = QPushButton("◀ Back")
        self._forward_btn = QPushButton("Forward ▶")
        self._home_btn = QPushButton("Start page")
        toolbar.addWidget(self._back_btn)
        toolbar.addWidget(self._forward_btn)
        toolbar.addWidget(self._home_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("Manual language:"))
        self._language_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self._language_combo.addItem(name, code)
        self._language_combo.setCurrentIndex(max(self._language_combo.findData(view_model.language), 0))
        self._language_combo.setToolTip("Language of the user manual (the technical reference is in English)")
        toolbar.addWidget(self._language_combo)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._side_tabs = QTabWidget()
        self._contents_tree = QTreeWidget()
        self._contents_tree.setHeaderHidden(True)
        self._side_tabs.addTab(self._contents_tree, "Contents")
        search_page = QWidget()
        search_layout = QVBoxLayout(search_page)
        search_layout.setContentsMargins(0, 6, 0, 0)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search the manual and the technical reference")
        self._search_edit.setClearButtonEnabled(True)
        self._search_status = QLabel("")
        self._search_status.setProperty("role", "muted")
        self._search_results = QListWidget()
        self._search_results.setWordWrap(True)
        search_layout.addWidget(self._search_edit)
        search_layout.addWidget(self._search_status)
        search_layout.addWidget(self._search_results, 1)
        self._side_tabs.addTab(search_page, "Search")
        splitter.addWidget(self._side_tabs)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        splitter.addWidget(self._browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([px(300), px(800)])
        layout.addWidget(splitter, 1)

        self._back_btn.clicked.connect(self._browser.backward)
        self._forward_btn.clicked.connect(self._browser.forward)
        self._home_btn.clicked.connect(lambda: self.show_topic(view_model.topic_for_widgets([])))
        self._browser.backwardAvailable.connect(self._back_btn.setEnabled)
        self._browser.forwardAvailable.connect(self._forward_btn.setEnabled)
        self._back_btn.setEnabled(False)
        self._forward_btn.setEnabled(False)
        self._contents_tree.itemClicked.connect(self._on_item_chosen)
        self._contents_tree.itemActivated.connect(self._on_item_chosen)
        self._search_edit.textChanged.connect(self._on_search)
        self._search_results.itemClicked.connect(self._on_item_chosen)
        self._search_results.itemActivated.connect(self._on_item_chosen)
        self._language_combo.currentIndexChanged.connect(
            lambda: view_model.set_language(self._language_combo.currentData())
        )
        view_model.language_changed.connect(self._on_language_changed)
        QShortcut(QKeySequence.StandardKey.Find, self, activated=self._focus_search)
        QShortcut(QKeySequence.StandardKey.Back, self, activated=self._browser.backward)
        QShortcut(QKeySequence.StandardKey.Forward, self, activated=self._browser.forward)

        self._fill_contents()
        if not view_model.available:
            self._browser.setHtml(
                "<h2>Help not available</h2><p>The help pages were not found. The PDF manuals are in the "
                "<i>manual</i> folder next to BICTool.exe (or build them with docs/manual/build_manual.py).</p>"
            )

    # ---- navigation ----

    def show_topic(self, topic: Topic | None) -> None:
        """Opens `topic` and brings the window to the front."""
        if topic is not None:
            url = QUrl.fromLocalFile(str(topic.document.path))
            if topic.anchor:
                url.setFragment(topic.anchor)
            if url == self._browser.source():
                self._browser.scrollToAnchor(topic.anchor)  # same page and anchor: setSource would not scroll back
            else:
                self._browser.setSource(url)
            self._select_in_tree(topic)
        self.show()
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def current_url(self) -> QUrl:
        return self._browser.source()

    def _focus_search(self) -> None:
        self._side_tabs.setCurrentIndex(1)
        self._search_edit.setFocus()
        self._search_edit.selectAll()

    def _on_item_chosen(self, item) -> None:
        topic = item.data(0, _ROLE_TOPIC) if isinstance(item, QTreeWidgetItem) else item.data(_ROLE_TOPIC)
        if topic is not None:
            self.show_topic(topic)

    def _on_language_changed(self, _language: str) -> None:
        anchor = self._browser.source().fragment()
        on_user_guide = "user_guide" in self._browser.source().fileName()
        self._fill_contents()
        self._on_search(self._search_edit.text())
        if on_user_guide:
            self.show_topic(self.view_model.topic(anchor))

    # ---- contents and search ----

    def _fill_contents(self) -> None:
        self._contents_tree.clear()
        for document in self.view_model.documents():
            root = QTreeWidgetItem([document.title])
            root.setData(0, _ROLE_TOPIC, Topic(document, ""))
            self._contents_tree.addTopLevelItem(root)
            chapter = root
            for section in document.sections:
                item = QTreeWidgetItem([section.title])
                item.setData(0, _ROLE_TOPIC, Topic(document, section.id))
                if section.level <= 2:
                    root.addChild(item)
                    chapter = item
                else:
                    chapter.addChild(item)
            root.setExpanded(True)

    def _select_in_tree(self, topic: Topic) -> None:
        iterator = [self._contents_tree.topLevelItem(i) for i in range(self._contents_tree.topLevelItemCount())]
        while iterator:
            item = iterator.pop()
            shown = item.data(0, _ROLE_TOPIC)
            if shown is not None and shown.document.path == topic.document.path and shown.anchor == topic.anchor:
                self._contents_tree.setCurrentItem(item)
                self._contents_tree.scrollToItem(item)
                return
            iterator.extend(item.child(i) for i in range(item.childCount()))

    def _on_search(self, query: str) -> None:
        self._search_results.clear()
        if not query.strip():
            self._search_status.setText("")
            return
        hits = self.view_model.search(query)
        self._search_status.setText(f"{len(hits)} section(s) found" if hits else "Nothing found")
        for hit in hits:
            item = QListWidgetItem(f"{hit.section.title}  ({hit.document.title})\n{hit.snippet}")
            item.setData(_ROLE_TOPIC, Topic(hit.document, hit.section.id))
            self._search_results.addItem(item)
