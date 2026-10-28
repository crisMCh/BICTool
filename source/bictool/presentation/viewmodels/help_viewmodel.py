"""The help window's state: which documents are shown (the user manual in the
chosen language, the technical reference in English), which manual section
belongs to which control (F1), and the full-text search.

The documents come from `integration.help_files` (built by
docs/manual/build_manual.py); the View (`views/help_window.py`) only renders
them. QtCore only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from bictool.integration.help_files import HelpDocument, HelpSection

USER_GUIDE = "user_guide"
TECHNICAL_REFERENCE = "technical_reference"
LANGUAGES = {"en": "English", "de": "Deutsch"}
DEFAULT_LANGUAGE = "en"
START_TOPIC = "main-window"

# Main-window controls -> the user-manual section about them (heading ids,
# identical in en.md and de.md). Keys are the window's attribute names, as in
# `help_text.TOOLTIPS`. F1 walks from the control under the mouse up through
# its parents and takes the first one listed here.
WIDGET_TOPICS = {
    "_series_box": "open-study",
    "_series_combo": "open-study",
    "_step1_section": "segmentation",
    "_anatomical_box": "segmentation",
    "_segmentation_status_label": "segmentation",
    "_cancel_segment_volume_btn": "segmentation",
    "_select_segments_btn": "segmentation",
    "_show_segments_cb": "segmentation",
    "_wmgm_box": "segmentation",
    "_white_matter_cb": "segmentation",
    "_gray_matter_cb": "segmentation",
    "_csf_cb": "segmentation",
    "_brain_mask_cb": "segmentation",
    "_brain_tissue_method_combo": "wmgm-methods",
    "_brain_tissue_method_hint": "wmgm-methods",
    "_brain_tissue_settings_btn": "wmgm-methods",
    "_roi_section": "roi-selection",
    "_auto_rois_btn": "roi-selection",
    "_roi_details_btn": "roi-selection",
    "_roi_table": "roi-selection",
    "_delete_roi_btn": "roi-selection",
    "_restore_auto_btn": "roi-selection",
    "_confirm_box": "roi-selection",
    "_confirm_rois_cb": "roi-selection",
    "_roi_viewport": "roi-selection",
    "_metrics_box": "qa-run",
    "_run_all_btn": "qa-run",
    "_cancel_run_btn": "qa-run",
    "_metrics_table": "measurements",
    "_qa_recipe_label": "program-settings",
    "_trace_box": "log-file",
    "_trace_log": "log-file",
    "_viewer_tab_btn": "main-window",
    "_results_tab_btn": "results-per-slice",
    "_window_preset_combo": "viewing",
    "_slice_slider": "viewing",
    "_results_prev_btn": "results-per-slice",
    "_results_next_btn": "results-per-slice",
    "_results_slider": "results-per-slice",
    "_results_pair_buttons": "results-per-slice",
    "_results_curves_stack": "results-per-slice",
    "_curves_view_btn": "results-per-slice",
    "_steps_view_btn": "results-per-slice",
    "_slice_steps_panel": "results-per-slice",
    "_study_table": "study-summary",
    "_study_verdict_label": "study-summary",
    "_verdict_details_btn": "study-summary",
    "_study_provisional_label": "study-summary",
}


@dataclass(frozen=True)
class Topic:
    document: HelpDocument
    anchor: str


@dataclass(frozen=True)
class SearchHit:
    document: HelpDocument
    section: HelpSection
    snippet: str


class HelpViewModel(QObject):
    language_changed = Signal(str)

    def __init__(self, documents: Sequence[HelpDocument], language: str = DEFAULT_LANGUAGE,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._documents = list(documents)
        self._language = language if language in LANGUAGES else DEFAULT_LANGUAGE

    @property
    def available(self) -> bool:
        return bool(self._documents)

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        if language in LANGUAGES and language != self._language:
            self._language = language
            self.language_changed.emit(language)

    def documents(self) -> list[HelpDocument]:
        """What the window shows: the user manual in the chosen language (English if missing), then the
        technical reference."""
        shown = []
        for key in (USER_GUIDE, TECHNICAL_REFERENCE):
            matches = [d for d in self._documents if d.key == key]
            preferred = [d for d in matches if d.language == self._language] or \
                [d for d in matches if d.language == DEFAULT_LANGUAGE] or matches
            if preferred:
                shown.append(preferred[0])
        return shown

    def document(self, key: str) -> HelpDocument | None:
        return next((d for d in self.documents() if d.key == key), None)

    def topic(self, anchor: str, key: str = USER_GUIDE) -> Topic | None:
        """`anchor` in document `key`, or that document's start when the anchor is unknown."""
        document = self.document(key)
        if document is None:
            return None
        known = any(section.id == anchor for section in document.sections)
        return Topic(document, anchor if known else "")

    def topic_for_widgets(self, names: Sequence[str], fallback: str = START_TOPIC) -> Topic | None:
        """The topic of the first named control that has one (`names` from the control under the mouse
        outwards), else `fallback`."""
        anchor = next((WIDGET_TOPICS[name] for name in names if name in WIDGET_TOPICS), fallback)
        return self.topic(anchor)

    def search(self, query: str, limit: int = 50) -> list[SearchHit]:
        """Sections of the shown documents that contain every word of `query` (case-insensitive), best
        first: words in the title count most, then how often the words occur."""
        words = [w for w in query.lower().split() if w]
        if not words:
            return []
        scored = []
        for order, document in enumerate(self.documents()):
            for position, section in enumerate(document.sections):
                title, text = section.title.lower(), section.text.lower()
                if not all(w in title or w in text for w in words):
                    continue
                score = sum(10 * (w in title) + text.count(w) for w in words)
                scored.append((-score, order, position, SearchHit(document, section, _snippet(section.text, words))))
        scored.sort(key=lambda item: item[:3])
        return [hit for *_, hit in scored[:limit]]


def _snippet(text: str, words: Sequence[str], width: int = 160) -> str:
    """About `width` characters of `text` around the first occurrence of any of `words`."""
    lower = text.lower()
    found = [i for i in (lower.find(w) for w in words) if i >= 0]
    start = max(min(found, default=0) - width // 3, 0)
    end = start + width
    if start > 0 and " " in text[start:min(found, default=start)]:
        start = text.index(" ", start) + 1  # whole words only
    if end < len(text) and " " in text[start:end]:
        end = text.rindex(" ", start, end)
    snippet = text[start:end].strip()
    return ("… " if start > 0 else "") + snippet + (" …" if end < len(text) else "")
