"""The help window's state: documents per language, F1 topics, search."""

from pathlib import Path

from bictool.integration.help_files import HelpDocument, HelpSection
from bictool.presentation.viewmodels.help_viewmodel import HelpViewModel


def _document(key, language, sections):
    return HelpDocument(key, language, f"{key} {language}", Path(f"{key}_{language}.html"),
                        tuple(HelpSection(i, title, 2, text) for i, title, text in sections))


DOCUMENTS = [
    _document("user_guide", "en", [
        ("main-window", "3. The main window", "Menu bar and sidebar."),
        ("roi-selection", "4.3 Step 2 - ROI selection", "Place the ROIs automatically. Noise ROI next to the falx."),
    ]),
    _document("user_guide", "de", [
        ("main-window", "3. Das Hauptfenster", "Menüleiste und Seitenleiste."),
        ("roi-selection", "4.3 Schritt 2 - ROI-Auswahl", "ROIs automatisch platzieren."),
    ]),
    _document("technical_reference", "en", [
        ("tr-nps", "8.2 Power spectra: NPS and PS", "The noise power spectrum of ROI2. noise noise noise"),
        ("tr-roi2", "7.3 ROI2 in homogeneous tissue", "White matter next to the falx; noise ROI."),
    ]),
]


def test_shows_the_manual_in_the_chosen_language_and_the_english_reference():
    view_model = HelpViewModel(DOCUMENTS, "de")
    assert [(d.key, d.language) for d in view_model.documents()] == [
        ("user_guide", "de"), ("technical_reference", "en"),
    ]
    changes = []
    view_model.language_changed.connect(changes.append)
    view_model.set_language("en")
    view_model.set_language("en")
    view_model.set_language("fr")
    assert changes == ["en"] and view_model.documents()[0].language == "en"


def test_falls_back_to_english_and_to_the_start_of_a_document():
    view_model = HelpViewModel([DOCUMENTS[0], DOCUMENTS[2]], "de")
    assert view_model.documents()[0].language == "en"
    assert view_model.topic("no-such-section").anchor == ""
    assert view_model.topic("tr-nps", "technical_reference").document.key == "technical_reference"
    assert HelpViewModel([]).topic("main-window") is None and not HelpViewModel([]).available


def test_f1_takes_the_first_control_with_a_topic_from_the_mouse_outwards():
    view_model = HelpViewModel(DOCUMENTS)
    assert view_model.topic_for_widgets(["_unknown", "_auto_rois_btn", "_roi_section"]).anchor == "roi-selection"
    assert view_model.topic_for_widgets([]).anchor == "main-window"
    assert view_model.topic_for_widgets([], fallback="roi-selection").anchor == "roi-selection"


def test_search_needs_every_word_and_ranks_titles_first():
    view_model = HelpViewModel(DOCUMENTS)
    hits = view_model.search("noise ROI")
    # "roi" in the title counts 10, each occurrence in the text 1: 13, 12, 5
    assert [hit.section.id for hit in hits] == ["roi-selection", "tr-roi2", "tr-nps"]
    assert "noise" in hits[0].snippet.lower()
    assert view_model.search("   ") == []
    assert view_model.search("falx spectrum") == []


def test_search_covers_the_manual_of_the_chosen_language():
    view_model = HelpViewModel(DOCUMENTS, "de")
    assert [hit.section.id for hit in view_model.search("platzieren")] == ["roi-selection"]
    assert view_model.search("Place the ROIs") == []
