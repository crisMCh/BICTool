"""Tests `viewmodels.artefact_text` - the possible-artefact lines on the Results page."""

from bictool.domain.artefacts.artefact_flags import ArtefactFlags
from bictool.presentation.viewmodels.artefact_text import BAND_NOTE, artefact_basics, artefact_details


def test_nothing_found_gives_no_basic_lines_but_says_so_in_the_details():
    flags = ArtefactFlags(band_depths={4: 1.2, 5: 2.0})
    assert artefact_basics(flags) == []
    details = artefact_details(flags)
    assert details[0].startswith("Metal: none") and "5: 1.2, 6: 2.0" in details[1] and BAND_NOTE in details


def test_metal_and_bands_are_listed_with_1_based_slice_ranges():
    flags = ArtefactFlags(metal_slices=(13, 14, 15, 20), metal_max_hu=3071.0, band_depths={4: 5.0, 5: 4.1, 7: 1.0})
    basics = artefact_basics(flags)
    assert basics[0] == "Artefacts (metal): on slices 14–16, 21 - expect streaks nearby."
    assert basics[1] == "Artefacts (posterior fossa): possible dark band across the brainstem on slices 5–6."
    assert "highest 3071 HU" in artefact_details(flags)[0]


def test_without_flags_there_is_nothing_to_say():
    assert artefact_basics(None) == [] and artefact_details(None) == []


def test_slice_notes_name_the_artefacts_on_that_slice():
    from bictool.presentation.viewmodels.artefact_text import slice_artefact_notes

    flags = ArtefactFlags(metal_slices=(3,), metal_max_hu=3071.0, band_depths={3: 4.0, 4: 1.0})
    assert slice_artefact_notes(flags, 3) == ["metal", "possible brainstem band"]
    assert slice_artefact_notes(flags, 4) == [] and slice_artefact_notes(None, 3) == []
