"""Wording of the study-level segmentation check."""

from bictool.domain.segmentation.segmentation_plausibility import SegmentationPlausibility
from bictool.presentation.viewmodels.segmentation_plausibility_text import (
    PLAUSIBILITY_NOTE,
    plausibility_basics,
    plausibility_details,
    plausibility_line,
)

BAD = SegmentationPlausibility(fragmentation=38.0, slices=28, worst_slices=(13, 18, 11), lobe_islands=14)
OK = SegmentationPlausibility(fragmentation=14.5, slices=28, worst_slices=(16,), lobe_islands=0)


def test_box_line_and_summary():
    assert plausibility_line(BAD) == "Segmentation may be unreliable in this study - check the overlays."
    assert plausibility_line(OK) == "Segmentation looks consistent."
    assert plausibility_line(None) == "" and plausibility_line(SegmentationPlausibility(0.0, 0, (), 0)) == ""
    assert plausibility_basics(OK) == ["Segmentation (provisional): segmentation looks consistent."]
    assert plausibility_basics(None) == []


def test_details_name_the_value_and_where_to_look():
    details = plausibility_details(BAD)
    assert details[0].startswith("Segmentation fragmentation 38.0 pieces per slice (limit 30, 28 slices)")
    assert "most fragmented slices: 14, 19, 12" in details[0] and details[-1] == PLAUSIBILITY_NOTE
