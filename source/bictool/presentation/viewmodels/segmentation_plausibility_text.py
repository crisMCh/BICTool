"""Text for the study-level segmentation check (`domain.segmentation.segmentation_plausibility`): the line in the
Anatomical segmentation box, the study-summary line and the details. Provisional. Qt-free."""

from __future__ import annotations

from bictool.domain.segmentation.segmentation_plausibility import FRAGMENTATION_LIMIT, SegmentationPlausibility

UNRELIABLE_TEXT = "Segmentation may be unreliable in this study - check the overlays."
CONSISTENT_TEXT = "Segmentation looks consistent."

PLAUSIBILITY_NOTE = (
    "Segmentation check (provisional): the anatomy CNN was trained on normal-looking brains; where it struggles "
    "(lesions, artefacts, noise) its labels break into many scattered pieces. Fragmentation = pieces of at least "
    f"20 mm² per label beyond its largest piece, summed over the labels, median over the slices with enough brain; "
    f"above {FRAGMENTATION_LIMIT:g} the segmentation may be unreliable (limit from 16 studies). It does not say "
    "why, and it is not part of the verdict."
)


def plausibility_line(plausibility: SegmentationPlausibility | None) -> str:
    """The line in the Anatomical box ("" before the check ran or without slices to check)."""
    if plausibility is None or plausibility.slices == 0:
        return ""
    return UNRELIABLE_TEXT if plausibility.unreliable else CONSISTENT_TEXT


def plausibility_basics(plausibility: SegmentationPlausibility | None) -> list[str]:
    line = plausibility_line(plausibility)
    return [f"Segmentation (provisional): {line[0].lower()}{line[1:]}"] if line else []


def plausibility_details(plausibility: SegmentationPlausibility | None) -> list[str]:
    if plausibility is None or plausibility.slices == 0:
        return []
    worst = ", ".join(str(index + 1) for index in plausibility.worst_slices)
    return [
        f"Segmentation fragmentation {plausibility.fragmentation:.1f} pieces per slice (limit {FRAGMENTATION_LIMIT:g}, "
        f"{plausibility.slices} slices); most fragmented slices: {worst}; lobe islands: {plausibility.lobe_islands}.",
        PLAUSIBILITY_NOTE,
    ]
