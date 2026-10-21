"""Does the anatomy segmentation of a study look plausible? Study-level, provisional, descriptive.

The anatomy CNN was trained on normal-looking brains (and Alzheimer's). Where it struggles - lesions, artefacts,
noise - its labels break into many scattered pieces: a lobe becomes a patchwork, specks of one structure appear
inside another. A clean segmentation has a few large, compact regions per slice. BICTool is not a diagnostic
tool, so this only says *whether* to check the segmentation, not why.

**Label fragmentation** (per slice): for every label, the pieces of at least `MIN_PIECE_MM2` beyond its largest
piece, summed over the labels. Study value = median over the slices with enough brain (the automatic-ROI rule,
>= 20 % of the largest brain slice). Above `FRAGMENTATION_LIMIT` the segmentation may be unreliable.

Evidence (2026-10-20, 16 studies, docs/qa_method/segmentation_plausibility.md): the four studies whose
segmentation visibly falls apart (motion artefacts, acute SDH, stroke, implants) 34-40; all others 9-25. Label
sizes did not separate the studies (they follow anatomy and age, e.g. CSF spaces in atrophy). Lobe "islands"
(a piece of one lobe surrounded by another) are reported but don't decide - they also occur in normal studies.
Thresholds provisional. Pure NumPy/SciPy, no I/O; slice indices 0-based.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from bictool.domain.roi_selection.slice_eligibility import eligible_slices

MIN_PIECE_MM2 = 20.0
FRAGMENTATION_LIMIT = 30.0      # provisional
WORST_SLICES = 3                # the most fragmented slices, reported as where to look
LOBES = ("Frontal Lobe", "Parietal Lobe", "Occipital Lobe", "Temporal Lobe")
_ISLAND_SURROUNDED = 0.9        # share of a piece's border that one other lobe must cover


@dataclass(frozen=True)
class SegmentationPlausibility:
    fragmentation: float                 # median extra pieces per slice (study value)
    slices: int                          # slices evaluated
    worst_slices: tuple[int, ...]        # most fragmented slices, most fragmented first
    lobe_islands: int                    # pieces of a lobe surrounded by another lobe (all evaluated slices)

    @property
    def unreliable(self) -> bool:
        return self.slices > 0 and self.fragmentation > FRAGMENTATION_LIMIT


def slice_fragments(masks: Mapping[str, np.ndarray], pixel_spacing_mm: float) -> tuple[int, int]:
    """(extra pieces >= MIN_PIECE_MM2 beyond each label's largest, lobe islands) on one slice."""
    min_px = MIN_PIECE_MM2 / (pixel_spacing_mm * pixel_spacing_mm)
    present = {name: mask for name, mask in masks.items() if mask is not None and mask.any()}
    extra = islands = 0
    for name, mask in present.items():
        labels, count = ndi.label(mask)
        if count <= 1:
            continue
        sizes = np.bincount(labels.ravel())[1:]
        big = np.flatnonzero(sizes >= min_px)
        extra += max(len(big) - 1, 0)
        if name not in LOBES or len(big) <= 1:
            continue
        largest = int(np.argmax(sizes))
        for k in big:
            if k == largest:
                continue
            piece = labels == k + 1
            ring = ndi.binary_dilation(piece, iterations=2) & ~piece
            if any(other != name and other in LOBES and (present[other] & ring).sum() >= _ISLAND_SURROUNDED * ring.sum()
                   for other in present):
                islands += 1
    return extra, islands


def assess_plausibility(
    masks_by_slice: Mapping[int, dict[str, np.ndarray]], pixel_spacing_mm: float
) -> SegmentationPlausibility:
    """Study-level label fragmentation of a segmented volume (see the module docstring)."""
    per_slice = {index: slice_fragments(masks_by_slice[index], pixel_spacing_mm) for index in sorted(eligible_slices(masks_by_slice))}
    if not per_slice:
        return SegmentationPlausibility(0.0, 0, (), 0)
    extra = {index: value[0] for index, value in per_slice.items()}
    worst = tuple(sorted(extra, key=lambda index: (-extra[index], index))[:WORST_SLICES])
    return SegmentationPlausibility(
        fragmentation=float(np.median(list(extra.values()))),
        slices=len(per_slice),
        worst_slices=worst,
        lobe_islands=sum(value[1] for value in per_slice.values()),
    )
