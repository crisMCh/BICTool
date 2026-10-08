"""Which slices may get automatic ROIs: only those with enough brain.

At the ends of the brain the auto-ROI search still finds "structures", but
the measurement there is unreliable:

* at the skull base the slice holds a small brain cross-section between the
  petrous bones, whose beam-hardening streaks raise the noise (Barrett &
  Keat 2004, "Artifacts in CT: recognition and avoidance", RadioGraphics
  24:1679-1691), and there is no
  visible falx line;
* at the vertex the slice cuts the skull tangentially (partial volume with
  bone) and holds little white matter, so ROI2 lands on sulci/cortex rather
  than homogeneous tissue;
* the segmentation itself is least reliable on these small cross-sections.

Rule: a slice is eligible when its brain area (the union of the CNN
segmentation's labels on that slice, `merge_segmentation_labels`) is at
least `MIN_BRAIN_FRACTION` of the study's largest brain slice. A relative
area adapts to head size and pixel spacing. Per slice rather than the
volume's largest 3D component (`brain_mask_from_segmentation`) so that a
partly segmented volume works too; on the 636 segmented slices of the 15
rated studies both give identical eligibility.

The 20 % was chosen with a sensitivity analysis on the 15 rated studies
(docs/qa_method/head_ct_classifier.md, Sec. 6.5): the falx slices below 20 %
are noisy more than twice as often as the rest (26 % vs 10 % above twice the
study median), while slices between 20 and 40 % are not; no study verdict
changes for thresholds from 10 to 30 %; at 20 % 5 % of the slices are
dropped. Manual ROIs are not restricted - the user decides there.

Pure NumPy/SciPy - no Qt, no file I/O.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from bictool.domain.segmentation.segmentation_brain_mask import merge_segmentation_labels

MIN_BRAIN_FRACTION = 0.2


def brain_area_fractions(masks_by_slice: Mapping[int, dict[str, np.ndarray]]) -> dict[int, float]:
    """Brain area of each segmented slice / the largest one (0 for all
    slices when no slice has brain)."""
    areas = {index: int(merge_segmentation_labels(masks).sum()) if masks else 0 for index, masks in masks_by_slice.items()}
    largest = max(areas.values(), default=0)
    return {index: (area / largest if largest else 0.0) for index, area in areas.items()}


def eligible_slices(
    masks_by_slice: Mapping[int, dict[str, np.ndarray]], min_fraction: float = MIN_BRAIN_FRACTION
) -> set[int]:
    """The segmented slices that may get automatic ROIs. With a partly
    segmented volume, "largest" means the largest slice segmented so far."""
    return {
        index for index, fraction in brain_area_fractions(masks_by_slice).items()
        if fraction >= min_fraction and fraction > 0
    }
