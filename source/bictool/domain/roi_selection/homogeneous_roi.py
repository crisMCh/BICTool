"""Automatic placement of ROI2 - the "homogeneous" ROI that feeds the second
NPS measurement (`domain.results.study_quality.evaluate_slice_quality`) -
next to an already-placed ROI1 (`structured_roi`).

`architecture/layers_description.md`: "The ventricles and white matter can
be used for NPS analysis." Which one depends on ROI1's structure:

- **Septum pellucidum -> ventricle** (CNN "Ventricular System") within
  `_VENTRICLE_SEARCH_RADIUS_MM` of ROI1. As close to ROI1 as possible,
  preferably not touching it - touching/overlapping ROI1 is accepted only
  when the nearby ventricle has no room otherwise - and never containing a
  septum pixel. A far-away part of the ventricle doesn't count (on the real
  postop scan the only non-touching fit on two slices was in the atrium
  ~40 mm away): if the nearby ventricle can't hold even a minimum-size
  square, falls back to the white-matter search below.
- **Dural folds (falx) -> white matter**, from the WM/GM classification the
  caller supplies (whichever Brain Tissue method is selected). Pure white
  matter preferred; some gray matter is accepted when pure white matter would
  mean moving too far from ROI1 - traded off as distance + a gray-matter
  penalty (`_GRAY_MATTER_PENALTY_MM` per 100 % gray matter). Never touches
  ROI1, never contains a dural-fold or septum pixel (WM/GM classifiers
  label the dense falx as gray matter, which the penalty alone wouldn't
  keep out).

ROI2 is the same size as ROI1 (comparable NPS), shrinking toward
`min_size` only if nothing that size fits. Where ROI1 exists ROI2 always
gets a placement: if no region-based square fits at all, the least-bad
square near ROI1 is returned, flagged as `FALLBACK`.

All positions are scored at once per size with integral images.
Pure NumPy - no Qt, no file I/O.
"""

from __future__ import annotations

import numpy as np

from bictool.domain.roi_selection.structured_roi import (
    DEFAULT_MIN_SIZE,
    DURAL_FOLDS,
    SEPTUM_PELLUCIDUM,
    RoiCandidate,
    _integral,
)
from bictool.domain.segmentation.segmentation_brain_mask import merge_segmentation_labels

VENTRICLE = "Ventricular System"
WHITE_MATTER = "White Matter"
FALLBACK = "fallback"

# Fraction of a square's pixels allowed outside its region (label edges,
# small classification holes) before it no longer counts as homogeneous.
_VENTRICLE_TOLERANCE = 0.02
_WHITE_MATTER_TOLERANCE = 0.05
# Cost of gray matter in the white-matter search, in mm of extra distance
# per 100 % gray-matter pixels (10 % gray matter ~ 4 mm further away).
_GRAY_MATTER_PENALTY_MM = 40.0
# "Relative vicinity" of ROI1 for the ventricle search, center to center.
_VENTRICLE_SEARCH_RADIUS_MM = 25.0
_GROWTH_STEP = 2


def find_roi2_candidate(
    roi1: RoiCandidate,
    masks: dict[str, np.ndarray],
    pixel_spacing_mm: float,
    tissue_masks: dict[str, np.ndarray] | None = None,
    *,
    min_size: int = DEFAULT_MIN_SIZE,
) -> RoiCandidate | None:
    """ROI2 for `roi1` on the same slice. `masks` = the slice's CNN
    `SegmentationResult.masks`; `tissue_masks` = its WM/GM classification
    (`BrainTissueResult.masks`, needs "White Matter" and "Gray Matter").

    Returns a candidate whose `structure` is `VENTRICLE`, `WHITE_MATTER` or
    `FALLBACK`. Returns None only when the white-matter search is needed
    and `tissue_masks` isn't available yet - call again once it is.
    """
    shape = next(iter(masks.values())).shape
    sizes = _sizes(roi1.size, min_size)

    if roi1.structure == SEPTUM_PELLUCIDUM:
        ventricle = masks.get(VENTRICLE, np.zeros(shape, dtype=bool))
        septum = masks.get(SEPTUM_PELLUCIDUM, np.zeros(shape, dtype=bool))
        candidate = _ventricle_search(roi1, ventricle, septum, sizes, pixel_spacing_mm)
        if candidate is not None:
            return candidate

    if tissue_masks is None:
        return None
    forbidden = np.zeros(shape, dtype=bool)
    for name in (DURAL_FOLDS, SEPTUM_PELLUCIDUM):
        if name in masks:
            forbidden |= masks[name]
    white = tissue_masks[WHITE_MATTER] & ~forbidden
    gray = tissue_masks["Gray Matter"] & ~forbidden
    candidate = _white_matter_search(roi1, white, gray, forbidden, sizes, pixel_spacing_mm)
    if candidate is not None:
        return candidate
    return _least_bad(roi1, white | gray, merge_segmentation_labels(masks), sizes[-1], pixel_spacing_mm)


def _sizes(roi1_size: int, min_size: int) -> list[int]:
    """ROI1's size first, then shrinking in 2 px steps down to `min_size`."""
    sizes = [roi1_size]
    while sizes[-1] - _GROWTH_STEP >= min_size:
        sizes.append(sizes[-1] - _GROWTH_STEP)
    return sizes


def _ventricle_search(roi1, ventricle, septum, sizes, pixel_spacing_mm) -> RoiCandidate | None:
    ventricle_sum, septum_sum = _integral(ventricle), _integral(septum)
    # Not touching ROI1 beats size; touching is only the fallback when the
    # nearby ventricle has no room otherwise.
    for allow_touching in (False, True):
        for size in sizes:
            inside = _all_boxes(ventricle_sum, size)
            valid = (inside >= (1 - _VENTRICLE_TOLERANCE) * size * size) & (_all_boxes(septum_sum, size) == 0)
            valid &= _center_distance_mm(roi1, size, valid.shape, pixel_spacing_mm) <= _VENTRICLE_SEARCH_RADIUS_MM
            if not allow_touching:
                valid &= ~_touches(roi1, size, valid.shape)
            best = _nearest(valid, np.zeros(valid.shape), roi1, size, pixel_spacing_mm)
            if best is not None:
                return RoiCandidate(best[0], best[1], size, VENTRICLE)
    return None


def _white_matter_search(roi1, white, gray, forbidden, sizes, pixel_spacing_mm) -> RoiCandidate | None:
    white_sum, gray_sum, forbidden_sum = _integral(white), _integral(gray), _integral(forbidden)
    for size in sizes:
        area = size * size
        white_count = _all_boxes(white_sum, size)
        tissue_count = white_count + _all_boxes(gray_sum, size)
        valid = (tissue_count >= (1 - _WHITE_MATTER_TOLERANCE) * area) & ~_touches(roi1, size, white_count.shape)
        valid &= _all_boxes(forbidden_sum, size) == 0
        gray_fraction = 1.0 - white_count / area
        best = _nearest(valid, _GRAY_MATTER_PENALTY_MM * gray_fraction, roi1, size, pixel_spacing_mm)
        if best is not None:
            return RoiCandidate(best[0], best[1], size, WHITE_MATTER)
    return None


def _least_bad(roi1, tissue, brain, size, pixel_spacing_mm) -> RoiCandidate:
    """Guaranteed placement: the square near ROI1 with the fewest pixels
    outside white/gray matter (then outside the brain), not touching ROI1
    if at all possible.
    """
    area = size * size
    non_tissue = _all_boxes(_integral(~tissue), size) / area
    outside_brain = _all_boxes(_integral(~brain), size) / area
    penalty = 1000.0 * outside_brain + 100.0 * non_tissue
    valid = ~_touches(roi1, size, penalty.shape)
    if not valid.any():
        valid = np.ones(penalty.shape, dtype=bool)
    row0, col0 = _nearest(valid, penalty, roi1, size, pixel_spacing_mm)
    return RoiCandidate(row0, col0, size, FALLBACK)


def _all_boxes(integral: np.ndarray, size: int) -> np.ndarray:
    """Pixel count of every size x size square, indexed by its top-left corner."""
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def _touches(roi1: RoiCandidate, size: int, shape: tuple[int, int]) -> np.ndarray:
    """True for every top-left position whose square overlaps or directly
    abuts ROI1 (no 1 px gap between them).
    """
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    rows_touch = (rows <= roi1.row1) & (rows + size >= roi1.row0)
    cols_touch = (cols <= roi1.col1) & (cols + size >= roi1.col0)
    return rows_touch & cols_touch


def _center_distance_mm(roi1: RoiCandidate, size: int, shape: tuple[int, int], pixel_spacing_mm: float) -> np.ndarray:
    """Center-to-center distance to ROI1 (mm) for every top-left position of a size x size square."""
    rows = np.arange(shape[0])[:, None] + size / 2
    cols = np.arange(shape[1])[None, :] + size / 2
    center_row, center_col = roi1.row0 + roi1.size / 2, roi1.col0 + roi1.size / 2
    return np.hypot(rows - center_row, cols - center_col) * pixel_spacing_mm


def _nearest(valid, extra_cost_mm, roi1, size, pixel_spacing_mm) -> tuple[int, int] | None:
    """Valid top-left position minimizing center distance to ROI1 (mm) + `extra_cost_mm`."""
    if not valid.any():
        return None
    distance_mm = _center_distance_mm(roi1, size, valid.shape, pixel_spacing_mm)
    cost = np.where(valid, distance_mm + extra_cost_mm, np.inf)
    row0, col0 = np.unravel_index(np.argmin(cost), cost.shape)
    return int(row0), int(col0)
