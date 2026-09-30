"""Automatic placement of ROI1 - the "structured" ROI that feeds MTF and PS
(`domain.results.study_quality.evaluate_slice_quality`) - on top of
`head_ct_unet`'s anatomy segmentation.

`architecture/layers_description.md` names falx and septum pellucidum as the
MTF/PS targets. The CNN has no standalone falx class - falx and tentorium
share "Dural Folds". Both are usable edge structures, but the falx is
preferred: a Dural Folds candidate only falls back to tentorium when no
falx candidate fits on that slice.

Placement per slice:
1. Several candidate centers sampled along each connected component of the
   target structure (never just one heuristic's top pick).
2. For each center, a square grows from `min_size` in 2 px steps - only as
   far as MTF actually needs (the edge plus `calc_mtf`'s +/-VEC_LEN_MM line
   profiles on both sides), not as large as possible, so ROI1 doesn't
   swallow the homogeneous area ROI2 will later have to be placed in next
   to it. Every size must stay inside the brain and contain at most
   `tolerance` pixels of labels that aren't part of the target's own
   neighborhood ("isolated from other anatomy").
3. Best center, in priority order: (Dural Folds only) is falx rather than
   tentorium; reached the MTF size target; the line is visible in the CT
   image across the whole square (given the image - the segmentation mask
   can run on where the real line has faded, and a line ending inside the
   square gives `calc_mtf`'s contour a tip whose "across the edge" profiles
   run along the line instead); holds one long straight line
   (`calc_mtf` measures across an edge - one clean edge spanning the square
   gives it the most usable edge points, junctions and fragments don't);
   larger size; most ROI2-friendly homogeneous space around it; closest to
   the component centroid.

Pure NumPy/SciPy - no Qt, no file I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from bictool.domain.metrics.mtf import VEC_LEN_MM
from bictool.domain.segmentation.segmentation_brain_mask import merge_segmentation_labels

SEPTUM_PELLUCIDUM = "Septum Pellucidum"
DURAL_FOLDS = "Dural Folds"
STRUCTURES = (SEPTUM_PELLUCIDUM, DURAL_FOLDS)

# Labels allowed inside an ROI1 square besides "outside nothing" - the
# structure itself plus the tissue it naturally borders. First-guess values,
# to be tuned against real data.
_ALLOWED_LABELS: dict[str, frozenset[str]] = {
    SEPTUM_PELLUCIDUM: frozenset({SEPTUM_PELLUCIDUM, "Ventricular System"}),
    DURAL_FOLDS: frozenset(
        {
            DURAL_FOLDS,
            "Subarachnoid Space",
            "Brain Parenchyma",
            "Frontal Lobe",
            "Parietal Lobe",
            "Occipital Lobe",
            "Temporal Lobe",
            "Cerebellum",
        }
    ),
}

# Homogeneous labels ROI2 is likely to use (ventricles / white-matter-like
# parenchyma, per the architecture doc) - used only to rank ROI1 candidates
# by how much of that space they leave free next to themselves.
_ROI2_FRIENDLY_LABELS = frozenset({"Ventricular System", "Brain Parenchyma"})

DEFAULT_MIN_SIZE = 18
DEFAULT_MAX_SIZE = 48
DEFAULT_TOLERANCE = 0.02
_GROWTH_STEP = 2
_CENTERS_PER_COMPONENT = 12

# Falx vs. tentorium within "Dural Folds": the falx runs along the brain's
# midline (its principal axis on an axial slice); the tentorial edges run
# obliquely (e.g. the arms of the V where they meet the falx, ~20 deg off
# the midline on the real postop scan). A candidate counts as falx when its
# line lies within this distance of the midline and runs within this angle
# of it. No "away from the cerebellum" rule: the posterior falx stem runs
# right next to it.
_FALX_MAX_MIDLINE_DISTANCE_MM = 8.0
_FALX_MAX_ANGLE_DEG = 15.0
# Midline direction estimate (see `estimate_midline_direction`): standard
# axial DICOM has image rows running anterior->posterior, so the midline is
# roughly the image's vertical axis; falx lines within this angle of it
# refine that for head tilt. Only a slice whose Dural Folds pixels form one
# clearly line-like shape (major/minor spread at least this) votes - V/Y
# junction slices and blobs don't.
_MIDLINE_PRIOR = np.array([1.0, 0.0])
_MIDLINE_PRIOR_MAX_ANGLE_DEG = 30.0
_MIN_FALX_VOTE_ELONGATION = 4.0
_MIN_FALX_VOTE_PIXELS = 20

# Line scores within this of each other count as equal, so ROI2 room and
# centering can still break near-ties.
_LINE_SCORE_RESOLUTION = 0.05
# A candidate whose structure pixels don't form a usable line at all (a blob,
# scattered fragments) gives calc_mtf no real edge - rejected outright.
# Calibrated on the real postop scan: skull-base blobs/fragments scored
# 0.10-0.47, clean falx/septum/tentorium lines 0.64-0.86.
_MIN_LINE_SCORE = 0.5
# The falx is one straight line; a lower-scoring Dural Folds window (e.g. the
# V where the tentorium meets the falx) is treated as tentorium instead.
_FALX_MIN_LINE_SCORE = 0.6
# Two pixels in the same cross-section more than this far apart belong to
# separate runs (i.e. separate lines).
_CROSS_SECTION_GAP_PX = 1.5
# Line visibility in the image (see `_visible_line_fraction`): a cross-section
# of the line counts as visible when its brightest structure pixel exceeds the
# median of the flanking tissue by this much (lightly smoothed HU). Calibrated
# on the real postop scan: clearly visible falx/septum cross-sections >= ~5 HU,
# the faded falx stretch that broke calc_mtf on slice 28 -10..2 HU.
_MIN_LINE_CONTRAST_HU = 5.0
_LINE_FLANK_WIDTH_PX = 6.0
_VISIBILITY_SMOOTHING_SIGMA_PX = 1.0
_VISIBILITY_RESOLUTION = 0.1


@dataclass(frozen=True)
class RoiCandidate:
    """A square ROI in image-pixel coordinates (`[row0:row0+size, col0:col0+size]`).

    The structure runs through its center; `calc_mtf`'s seed is placed from
    the cropped image at QA time (`domain.metrics.mtf_seed`).
    """

    row0: int
    col0: int
    size: int
    structure: str

    @property
    def row1(self) -> int:
        return self.row0 + self.size

    @property
    def col1(self) -> int:
        return self.col0 + self.size


def mtf_target_size(pixel_spacing_mm: float, min_size: int = DEFAULT_MIN_SIZE) -> int:
    """Smallest even-step size from `min_size` that fits `calc_mtf`'s
    +/-VEC_LEN_MM line profiles on both sides of an edge at the center.
    """
    needed = math.ceil(2 * VEC_LEN_MM / pixel_spacing_mm) + 2
    size = min_size
    while size < needed:
        size += _GROWTH_STEP
    return size


def find_roi1_candidate(
    masks: dict[str, np.ndarray],
    structure: str,
    pixel_spacing_mm: float,
    *,
    midline_direction: np.ndarray | None = None,
    image: np.ndarray | None = None,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
) -> RoiCandidate | None:
    """Best ROI1 square for `structure` on one slice, or None if none fits.

    `image` (the slice's HU, same shape as the masks) - optional; when given,
    candidates whose line is visible in the image across the whole square
    win (see the module docstring). Without it, the masks alone decide.

    `midline_direction` (unit (row, col)) - the head's midline direction,
    ideally estimated over the whole volume (`estimate_midline_direction`,
    which `find_roi1_candidates` does). Defaults to an estimate from this
    slice alone.
    """
    if structure not in _ALLOWED_LABELS:
        raise ValueError(f"Unknown ROI1 structure {structure!r} - expected one of {STRUCTURES}")
    target = masks.get(structure)
    if target is None or not target.any():
        return None

    brain = merge_segmentation_labels(masks)
    disallowed = ~brain
    for name, mask in masks.items():
        if name not in _ALLOWED_LABELS[structure]:
            disallowed = disallowed | mask
    friendly = np.zeros_like(brain)
    for name in _ROI2_FRIENDLY_LABELS:
        if name in masks:
            friendly = friendly | masks[name]

    # Integral images make every square's pixel count O(1).
    disallowed_sum = _integral(disallowed)
    friendly_sum = _integral(friendly)
    target_size = min(max(mtf_target_size(pixel_spacing_mm, min_size), min_size), max_size)
    height, width = target.shape
    if midline_direction is None:
        midline_direction = estimate_midline_direction({0: masks})
    brain_center, _, _ = _principal_axis(*np.nonzero(brain))
    midline = (brain_center, midline_direction)
    smoothed = None
    if image is not None:
        smoothed = ndimage.gaussian_filter(np.asarray(image, dtype=np.float64), _VISIBILITY_SMOOTHING_SIGMA_PX)

    best: tuple | None = None
    labeled, num = ndimage.label(target)
    for component_label in range(1, num + 1):
        rows, cols = np.nonzero(labeled == component_label)
        centroid = (rows.mean(), cols.mean())
        for row, col in _sample_centers(rows, cols):
            size = _grow(row, col, disallowed_sum, height, width, min_size, target_size, tolerance)
            if size is None:
                continue
            row0, col0 = row - size // 2, col - size // 2
            window = target[row0 : row0 + size, col0 : col0 + size]
            line_score, line_direction = _line_score(window)
            if line_score < _MIN_LINE_SCORE:
                continue
            is_falx = structure != DURAL_FOLDS or (
                line_score >= _FALX_MIN_LINE_SCORE
                and _looks_like_falx(row, col, line_direction, midline, pixel_spacing_mm)
            )
            visible = 1.0
            if smoothed is not None:
                visible = _visible_line_fraction(window, smoothed[row0 : row0 + size, col0 : col0 + size], line_direction)
            free = _ring_count(friendly_sum, row0, col0, size, height, width)
            distance = math.hypot(row - centroid[0], col - centroid[1])
            key = (
                is_falx,
                size >= target_size,
                round(visible / _VISIBILITY_RESOLUTION),
                round(line_score / _LINE_SCORE_RESOLUTION),
                size,
                free,
                -distance,
            )
            if best is None or key > best[0]:
                best = (key, RoiCandidate(int(row0), int(col0), int(size), structure))
    return None if best is None else best[1]


def find_roi1_candidates(
    masks_by_slice: dict[int, dict[str, np.ndarray]],
    structure: str,
    pixel_spacing_mm: float,
    *,
    images_by_slice: Mapping[int, np.ndarray] | None = None,
    **kwargs,
) -> dict[int, RoiCandidate]:
    """`find_roi1_candidate` for every slice (sharing one volume-wide
    midline direction); slices without a candidate are omitted.
    `images_by_slice` optionally gives each slice's HU image.
    """
    kwargs.setdefault("midline_direction", estimate_midline_direction(masks_by_slice))
    candidates = {}
    for slice_index, masks in masks_by_slice.items():
        image = None if images_by_slice is None else images_by_slice.get(slice_index)
        candidate = find_roi1_candidate(masks, structure, pixel_spacing_mm, image=image, **kwargs)
        if candidate is not None:
            candidates[slice_index] = candidate
    return candidates


def estimate_midline_direction(masks_by_slice: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    """The head's midline direction (unit (row, col)) for a whole volume.

    Estimated from the falx itself, not the brain outline: on the real
    postop scan the outline's long axis ran ~90 deg off the midline at the
    skull base (temporal lobes/cerebellum make it wider left-right) and
    drifted up to ~50 deg at the vertex (craniotomy asymmetry), while the
    falx sat within ~10 deg of the image vertical throughout. So: every
    slice whose Dural Folds pixels form one clearly line-like shape within
    `_MIDLINE_PRIOR_MAX_ANGLE_DEG` of the image vertical votes with that
    line's direction, weighted by its pixel count. Falls back to the image
    vertical (standard axial orientation) if no slice qualifies.
    """
    total = np.zeros(2)
    min_cos = math.cos(math.radians(_MIDLINE_PRIOR_MAX_ANGLE_DEG))
    for masks in masks_by_slice.values():
        dural = masks.get(DURAL_FOLDS)
        if dural is None:
            continue
        rows, cols = np.nonzero(dural)
        if len(rows) < _MIN_FALX_VOTE_PIXELS:
            continue
        _, axis, elongation = _principal_axis(rows, cols)
        if elongation < _MIN_FALX_VOTE_ELONGATION:
            continue
        if np.dot(axis, _MIDLINE_PRIOR) < 0:
            axis = -axis
        if np.dot(axis, _MIDLINE_PRIOR) < min_cos:
            continue
        total += len(rows) * axis
    norm = np.linalg.norm(total)
    return _MIDLINE_PRIOR.copy() if norm == 0 else total / norm


def _principal_axis(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(centroid, unit direction, elongation = major/minor spread) of a pixel set, in (row, col)."""
    points = np.column_stack((rows, cols)).astype(np.float64)
    center = points.mean(axis=0)
    if len(points) < 2:
        return center, np.array([1.0, 0.0]), 1.0
    _, singular_values, vt = np.linalg.svd(points - center, full_matrices=False)
    minor = singular_values[1] if len(singular_values) > 1 else 0.0
    elongation = singular_values[0] / minor if minor > 0 else float("inf")
    return center, vt[0], elongation


def _line_score(window: np.ndarray) -> tuple[float, np.ndarray | None]:
    """How well the structure pixels inside an ROI square form one long
    straight line, in [0, 1]: straightness (1 - minor/major PCA spread)
    x single-line fraction (cross-sections perpendicular to the line that
    cross exactly one run of pixels - a V or Y junction crosses two)
    x span across the square (capped at 1) / number of separate pieces.
    0 for no pixels. Also returns the line's direction (row, col), for the
    falx check.
    """
    rows, cols = np.nonzero(window)
    if len(rows) < 3:
        return 0.0, None
    _, pieces = ndimage.label(window)
    points = np.column_stack((rows, cols)).astype(np.float64)
    centered = points - points.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    major = singular_values[0]
    minor = singular_values[1] if len(singular_values) > 1 else 0.0
    straightness = 1.0 - minor / major if major > 0 else 0.0
    along = centered @ vt[0]
    across = centered @ np.array([-vt[0][1], vt[0][0]])
    span = min((along.max() - along.min() + 1) / window.shape[0], 1.0)
    return straightness * _single_line_fraction(along, across) * span / pieces, vt[0]


def _single_line_fraction(along: np.ndarray, across: np.ndarray) -> float:
    """Fraction of 1 px cross-sections (binned along the line) whose pixels
    form a single contiguous run across it.
    """
    bins = np.round(along).astype(int)
    single = 0
    positions = np.unique(bins)
    for position in positions:
        offsets = np.sort(across[bins == position])
        if not np.any(np.diff(offsets) > _CROSS_SECTION_GAP_PX):
            single += 1
    return single / len(positions)


def _visible_line_fraction(window: np.ndarray, smoothed_window: np.ndarray, direction: np.ndarray) -> float:
    """Fraction of the line's 1 px cross-sections, over its whole extent
    through the square, where the line stands out in the image: brightest
    structure pixel at least `_MIN_LINE_CONTRAST_HU` above the median of the
    non-structure pixels within `_LINE_FLANK_WIDTH_PX` beside it. A cross-
    section without structure pixels counts as not visible.
    """
    rows, cols = np.nonzero(window)
    center = np.array([rows.mean(), cols.mean()])
    normal = np.array([-direction[1], direction[0]])
    grid = np.indices(window.shape).reshape(2, -1).T - center
    along = np.round(grid @ direction).astype(int)
    across = grid @ normal
    on_structure = window.ravel()
    values = smoothed_window.ravel()
    extent = np.unique(along[np.abs(across) <= 1.0])  # where the line runs through the square
    visible = 0
    for position in extent:
        in_section = along == position
        line = in_section & on_structure
        flank = in_section & ~on_structure & (np.abs(across) <= _LINE_FLANK_WIDTH_PX)
        if line.any() and flank.any():
            if values[line].max() - np.median(values[flank]) >= _MIN_LINE_CONTRAST_HU:
                visible += 1
    return visible / len(extent) if len(extent) else 0.0


def _looks_like_falx(row, col, line_direction, midline, pixel_spacing_mm) -> bool:
    if line_direction is None:
        return False
    center, axis = midline
    offset = np.array([row, col], dtype=np.float64) - center
    distance_mm = abs(offset[0] * axis[1] - offset[1] * axis[0]) * pixel_spacing_mm
    cos_angle = abs(float(np.dot(line_direction, axis)))
    return distance_mm <= _FALX_MAX_MIDLINE_DISTANCE_MM and cos_angle >= math.cos(math.radians(_FALX_MAX_ANGLE_DEG))


def _sample_centers(rows: np.ndarray, cols: np.ndarray) -> list[tuple[int, int]]:
    """Up to `_CENTERS_PER_COMPONENT` component pixels spread evenly along
    the component's principal axis (thin line-like structures), plus the
    pixel nearest the centroid.
    """
    points = np.column_stack((rows, cols)).astype(np.float64)
    center = points.mean(axis=0)
    if len(points) > 1:
        _, _, vt = np.linalg.svd(points - center, full_matrices=False)
        projection = (points - center) @ vt[0]
    else:
        projection = np.zeros(1)
    order = np.argsort(projection)
    picks = np.unique(np.linspace(0, len(order) - 1, min(_CENTERS_PER_COMPONENT, len(order))).round().astype(int))
    chosen = {(int(rows[order[i]]), int(cols[order[i]])) for i in picks}
    nearest = int(np.argmin(((points - center) ** 2).sum(axis=1)))
    chosen.add((int(rows[nearest]), int(cols[nearest])))
    return sorted(chosen)


def _grow(row, col, disallowed_sum, height, width, min_size, target_size, tolerance) -> int | None:
    """Largest size (min_size..target_size, even steps) whose square around
    (row, col) fits the image and stays isolated; None if min_size fails.
    """
    accepted = None
    size = min_size
    while size <= target_size:
        row0, col0 = row - size // 2, col - size // 2
        if row0 < 0 or col0 < 0 or row0 + size > height or col0 + size > width:
            break
        bad = _box_sum(disallowed_sum, row0, col0, size, size)
        if bad > tolerance * size * size:
            break
        accepted = size
        size += _GROWTH_STEP
    return accepted


def _ring_count(integral, row0, col0, size, height, width) -> int:
    """Pixels counted by `integral` in a band one ROI-width wide around the square, excluding it."""
    r0, c0 = max(row0 - size, 0), max(col0 - size, 0)
    r1, c1 = min(row0 + 2 * size, height), min(col0 + 2 * size, width)
    outer = _box_sum(integral, r0, c0, r1 - r0, c1 - c0)
    return int(outer - _box_sum(integral, row0, col0, size, size))


def _integral(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    out[1:, 1:] = mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return out


def _box_sum(integral: np.ndarray, row0: int, col0: int, h: int, w: int) -> int:
    r1, c1 = row0 + h, col0 + w
    return int(integral[r1, c1] - integral[row0, c1] - integral[r1, col0] + integral[row0, col0])
