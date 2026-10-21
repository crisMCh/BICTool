"""Where to seed `calc_mtf`'s active contour, and retrying other seeds when
one yields no usable edge.

`calc_mtf` grows a Chan-Vese region from a small seed box and measures the
MTF across that region's boundary. The seed must lie *inside* the structure
whose edges are measured (for a thin falx/septum line: on its centerline) -
a seed straddling the edge starts half bright/half dark and the contour
collapses; a seed in flat tissue next to the structure leaks or dies. Both
end in "No edge columns survived bump removal".

Seed coordinates: `calc_mtf` upsamples the ROI with `interp2_octave(., 2)`,
i.e. two refine passes = 4x, and centers its seed box on interpolated pixel
`4 * seed - 1`. So seed `(x, y)` lands at ORIGINAL (0-based) pixel
`(x - 0.25, y - 0.25)` - the seed value is simply the original column/row.
The Octave default `(10, 10)` is therefore the top-left quarter of a 40 px
ROI, not its center.

Pure NumPy/SciPy - no Qt, no file I/O.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from bictool.domain.metrics.mtf import IMAGE_INTERP_FACTOR, calc_mtf, calc_mtf_with_steps

# The seed box spans 4*seed-5 .. 4*seed+3 in the 4x-interpolated image, so
# seeds must stay this far inside the ROI.
_SEED_MARGIN = 2
# Candidates are taken from the ROI's central area (Chebyshev radius as a
# fraction of the ROI size) - the middle of the line, away from ROI borders
# where `calc_mtf`'s line profiles don't fit anyway.
_CENTRAL_RADIUS_FRACTION = 0.25
_SMOOTHING_SIGMA_PX = 1.0
DEFAULT_MAX_CANDIDATES = 5
_MIN_SEPARATION_PX = 3.0
# A seed's result only counts if at least this fraction of the active-contour
# region it grew lies on the structure (within 1 px; when the structure is known,
# i.e. the CNN falx/septum label). On thin lines the good seeds can all fail,
# and a fallback seed in the tissue beside the line grows a region there and
# measures the wrong edge. 2026-10-20, all 488 ROI1s of the 15 rated studies:
# wrong legacy-seed results 0.00-0.29, correct ones 0.69-1.00, regular seeds
# median 0.98 (13 of 464 below 0.30). docs/qa_method/physical_qa_recipe.md.
MIN_REGION_ON_STRUCTURE = 0.3
# The Octave-era fixed seed, relative to ROI size (10 of 40 px) - kept as a
# last resort so a ROI that worked before still can.
_LEGACY_SEED_FRACTION = 0.25


def mtf_seed_candidates(
    img_roi: np.ndarray,
    structure_mask: np.ndarray | None = None,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[tuple[int, int]]:
    """`calc_mtf` seeds `(x, y)` to try in order, best first.

    Candidates are the brightest (lightly smoothed) pixels in the ROI's
    central area, at least `_MIN_SEPARATION_PX` apart - the centerline of a
    bright falx/septum line, near the middle of its length. `structure_mask`
    (same shape as `img_roi`, e.g. the segmentation's structure crop)
    restricts them to the structure when it has pixels there; without it
    (manual ROIs) the image alone decides. The legacy quarter seed comes last.
    """
    img = np.asarray(img_roi, dtype=np.float64)
    n_rows, n_cols = img.shape
    smoothed = ndimage.gaussian_filter(img, _SMOOTHING_SIGMA_PX)

    rows, cols = np.mgrid[0:n_rows, 0:n_cols]
    valid = (
        (rows >= _SEED_MARGIN)
        & (rows <= n_rows - _SEED_MARGIN)
        & (cols >= _SEED_MARGIN)
        & (cols <= n_cols - _SEED_MARGIN)
    )
    radius = max(_CENTRAL_RADIUS_FRACTION * min(n_rows, n_cols), _MIN_SEPARATION_PX)
    central = (
        valid
        & (np.abs(rows - (n_rows - 1) / 2) <= radius)
        & (np.abs(cols - (n_cols - 1) / 2) <= radius)
    )
    allowed = central
    if structure_mask is not None:
        on_structure = central & np.asarray(structure_mask, dtype=bool)
        if on_structure.any():
            allowed = on_structure

    cand_rows, cand_cols = np.nonzero(allowed)
    order = np.argsort(-smoothed[cand_rows, cand_cols], kind="stable")
    seeds: list[tuple[int, int]] = []
    for k in order:
        if len(seeds) >= max_candidates:
            break
        _add_if_separated(seeds, int(cand_cols[k]), int(cand_rows[k]))

    legacy_x = min(max(int(_LEGACY_SEED_FRACTION * n_cols), _SEED_MARGIN), n_cols - _SEED_MARGIN)
    legacy_y = min(max(int(_LEGACY_SEED_FRACTION * n_rows), _SEED_MARGIN), n_rows - _SEED_MARGIN)
    _add_if_separated(seeds, legacy_x, legacy_y)
    return seeds


def _add_if_separated(seeds: list[tuple[int, int]], x: int, y: int) -> None:
    if all(np.hypot(x - sx, y - sy) >= _MIN_SEPARATION_PX for sx, sy in seeds):
        seeds.append((x, y))


def region_on_structure(segmentation: np.ndarray, structure_mask: np.ndarray, tolerance_px: int = 1) -> float:
    """Fraction of the active-contour region (on `calc_mtf`'s upsampled grid) that lies on `structure_mask`
    (original ROI grid), the structure widened by `tolerance_px` original pixels (the CNN label can sit a pixel
    beside the bright line). Compared on the upsampled grid: it has 4(n-1)+1 points and original pixel j covers
    points 4j-2 .. 4j+1, so thin regions along a line are not lost by subsampling."""
    factor = 2 ** IMAGE_INTERP_FACTOR
    region = np.asarray(segmentation, dtype=bool)
    structure = np.asarray(structure_mask, dtype=bool)
    if tolerance_px > 0:
        structure = ndimage.binary_dilation(structure, iterations=tolerance_px)
    up = np.kron(structure, np.ones((factor, factor), dtype=bool))[factor // 2:, factor // 2:]
    up = up[: region.shape[0], : region.shape[1]]
    region = region[: up.shape[0], : up.shape[1]]
    return float((region & up).sum() / max(int(region.sum()), 1))


def _knows_structure(structure_mask) -> bool:
    return structure_mask is not None and bool(np.asarray(structure_mask, dtype=bool).any())


def calc_mtf_first_viable_seed(
    img_roi, seeds: Sequence[tuple[int, int]], pix_spc, structure_mask=None, *, octave_step=False
):
    """`calc_mtf` with each seed in turn until one yields usable edges.

    With `structure_mask` (same shape as `img_roi`, the structure the edge belongs to), a seed whose
    active-contour region is not mostly on the structure (`MIN_REGION_ON_STRUCTURE`) is rejected too.
    Returns `(xaxis, MTF, MTFarea, seed)`. Raises the last seed's error if none works (and `ValueError`
    for an empty `seeds`).
    """
    xaxis, mtf, mtf_area, seed, _steps = calc_mtf_steps_first_viable_seed(
        img_roi, seeds, pix_spc, structure_mask, keep_steps=False, octave_step=octave_step
    )
    return xaxis, mtf, mtf_area, seed


def calc_mtf_steps_first_viable_seed(
    img_roi, seeds, pix_spc, structure_mask=None, *, keep_steps=True, octave_step=False
):
    """`calc_mtf_first_viable_seed` with the MTF's intermediate steps: `(xaxis, MTF, MTFarea, seed, MtfSteps)`
    (steps None when `keep_steps` is False and no structure check needs them)."""
    if not seeds:
        raise ValueError("No MTF seed to try")
    check = _knows_structure(structure_mask)
    last_error: ValueError | None = None
    for x, y in seeds:
        try:
            if keep_steps or check:
                xaxis, mtf, mtf_area, steps = calc_mtf_with_steps(img_roi, x, y, pix_spc, octave_step=octave_step)
            else:
                (xaxis, mtf, mtf_area), steps = calc_mtf(img_roi, x, y, pix_spc, octave_step=octave_step), None
        except ValueError as exc:
            last_error = exc
            continue
        if check and region_on_structure(steps.segmentation, structure_mask) < MIN_REGION_ON_STRUCTURE:
            last_error = ValueError("MTF edge not on the structure (no seed grew a region on the falx/septum)")
            continue
        return xaxis, mtf, mtf_area, (x, y), (steps if keep_steps else None)
    raise last_error
