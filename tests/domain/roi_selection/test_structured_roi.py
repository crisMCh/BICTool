"""Tests `structured_roi.py` against synthetic label maps - never real
patient data, per this project's test-data policy (plus one real-data test,
gated on the scan being present, per the project's established pattern).
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from bictool.domain.roi_selection.structured_roi import (
    DURAL_FOLDS,
    SEPTUM_PELLUCIDUM,
    _line_score,
    estimate_midline_direction,
    find_roi1_candidate,
    find_roi1_candidates,
    mtf_target_size,
)

_SHAPE = (120, 120)
_PIXEL_SPACING = 0.5  # -> MTF target size 22 px


def _septum_slice() -> dict[str, np.ndarray]:
    """Two ventricle blobs with a thin septum line between them, all inside
    a large parenchyma disk.
    """
    parenchyma = np.zeros(_SHAPE, dtype=bool)
    parenchyma[10:110, 10:110] = True
    ventricles = np.zeros(_SHAPE, dtype=bool)
    ventricles[30:90, 40:80] = True
    septum = np.zeros(_SHAPE, dtype=bool)
    septum[35:85, 59:61] = True
    ventricles &= ~septum
    parenchyma &= ~(ventricles | septum)
    return {"Brain Parenchyma": parenchyma, "Ventricular System": ventricles, SEPTUM_PELLUCIDUM: septum}


def test_mtf_target_size_covers_both_line_profiles():
    # +/-5 mm at 0.5 mm/px = 20 px + margin -> 22, grown in even steps from 18
    assert mtf_target_size(0.5) == 22
    assert mtf_target_size(5.0) == 18  # coarse spacing never goes below the minimum


def test_septum_candidate_is_centered_on_the_septum():
    candidate = find_roi1_candidate(_septum_slice(), SEPTUM_PELLUCIDUM, _PIXEL_SPACING)

    assert candidate is not None
    assert candidate.size == mtf_target_size(_PIXEL_SPACING)
    assert candidate.col0 <= 59 and candidate.col1 >= 61  # contains the septum line
    assert candidate.row0 >= 30 and candidate.row1 <= 90  # stays inside the ventricles


def test_growth_stops_at_the_mtf_target_instead_of_filling_the_ventricle():
    # The ventricle has room for a much bigger square - ROI1 must still stop
    # at the MTF target size, leaving the rest for ROI2.
    candidate = find_roi1_candidate(_septum_slice(), SEPTUM_PELLUCIDUM, _PIXEL_SPACING, max_size=48)

    assert candidate.size == mtf_target_size(_PIXEL_SPACING)


def test_growth_stops_when_a_disallowed_label_enters():
    masks = _septum_slice()
    caudate = np.zeros(_SHAPE, dtype=bool)
    caudate[30:90, 40:50] = True  # narrows the ventricle on the left
    caudate[30:90, 70:80] = True
    masks["Ventricular System"] &= ~caudate
    masks["Caudate Nucleus"] = caudate

    candidate = find_roi1_candidate(masks, SEPTUM_PELLUCIDUM, _PIXEL_SPACING)

    assert candidate is not None
    assert candidate.size < mtf_target_size(_PIXEL_SPACING)  # stopped early: 22 won't fit in cols 50-70
    assert candidate.col0 >= 50 and candidate.col1 <= 70


def test_returns_none_when_even_the_minimum_square_is_not_isolated():
    masks = _septum_slice()
    caudate = np.zeros(_SHAPE, dtype=bool)
    caudate[30:90, 40:55] = True
    caudate[30:90, 65:80] = True  # leaves only 10 px of ventricle around the septum
    masks["Ventricular System"] &= ~caudate
    masks["Caudate Nucleus"] = caudate

    assert find_roi1_candidate(masks, SEPTUM_PELLUCIDUM, _PIXEL_SPACING) is None


def test_returns_none_when_structure_absent():
    masks = _septum_slice()
    masks[SEPTUM_PELLUCIDUM][:] = False

    assert find_roi1_candidate(masks, SEPTUM_PELLUCIDUM, _PIXEL_SPACING) is None


def test_square_never_leaves_the_brain():
    masks = _septum_slice()
    # Ventricle and septum reach the top edge of the brain (row 10): centers
    # near that end would need a square extending outside the brain, which
    # must count as not isolated.
    masks["Ventricular System"][10:30, 40:80] = True
    masks[SEPTUM_PELLUCIDUM][10:35, 59:61] = True
    masks["Ventricular System"] &= ~masks[SEPTUM_PELLUCIDUM]
    masks["Brain Parenchyma"][10:30, 40:80] = False

    candidate = find_roi1_candidate(masks, SEPTUM_PELLUCIDUM, _PIXEL_SPACING)

    assert candidate is not None
    assert candidate.row0 >= 10


def test_dural_folds_next_to_cerebellum_is_accepted():
    # Tentorium bordering the cerebellum is a valid ROI1 target too.
    cerebellum = np.zeros(_SHAPE, dtype=bool)
    cerebellum[60:110, 10:110] = True
    dural = np.zeros(_SHAPE, dtype=bool)
    dural[58:60, 20:100] = True
    lobes = np.zeros(_SHAPE, dtype=bool)
    lobes[10:58, 10:110] = True
    masks = {"Cerebellum": cerebellum, DURAL_FOLDS: dural, "Occipital Lobe": lobes}

    candidate = find_roi1_candidate(masks, DURAL_FOLDS, _PIXEL_SPACING)

    assert candidate is not None
    assert candidate.row0 <= 58 < candidate.row1


def _dural_slice(*, falx: bool = True, tentorium: bool = True) -> dict[str, np.ndarray]:
    """A parenchyma box with a vertical midline falx line (col 60, rows
    20-100) and/or an oblique tentorium-like line in the lower left.
    """
    lobes = np.zeros(_SHAPE, dtype=bool)
    lobes[10:110, 10:110] = True
    dural = np.zeros(_SHAPE, dtype=bool)
    if falx:
        dural[20:100, 59:61] = True
    if tentorium:
        for i in range(40):  # ~45 deg line, well away from the falx
            dural[60 + i, 15 + i : 17 + i] = True
    lobes &= ~dural
    return {"Frontal Lobe": lobes, DURAL_FOLDS: dural}


def test_dural_folds_prefers_the_falx_over_the_tentorium():
    candidate = find_roi1_candidate(_dural_slice(), DURAL_FOLDS, _PIXEL_SPACING)

    assert candidate.col0 <= 59 and candidate.col1 >= 61  # on the vertical midline line


def test_dural_folds_falls_back_to_the_tentorium_without_a_falx():
    candidate = find_roi1_candidate(_dural_slice(falx=False), DURAL_FOLDS, _PIXEL_SPACING)

    assert candidate is not None
    assert candidate.col1 < 59  # on the oblique line, left of where the falx would be


def test_line_score_prefers_one_straight_line_over_a_v_junction():
    straight = np.zeros((26, 26), dtype=bool)
    straight[:, 12:14] = True
    v_shape = np.zeros((26, 26), dtype=bool)
    for row in range(26):  # two arms diverging from the bottom center
        v_shape[row, 12 - row // 2 : 14 - row // 2] = True
        v_shape[row, 12 + row // 2 : 14 + row // 2] = True

    assert _line_score(straight)[0] > 0.9
    assert _line_score(v_shape)[0] < 0.6


def test_a_blob_without_a_line_is_rejected():
    masks = _septum_slice()
    masks[SEPTUM_PELLUCIDUM][:] = False
    masks[SEPTUM_PELLUCIDUM][55:65, 55:65] = True  # a square blob, no edge line
    masks["Ventricular System"] &= ~masks[SEPTUM_PELLUCIDUM]

    assert find_roi1_candidate(masks, SEPTUM_PELLUCIDUM, _PIXEL_SPACING) is None


def test_midline_direction_comes_from_falx_lines_not_tentorium():
    tilted = np.zeros(_SHAPE, dtype=bool)
    for i in range(80):  # falx tilted ~10 deg from the image vertical
        tilted[20 + i, 60 + round(i * np.tan(np.radians(10)))] = True
    horizontal_tentorium = np.zeros(_SHAPE, dtype=bool)
    horizontal_tentorium[80, 10:110] = True  # 90 deg off - must not vote
    masks_by_slice = {0: {DURAL_FOLDS: tilted}, 1: {DURAL_FOLDS: horizontal_tentorium}}

    direction = estimate_midline_direction(masks_by_slice)

    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    assert abs(angle - 10) < 2


def test_midline_direction_defaults_to_image_vertical():
    direction = estimate_midline_direction({0: {DURAL_FOLDS: np.zeros(_SHAPE, dtype=bool)}})

    np.testing.assert_allclose(direction, [1.0, 0.0])


def test_unknown_structure_raises():
    with pytest.raises(ValueError):
        find_roi1_candidate(_septum_slice(), "Thalamus", _PIXEL_SPACING)


def test_volume_function_only_returns_slices_with_a_candidate():
    empty = _septum_slice()
    empty[SEPTUM_PELLUCIDUM][:] = False
    masks_by_slice = {3: _septum_slice(), 4: empty, 5: _septum_slice()}

    candidates = find_roi1_candidates(masks_by_slice, SEPTUM_PELLUCIDUM, _PIXEL_SPACING)

    assert set(candidates) == {3, 5}


# Real CNN segmentation of the postop scan under E:\Datasets, pickled as
# {slice_index: SegmentationResult.masks}. Lives in the gitignored
# `reference/` folder - derived from real patient data, never committed.
_REAL_SEG_CACHE = Path(__file__).resolve().parents[3] / "reference" / "segmentation" / "postop_seg_cache.pkl"


@pytest.mark.skipif(not _REAL_SEG_CACHE.is_file(), reason="needs cached real segmentation of the postop scan (not in repo)")
def test_real_postop_scan_finds_septum_candidates_around_the_ventricles():
    with open(_REAL_SEG_CACHE, "rb") as f:
        masks_by_slice = pickle.load(f)

    candidates = find_roi1_candidates(masks_by_slice, SEPTUM_PELLUCIDUM, 0.44041502)

    assert set(candidates) == set(range(19, 26))


def test_with_the_image_roi1_moves_to_where_the_line_is_actually_visible():
    # The septum mask spans rows 35-85, but in the image the line has faded
    # out below row 60 - an ROI there would give calc_mtf a contour tip.
    masks = _septum_slice()
    image = np.full(_SHAPE, 5.0)
    image[35:60, 59:61] = 30.0

    without_image = find_roi1_candidate(masks, SEPTUM_PELLUCIDUM, _PIXEL_SPACING)
    with_image = find_roi1_candidate(masks, SEPTUM_PELLUCIDUM, _PIXEL_SPACING, image=image)

    assert without_image.row1 > 61  # the mask alone can't tell
    assert with_image.row0 >= 34 and with_image.row1 <= 61
