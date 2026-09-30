"""Tests `homogeneous_roi.py` (ROI2 placement next to ROI1) against synthetic
label maps - never real patient data.
"""

import numpy as np

from bictool.domain.roi_selection.homogeneous_roi import (
    FALLBACK,
    VENTRICLE,
    WHITE_MATTER,
    find_roi2_candidate,
)
from bictool.domain.roi_selection.structured_roi import DURAL_FOLDS, SEPTUM_PELLUCIDUM, RoiCandidate

_SHAPE = (160, 160)
_PIXEL_SPACING = 0.5


def _touching(a: RoiCandidate, b: RoiCandidate) -> bool:
    return a.row0 <= b.row1 and b.row0 <= a.row1 and a.col0 <= b.col1 and b.col0 <= a.col1


def _overlap_count(candidate: RoiCandidate, mask: np.ndarray) -> int:
    return int(mask[candidate.row0 : candidate.row1, candidate.col0 : candidate.col1].sum())


def _septum_masks(ventricle_rows=(40, 120), ventricle_cols=(40, 120)) -> dict[str, np.ndarray]:
    ventricle = np.zeros(_SHAPE, dtype=bool)
    ventricle[slice(*ventricle_rows), slice(*ventricle_cols)] = True
    septum = np.zeros(_SHAPE, dtype=bool)
    septum[45:115, 79:81] = True
    ventricle &= ~septum
    parenchyma = np.zeros(_SHAPE, dtype=bool)
    parenchyma[10:150, 10:150] = True
    parenchyma &= ~(ventricle | septum)
    return {"Brain Parenchyma": parenchyma, VENTRICLE: ventricle, SEPTUM_PELLUCIDUM: septum}


def _tissue(white: np.ndarray, gray: np.ndarray | None = None) -> dict[str, np.ndarray]:
    return {WHITE_MATTER: white, "Gray Matter": np.zeros(_SHAPE, dtype=bool) if gray is None else gray}


_SEPTUM_ROI1 = RoiCandidate(row0=67, col0=67, size=26, structure=SEPTUM_PELLUCIDUM)


def test_septum_roi2_is_in_the_ventricle_next_to_roi1_without_touching_it():
    masks = _septum_masks()

    roi2 = find_roi2_candidate(_SEPTUM_ROI1, masks, _PIXEL_SPACING)

    assert roi2.structure == VENTRICLE
    assert roi2.size == _SEPTUM_ROI1.size
    assert not _touching(roi2, _SEPTUM_ROI1)
    assert _overlap_count(roi2, masks[SEPTUM_PELLUCIDUM]) == 0
    assert _overlap_count(roi2, masks[VENTRICLE]) == roi2.size**2
    # "close": right next to ROI1, not somewhere far away in the ventricle
    assert abs((roi2.col0 + roi2.size / 2) - (_SEPTUM_ROI1.col0 + 13)) <= 30


def test_septum_roi2_shrinks_before_touching_roi1():
    # Ventricle cols 57-103: a 26 px square beside ROI1 (cols 67-93) won't
    # fit without touching it, but a smaller one will above/below it.
    masks = _septum_masks(ventricle_rows=(40, 120), ventricle_cols=(58, 102))

    roi2 = find_roi2_candidate(_SEPTUM_ROI1, masks, _PIXEL_SPACING)

    assert roi2.structure == VENTRICLE
    assert not _touching(roi2, _SEPTUM_ROI1)
    assert _overlap_count(roi2, masks[SEPTUM_PELLUCIDUM]) == 0


def test_septum_roi2_touches_roi1_only_when_the_ventricle_has_no_other_room():
    # Ventricle only just around ROI1 - no non-touching square fits.
    masks = _septum_masks(ventricle_rows=(62, 98), ventricle_cols=(56, 104))
    masks[SEPTUM_PELLUCIDUM][:] = False
    masks[SEPTUM_PELLUCIDUM][62:98, 79:81] = True
    masks[VENTRICLE] &= ~masks[SEPTUM_PELLUCIDUM]

    roi2 = find_roi2_candidate(_SEPTUM_ROI1, masks, _PIXEL_SPACING)

    assert roi2.structure == VENTRICLE
    assert _touching(roi2, _SEPTUM_ROI1)
    assert _overlap_count(roi2, masks[SEPTUM_PELLUCIDUM]) == 0  # never the septum itself


def test_septum_falls_back_to_white_matter_when_the_ventricle_is_too_small():
    masks = _septum_masks(ventricle_rows=(70, 90), ventricle_cols=(72, 88))  # ~8 px either side of the septum
    white = masks["Brain Parenchyma"].copy()

    roi2 = find_roi2_candidate(_SEPTUM_ROI1, masks, _PIXEL_SPACING, _tissue(white))

    assert roi2.structure == WHITE_MATTER
    assert _overlap_count(roi2, masks[SEPTUM_PELLUCIDUM]) == 0


def test_septum_ventricle_fallback_waits_for_wm_gm_classification():
    masks = _septum_masks(ventricle_rows=(70, 90), ventricle_cols=(72, 88))

    assert find_roi2_candidate(_SEPTUM_ROI1, masks, _PIXEL_SPACING, tissue_masks=None) is None


def test_a_distant_part_of_the_ventricle_does_not_count_as_nearby():
    masks = _septum_masks(ventricle_rows=(70, 90), ventricle_cols=(72, 88))
    masks[VENTRICLE][10:40, 120:150] = True  # far away (~40 mm) roomy ventricle part
    white = masks["Brain Parenchyma"] & ~masks[VENTRICLE]

    roi2 = find_roi2_candidate(_SEPTUM_ROI1, masks, _PIXEL_SPACING, _tissue(white))

    assert roi2.structure == WHITE_MATTER


def _falx_masks() -> dict[str, np.ndarray]:
    dural = np.zeros(_SHAPE, dtype=bool)
    dural[20:140, 79:81] = True
    lobes = np.zeros(_SHAPE, dtype=bool)
    lobes[10:150, 10:150] = True
    lobes &= ~dural
    return {"Frontal Lobe": lobes, DURAL_FOLDS: dural}


_FALX_ROI1 = RoiCandidate(row0=67, col0=67, size=26, structure=DURAL_FOLDS)


def test_falx_roi2_is_pure_white_matter_next_to_roi1():
    masks = _falx_masks()
    white = masks["Frontal Lobe"].copy()

    roi2 = find_roi2_candidate(_FALX_ROI1, masks, _PIXEL_SPACING, _tissue(white))

    assert roi2.structure == WHITE_MATTER
    assert roi2.size == _FALX_ROI1.size
    assert not _touching(roi2, _FALX_ROI1)
    assert _overlap_count(roi2, masks[DURAL_FOLDS]) == 0


def test_falx_roi2_never_contains_the_falx_even_if_labelled_gray_matter():
    masks = _falx_masks()
    white = masks["Frontal Lobe"].copy()
    gray = masks[DURAL_FOLDS].copy()  # classifiers label the dense falx as gray matter

    roi2 = find_roi2_candidate(_FALX_ROI1, masks, _PIXEL_SPACING, _tissue(white, gray))

    assert _overlap_count(roi2, masks[DURAL_FOLDS]) == 0


def test_falx_roi2_accepts_some_gray_matter_rather_than_moving_far_away():
    masks = _falx_masks()
    gray = masks["Frontal Lobe"].copy()
    white = np.zeros(_SHAPE, dtype=bool)
    white[120:150, 120:150] = True  # the only pure white matter, far from ROI1
    gray &= ~white
    # Near ROI1: mostly white with a thin gray stripe (~8 % gray matter).
    white[40:120, 88:120] = True
    gray[40:120, 88:120] = False
    gray[40:120, 100:102] = True
    white[40:120, 100:102] = False

    roi2 = find_roi2_candidate(_FALX_ROI1, masks, _PIXEL_SPACING, _tissue(white, gray))

    assert roi2.structure == WHITE_MATTER
    assert roi2.row0 < 120 and roi2.col0 < 120  # stayed near ROI1 instead of the far pure-white patch


def test_falx_roi2_prefers_pure_white_matter_when_it_is_about_as_close():
    masks = _falx_masks()
    white = masks["Frontal Lobe"].copy()
    gray = np.zeros(_SHAPE, dtype=bool)
    gray[40:120, 95:150] = True  # right side of the falx is gray matter
    white &= ~gray

    roi2 = find_roi2_candidate(_FALX_ROI1, masks, _PIXEL_SPACING, _tissue(white, gray))

    assert _overlap_count(roi2, gray) == 0
    assert roi2.col1 <= 79  # took the pure-white left side


def test_roi2_always_exists_when_roi1_does():
    # No white/gray matter anywhere near - still a (flagged) placement.
    masks = _falx_masks()
    empty = np.zeros(_SHAPE, dtype=bool)

    roi2 = find_roi2_candidate(_FALX_ROI1, masks, _PIXEL_SPACING, _tissue(empty, empty))

    assert roi2 is not None
    assert roi2.structure == FALLBACK
    assert not _touching(roi2, _FALX_ROI1)
