"""Tests `classify_brain_tissue_fcm` against synthetic, hand-built slices -
never real patient data, per this project's test-data policy.
"""

import numpy as np

from bictool.domain.segmentation.brain_tissue_fcm import classify_brain_tissue_fcm


def _three_class_slice(size: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """A brain-mask-filled square containing three large, well-separated
    HU regions - CSF-like (~10), white-matter-like (~30), gray-matter-like
    (~40) - so a correct 3-cluster fit has an unambiguous answer.
    """
    slice_hu = np.full((size, size), 10.0, dtype=np.float64)
    slice_hu[0:20, :] = 30.0
    slice_hu[40:60, :] = 40.0
    brain_mask = np.ones((size, size), dtype=bool)
    return slice_hu, brain_mask


def test_classify_brain_tissue_fcm_separates_three_real_clusters():
    slice_hu, brain_mask = _three_class_slice()

    result = classify_brain_tissue_fcm(slice_hu, brain_mask)

    assert result.masks["White Matter"][5:15, 5:15].all()
    assert not result.masks["Gray Matter"][5:15, 5:15].any()

    assert result.masks["Gray Matter"][45:55, 5:15].all()
    assert not result.masks["White Matter"][45:55, 5:15].any()

    # The CSF-like middle band should be neither.
    assert not result.masks["White Matter"][25:35, 5:15].any()
    assert not result.masks["Gray Matter"][25:35, 5:15].any()


def test_classify_brain_tissue_fcm_never_labels_outside_the_brain_mask():
    slice_hu = np.full((20, 20), 40.0, dtype=np.float64)
    brain_mask = np.zeros((20, 20), dtype=bool)

    result = classify_brain_tissue_fcm(slice_hu, brain_mask)

    assert not result.masks["White Matter"].any()
    assert not result.masks["Gray Matter"].any()


def test_classify_brain_tissue_fcm_masks_are_boolean_and_mutually_exclusive():
    slice_hu, brain_mask = _three_class_slice()

    result = classify_brain_tissue_fcm(slice_hu, brain_mask)

    white_matter, gray_matter = result.masks["White Matter"], result.masks["Gray Matter"]
    assert white_matter.dtype == bool
    assert gray_matter.dtype == bool
    assert white_matter.shape == slice_hu.shape
    assert not (white_matter & gray_matter).any()


def test_classify_brain_tissue_fcm_handles_a_too_small_brain_mask():
    slice_hu = np.full((5, 5), 40.0, dtype=np.float64)
    brain_mask = np.zeros((5, 5), dtype=bool)
    brain_mask[0, 0] = True  # fewer in-mask pixels than clusters

    result = classify_brain_tissue_fcm(slice_hu, brain_mask, n_clusters=3)

    assert not result.masks["White Matter"].any()
    assert not result.masks["Gray Matter"].any()
