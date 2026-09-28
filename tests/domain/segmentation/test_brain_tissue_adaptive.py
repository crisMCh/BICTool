"""Tests `classify_brain_tissue_adaptive` against synthetic, hand-built
slices - never real patient data, per this project's test-data policy.
"""

import numpy as np

from bictool.domain.segmentation.brain_tissue_adaptive import classify_brain_tissue_adaptive


def test_classify_brain_tissue_adaptive_tracks_a_slowly_varying_baseline():
    # A smoothly increasing baseline HU across the slice (simulating beam-
    # hardening/detector inhomogeneity), with a fine checkerboard +12 HU
    # "gray matter" bump interleaved with the (unbumped) "white matter"
    # baseline everywhere - every local window contains both, so a single
    # GLOBAL threshold can't correctly separate this (the low-baseline
    # region's gray matter reads lower than the high-baseline region's
    # white matter), but classifying each pixel relative to its own local
    # neighborhood mean should still split gray from white correctly at
    # every position, regardless of where it sits on the slow drift - the
    # actual point of local/adaptive thresholding over a global cutoff.
    size = 120
    baseline = np.linspace(20.0, 35.0, size)[None, :].repeat(size, axis=0)
    checker = (np.add.outer(np.arange(size), np.arange(size)) // 10) % 2  # 10px checkerboard cells
    slice_hu = baseline + checker * 12.0
    brain_mask = np.ones((size, size), dtype=bool)

    result = classify_brain_tissue_adaptive(slice_hu, brain_mask, block_size=41, min_component_size=0)

    low_cells = checker == 0
    high_cells = checker == 1
    assert result.masks["White Matter"][low_cells].mean() > 0.8
    assert result.masks["Gray Matter"][high_cells].mean() > 0.8


def test_classify_brain_tissue_adaptive_never_labels_outside_the_brain_mask():
    slice_hu = np.full((20, 20), 40.0, dtype=np.float64)
    brain_mask = np.zeros((20, 20), dtype=bool)

    result = classify_brain_tissue_adaptive(slice_hu, brain_mask)

    assert not result.masks["White Matter"].any()
    assert not result.masks["Gray Matter"].any()


def test_classify_brain_tissue_adaptive_masks_are_boolean_and_mutually_exclusive():
    size = 40
    slice_hu = np.full((size, size), 30.0, dtype=np.float64)
    brain_mask = np.ones((size, size), dtype=bool)

    result = classify_brain_tissue_adaptive(slice_hu, brain_mask)

    white_matter, gray_matter = result.masks["White Matter"], result.masks["Gray Matter"]
    assert white_matter.dtype == bool
    assert gray_matter.dtype == bool
    assert white_matter.shape == slice_hu.shape
    assert not (white_matter & gray_matter).any()
