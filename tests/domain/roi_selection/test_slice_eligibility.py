"""Tests `slice_eligibility.py` on synthetic label maps - no patient data."""

import numpy as np

from bictool.domain.roi_selection.slice_eligibility import (
    MIN_BRAIN_FRACTION,
    brain_area_fractions,
    eligible_slices,
)

_SHAPE = (100, 100)


def _slice(brain_pixels: int) -> dict[str, np.ndarray]:
    """Brain split over two labels (the union counts), `brain_pixels` in total."""
    flat = np.zeros(_SHAPE[0] * _SHAPE[1], dtype=bool)
    flat[:brain_pixels] = True
    parenchyma = flat.reshape(_SHAPE)
    falx = np.zeros(_SHAPE, dtype=bool)
    falx[:, 0] = parenchyma[:, 0]
    return {"Brain Parenchyma": parenchyma & ~falx, "Dural Folds": falx}


def test_fractions_are_relative_to_the_largest_slice():
    fractions = brain_area_fractions({0: _slice(500), 1: _slice(5000), 2: _slice(1000)})

    assert fractions == {0: 0.1, 1: 1.0, 2: 0.2}


def test_slices_below_the_minimum_fraction_are_not_eligible():
    masks = {0: _slice(500), 1: _slice(5000), 2: _slice(1000), 3: _slice(999)}

    assert MIN_BRAIN_FRACTION == 0.2
    assert eligible_slices(masks) == {1, 2}  # exactly 20 % is eligible
    assert eligible_slices(masks, min_fraction=0.05) == {0, 1, 2, 3}


def test_slices_without_brain_are_never_eligible():
    empty = {name: np.zeros(_SHAPE, dtype=bool) for name in ("Brain Parenchyma", "Dural Folds")}

    assert eligible_slices({0: empty, 1: _slice(3000)}) == {1}
    assert eligible_slices({0: empty}, min_fraction=0.0) == set()
    assert eligible_slices({}) == set()


def test_isolated_segmented_slices_work_without_their_neighbours():
    # e.g. only "Segment slice" on slices 3 and 30: not 3D-connected, both count.
    assert eligible_slices({3: _slice(4000), 30: _slice(5000)}) == {3, 30}
