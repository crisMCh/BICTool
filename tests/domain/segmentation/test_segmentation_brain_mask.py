"""Tests `segmentation_brain_mask.py` against synthetic, hand-built masks -
never real patient data, per this project's test-data policy.
"""

import numpy as np

from bictool.domain.segmentation.segmentation_brain_mask import (
    brain_mask_from_segmentation,
    largest_connected_volume,
    merge_segmentation_labels,
)


def test_merge_segmentation_labels_unions_every_structure():
    masks = {
        "Brain Parenchyma": np.array([[True, False], [False, False]]),
        "Cerebellum": np.array([[False, True], [False, False]]),
        "Ventricular System": np.array([[False, False], [False, False]]),
    }

    merged = merge_segmentation_labels(masks)

    np.testing.assert_array_equal(merged, [[True, True], [False, False]])


def test_merge_segmentation_labels_returns_boolean_array():
    masks = {"Brain Parenchyma": np.ones((5, 5), dtype=bool)}

    merged = merge_segmentation_labels(masks)

    assert merged.dtype == bool


def test_largest_connected_volume_keeps_only_the_biggest_3d_component():
    volume = np.zeros((5, 10, 10), dtype=bool)
    volume[1:4, 2:8, 2:8] = True  # a large blob spanning slices 1-3
    volume[0, 0:2, 0:2] = True  # a small, spatially separate blob on slice 0

    result = largest_connected_volume(volume)

    assert result[2, 5, 5]  # inside the large blob
    assert not result[0, 0, 0]  # the small separate blob is dropped


def test_largest_connected_volume_keeps_an_object_connected_only_through_z():
    # Two same-sized fragments that don't touch within any single 2D slice,
    # but are adjacent through the Z axis between consecutive slices - must
    # be treated as one connected 3D object, not two.
    volume = np.zeros((3, 10, 10), dtype=bool)
    volume[0, 2:5, 2:5] = True
    volume[1, 2:5, 2:5] = True  # same footprint - connects slice 0 to slice 2 through slice 1
    volume[2, 2:5, 2:5] = True

    result = largest_connected_volume(volume)

    labeled_check = result[0, 3, 3] and result[1, 3, 3] and result[2, 3, 3]
    assert labeled_check


def test_largest_connected_volume_empty_input_returns_all_false():
    volume = np.zeros((3, 5, 5), dtype=bool)

    result = largest_connected_volume(volume)

    assert not result.any()
    assert result.shape == volume.shape


def test_brain_mask_from_segmentation_drops_a_spurious_fragment_on_a_different_slice():
    # Regression test for the real scenario reported 2026-09-29: on an
    # irregular (postop) scan, the CNN segmentation can mislabel a small,
    # spatially-implausible patch on one slice (e.g. face/scalp) as a real
    # structure - the 3D volume check must drop it, since it isn't part of
    # the same real 3D brain object as the main mass on neighboring slices.
    shape = (10, 10)
    masks_by_slice = {
        4: {"Brain Parenchyma": np.zeros(shape, dtype=bool)},
        5: {"Brain Parenchyma": _square(shape, 2, 8)},
        6: {"Brain Parenchyma": _square(shape, 2, 8)},
        7: {"Brain Parenchyma": _square(shape, 2, 8)},
    }
    # A spurious, spatially-separate small patch on slice 4 (no overlap with
    # the real brain footprint on slices 5-7, and slice 4 itself has nothing
    # else) - must not survive.
    masks_by_slice[4]["Brain Parenchyma"][0:2, 0:2] = True

    result = brain_mask_from_segmentation(masks_by_slice, total_slices=10)

    assert not result[4].any()  # the spurious fragment is gone
    assert result[6][5, 5]  # the real brain mass survives


def test_brain_mask_from_segmentation_only_returns_slices_that_had_input():
    shape = (5, 5)
    masks_by_slice = {3: {"Brain Parenchyma": np.ones(shape, dtype=bool)}}

    result = brain_mask_from_segmentation(masks_by_slice, total_slices=10)

    assert set(result.keys()) == {3}


def test_brain_mask_from_segmentation_respects_real_slice_positions_for_3d_adjacency():
    # Two segmented slices far apart in the real volume (indices 0 and 9),
    # with different sizes - if they were wrongly compacted into adjacent Z
    # positions (ignoring the 8 real, empty slices between them) instead of
    # placed at their true indices, `largest_connected_volume` would merge
    # them into one artificially-connected component. Placed correctly, they
    # stay two separate components and only the genuinely larger one (index
    # 0) survives - the smaller one (index 9) is correctly dropped, exactly
    # as it would be if they really were unrelated, non-adjacent objects.
    shape = (5, 5)
    masks_by_slice = {
        0: {"Brain Parenchyma": np.ones(shape, dtype=bool)},
        9: {"Brain Parenchyma": _square(shape, 0, 2)},
    }

    result = brain_mask_from_segmentation(masks_by_slice, total_slices=10)

    assert result[0].all()
    assert not result[9].any()


def test_brain_mask_from_segmentation_empty_input_returns_empty_dict():
    result = brain_mask_from_segmentation({}, total_slices=10)

    assert result == {}


def _square(shape: tuple[int, int], low: int, high: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[low:high, low:high] = True
    return mask
