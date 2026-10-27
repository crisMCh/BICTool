"""Tests `domain.metrics.basal_ganglia` with synthetic slices and labels."""

import numpy as np

from bictool.domain.metrics.basal_ganglia import (
    CLEARLY,
    HARDLY,
    NOT_PLAUSIBLE,
    NOT_VISIBLE,
    basal_ganglia_block,
    evaluate_basal_ganglia,
    measure_slice,
)


def _labels(shape=(120, 120)):
    lentiform = np.zeros(shape, bool)
    lentiform[40:80, 30:50] = True
    capsule = np.zeros(shape, bool)
    capsule[40:80, 50:62] = True
    return {"Lentiform Nucleus": lentiform, "Internal Capsule": capsule}


def _slice(labels, nucleus_hu, noise=2.0, seed=0, blur=0.0):
    from scipy import ndimage as ndi

    img = np.full(labels["Internal Capsule"].shape, 30.0)
    img[labels["Lentiform Nucleus"]] = nucleus_hu
    if blur:
        img = ndi.gaussian_filter(img, blur)
    return img + np.random.default_rng(seed).normal(0, noise, img.shape)


def test_a_sharp_nucleus_7_hu_above_the_capsule_is_clearly_visible():
    labels = _labels()
    result = measure_slice(_slice(labels, 37.0), labels)
    assert result.plausible and 6.0 < result.differences["Lentiform Nucleus"] < 8.0
    assert result.level == CLEARLY and result.visible


def test_noise_and_blur_lower_the_visibility():
    labels = _labels()
    sharp = measure_slice(_slice(labels, 37.0), labels)
    noisy = measure_slice(_slice(labels, 37.0, noise=12.0, blur=3.0), labels)
    assert noisy.border < sharp.border
    assert noisy.level in (HARDLY, NOT_VISIBLE)


def test_a_label_without_contrast_is_not_visible():
    labels = _labels()
    result = measure_slice(_slice(labels, 30.0), labels)
    assert result.level == NOT_PLAUSIBLE or result.level == NOT_VISIBLE
    assert not result.visible


def test_a_label_on_something_much_denser_is_not_plausible():
    labels = _labels()
    result = measure_slice(_slice(labels, 70.0), labels)  # e.g. metal streaks or blood
    assert not result.plausible and result.level == NOT_PLAUSIBLE and result.border == 0.0


def test_slices_without_labelled_nuclei_are_left_out():
    labels = _labels()
    empty = {"Internal Capsule": labels["Internal Capsule"], "Lentiform Nucleus": np.zeros_like(labels["Internal Capsule"])}
    slices = {0: _slice(labels, 37.0), 1: _slice(labels, 30.0)}
    result = evaluate_basal_ganglia({0: labels, 1: empty, 2: {}}, lambda i: slices.get(i, np.zeros((120, 120))))
    assert set(result) == {0}
    assert measure_slice(slices[0], {"Lentiform Nucleus": labels["Lentiform Nucleus"]}) is None  # no capsule


def _deep(lentiform_rows, thalamus_rows=0, shape=(120, 120)):
    """Labels of one slice with `lentiform_rows` rows of lentiform (and thalamus) - the deep grey matter amount."""
    masks = {name: np.zeros(shape, bool) for name in ("Lentiform Nucleus", "Internal Capsule", "Thalamus")}
    masks["Lentiform Nucleus"][10:10 + lentiform_rows, 30:50] = True
    masks["Thalamus"][60:60 + thalamus_rows, 30:50] = True
    return masks


def test_the_block_is_the_contiguous_run_around_the_most_deep_grey_matter():
    masks = {
        0: _deep(2),          # skull base: a stray label, separated by an empty slice
        1: _deep(0),
        2: _deep(10), 3: _deep(30), 4: _deep(40), 5: _deep(20), 6: _deep(0, thalamus_rows=15),  # the real block
        7: _deep(1),          # below 8 % of the peak: the block ends here
        8: _deep(12), 9: _deep(15),  # vertex: labels the CNN draws far from the nuclei
    }

    assert basal_ganglia_block(masks) == {2, 3, 4, 5, 6}


def test_the_thalamus_keeps_the_block_together_over_a_weakly_labelled_slice():
    masks = {0: _deep(30), 1: _deep(1, thalamus_rows=20), 2: _deep(25)}

    assert basal_ganglia_block(masks) == {0, 1, 2}


def test_no_block_without_nuclei_or_segmentation():
    assert basal_ganglia_block({}) == set()
    assert basal_ganglia_block({0: _deep(0, thalamus_rows=20)}) == set()
    gap = {0: _deep(30), 2: _deep(30)}  # slice 1 not segmented: ends the block
    assert basal_ganglia_block(gap) == {0}


def test_labels_outside_the_block_are_not_measured():
    labels = _labels()
    stray = {name: mask.copy() for name, mask in labels.items()}
    stray["Lentiform Nucleus"][:] = False
    stray["Lentiform Nucleus"][40:52, 30:50] = True  # small label far from the block
    masks = {0: labels, 1: labels, 2: {}, 3: stray}
    img = _slice(labels, 37.0)

    assert set(evaluate_basal_ganglia(masks, lambda i: img)) == {0, 1}
