"""Tests `domain.metrics.basal_ganglia` with synthetic slices and labels."""

import numpy as np

from bictool.domain.metrics.basal_ganglia import (
    CLEARLY,
    HARDLY,
    NOT_PLAUSIBLE,
    NOT_VISIBLE,
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
