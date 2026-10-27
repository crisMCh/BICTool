"""Tests `domain.segmentation.tissue_references` with synthetic slices and labels."""

import numpy as np
import pytest

from bictool.domain.segmentation.tissue_references import (
    SliceSamples,
    TissueReferences,
    combine,
    estimate_tissue_references,
    slice_samples,
)


def _slice(nucleus_hu=37.0, capsule_hu=30.0, csf_hu=8.0, seed=0):
    shape = (120, 120)
    labels = {name: np.zeros(shape, bool) for name in
              ("Lentiform Nucleus", "Caudate Nucleus", "Internal Capsule", "Ventricular System")}
    labels["Lentiform Nucleus"][40:80, 20:40] = True
    labels["Internal Capsule"][40:80, 40:52] = True
    labels["Caudate Nucleus"][20:35, 55:70] = True
    labels["Ventricular System"][40:80, 60:90] = True
    img = np.full(shape, capsule_hu)
    img[labels["Lentiform Nucleus"] | labels["Caudate Nucleus"]] = nucleus_hu
    img[labels["Ventricular System"]] = csf_hu
    return img + np.random.default_rng(seed).normal(0, 1.0, shape), labels


def test_references_come_from_the_basal_ganglia_and_the_ventricles():
    img, labels = _slice()
    refs = estimate_tissue_references({0: labels, 1: labels}, lambda i: img)
    assert refs.gray == pytest.approx(37.0, abs=0.5)
    assert refs.white == pytest.approx(30.0, abs=0.5)
    assert refs.csf == pytest.approx(8.0, abs=0.5)
    low, high = refs.plausible_range()
    assert low == pytest.approx(19.0, abs=0.5) and high == pytest.approx(52.0, abs=0.5)


def test_an_implausible_basal_ganglia_slice_gives_no_gray_or_white_sample():
    img, labels = _slice(nucleus_hu=70.0)  # label on something much denser (e.g. metal streaks)
    samples = slice_samples(img, labels)
    assert samples.gray == () and samples.white == () and not samples.basal_ganglia
    assert samples.csf  # the ventricles still count
    assert combine([samples]) is None


def test_combine_takes_medians_and_needs_all_three():
    samples = [SliceSamples(gray=(36.0,), white=(29.0,), csf=(10.0,)), SliceSamples(gray=(38.0, 40.0), white=(31.0,))]
    assert combine(samples) == TissueReferences(gray=38.0, white=30.0, csf=10.0)
    assert combine([SliceSamples(gray=(36.0,), white=(29.0,))]) is None
