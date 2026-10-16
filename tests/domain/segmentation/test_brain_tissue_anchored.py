"""Tests the anchored WM/GM prototype and Adaptive's smoothing on synthetic slices."""

import numpy as np
from scipy import ndimage as ndi

from bictool.domain.segmentation.brain_tissue_adaptive import classify_brain_tissue_adaptive
from bictool.domain.segmentation.brain_tissue_anchored import classify_brain_tissue_anchored, fit_anchored_model
from bictool.domain.segmentation.tissue_references import TissueReferences

SPACING = 0.5


def _brain_slice(drift_hu=0.0, seed=0):
    """A round 'brain': white matter, a gray rim (cortex), a gray block (nucleus) next to a
    white strip (capsule); optional drift that brightens everything near the edge."""
    shape = (160, 160)
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    brain = (rr - 80) ** 2 + (cc - 80) ** 2 <= 70 ** 2
    depth = ndi.distance_transform_edt(brain) * SPACING
    img = np.full(shape, 30.0)
    img[brain & (depth < 3.0)] = 37.0  # cortex
    nucleus = np.zeros(shape, bool)
    nucleus[60:100, 50:70] = True
    capsule = np.zeros(shape, bool)
    capsule[60:100, 70:80] = True
    img[nucleus] = 37.0
    img += drift_hu * np.exp(-depth / 10.0) * brain
    img += np.random.default_rng(seed).normal(0, 3.0, shape)
    img[~brain] = -1000.0
    labels = {"Lentiform Nucleus": nucleus, "Internal Capsule": capsule}
    return img, brain, labels, depth


def test_smoothing_reduces_the_speckle_of_adaptive():
    img, brain, _labels, _depth = _brain_slice()
    raw = classify_brain_tissue_adaptive(img, brain).masks["Gray Matter"]
    smooth = classify_brain_tissue_adaptive(img, brain, smoothing_sigma_px=2.0).masks["Gray Matter"]
    assert ndi.label(smooth)[1] < ndi.label(raw)[1]


def test_the_anchored_prototype_removes_an_edge_drift():
    slices = {i: _brain_slice(drift_hu=6.0, seed=i) for i in range(3)}
    refs = TissueReferences(gray=37.0, white=30.0, csf=8.0)
    model = fit_anchored_model(
        {i: s[2] for i, s in slices.items()}, lambda i: slices[i][0], {i: s[1] for i, s in slices.items()}, SPACING, refs
    )
    assert model is not None
    img, brain, labels, depth = slices[1]
    result = classify_brain_tissue_anchored(img, brain, model, 1).masks
    deep_white = brain & (depth > 5.0) & (depth < 12.0) & ~ndi.binary_dilation(labels["Lentiform Nucleus"], iterations=4)
    # the band just inside the cortex is white matter but ~3 HU brighter from the drift
    assert result["White Matter"][deep_white].mean() > 0.7
    assert result["Gray Matter"][ndi.binary_erosion(labels["Lentiform Nucleus"], iterations=3)].mean() > 0.7


def test_without_basal_ganglia_there_is_no_model():
    img, brain, labels, _ = _brain_slice()
    empty = {"Internal Capsule": labels["Internal Capsule"]}
    model = fit_anchored_model({0: empty}, lambda i: img, {0: brain}, SPACING, TissueReferences(37.0, 30.0, 8.0))
    assert model is None
