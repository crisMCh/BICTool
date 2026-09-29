"""Tests `brain_tissue.py`'s skull-strip and HU-band classification against
synthetic, hand-built slices - never real patient data, per this project's
test-data policy.
"""

import numpy as np

from bictool.domain.segmentation.brain_tissue import classify_brain_tissue, extract_brain_mask

_AIR_HU = -1000.0
_BRAIN_HU = 40.0
_SKULL_HU = 1000.0
_CALCIFICATION_HU = 500.0


def _make_head_slice(size: int = 100) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Builds a synthetic axial slice: a filled brain-HU disk, wrapped in a
    thin skull-HU ring, surrounded by air - plus a small disconnected bright
    artifact elsewhere and a small calcification-HU dot inside the disk.

    Returns `(slice_hu, (cy, cx, radius))` - the disk's own geometry, so
    tests can address pixels relative to it without hardcoding coordinates
    twice.
    """
    slice_hu = np.full((size, size), _AIR_HU, dtype=np.float64)
    cy, cx, radius = size // 2, size // 2, size // 3

    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    ring = (dist >= radius) & (dist < radius + 4)
    slice_hu[ring] = _SKULL_HU

    disk = dist < radius
    slice_hu[disk] = _BRAIN_HU

    # Disconnected artifact, far from the brain disk - must not survive the
    # largest-connected-component step.
    slice_hu[2:6, 2:6] = _SKULL_HU

    # A calcification-HU dot inside the disk - must get filled back in.
    slice_hu[cy - 2 : cy + 2, cx - 2 : cx + 2] = _CALCIFICATION_HU

    return slice_hu, (cy, cx, radius)


def test_extract_brain_mask_keeps_only_the_brain_disk():
    slice_hu, (cy, cx, radius) = _make_head_slice()

    mask = extract_brain_mask(slice_hu)

    # Well inside the disk: brain.
    assert mask[cy, cx]
    # Well outside the disk (in the skull ring or air): not brain.
    assert not mask[0, 0]
    assert not mask[cy, cx + radius + 2]
    # The disconnected artifact must not survive largest-component selection.
    assert not mask[3, 3]


def test_extract_brain_mask_fills_the_calcification_hole():
    slice_hu, (cy, cx, _radius) = _make_head_slice()

    mask = extract_brain_mask(slice_hu)

    assert mask[cy, cx]


def test_extract_brain_mask_returns_boolean_array_of_same_shape():
    slice_hu, _ = _make_head_slice()

    mask = extract_brain_mask(slice_hu)

    assert mask.dtype == bool
    assert mask.shape == slice_hu.shape


def test_classify_brain_tissue_labels_csf_white_and_gray_matter_by_hu_band():
    # Blocks are large (16x16) and well separated so the median-filter/
    # small-component clean-up (see brain_tissue.py's docstring) doesn't
    # erode them away - assertions check each block's interior, away from
    # the block/background boundary the smoothing pass legitimately blurs.
    size = 60
    slice_hu = np.zeros((size, size), dtype=np.float64)
    brain_mask = np.ones((size, size), dtype=bool)

    slice_hu[0:16, 0:16] = 28.0  # white matter band
    slice_hu[22:38, 22:38] = 40.0  # gray matter band
    slice_hu[44:60, 44:60] = 10.0  # CSF band

    result = classify_brain_tissue(slice_hu, brain_mask)

    assert result.masks["White Matter"][3:13, 3:13].all()
    assert not result.masks["Gray Matter"][3:13, 3:13].any()
    assert not result.masks["CSF"][3:13, 3:13].any()

    assert result.masks["Gray Matter"][25:35, 25:35].all()
    assert not result.masks["White Matter"][25:35, 25:35].any()
    assert not result.masks["CSF"][25:35, 25:35].any()

    assert result.masks["CSF"][47:57, 47:57].all()
    assert not result.masks["White Matter"][47:57, 47:57].any()
    assert not result.masks["Gray Matter"][47:57, 47:57].any()


def test_classify_brain_tissue_csf_does_not_leak_into_white_matter_near_the_band_boundary():
    # Regression test for the real issue reported 2026-09-29: CSF/
    # subarachnoid-space pixels right at White Matter's lower band edge
    # were occasionally read as White Matter before CSF had its own
    # explicit class to claim them first. Checks the interior only, away
    # from the array boundary the clean-up morphology legitimately erodes
    # (see the "labels_csf_white_and_gray_matter" test above).
    size = 30
    slice_hu = np.full((size, size), 19.0, dtype=np.float64)  # just below White Matter's low (20.0)
    brain_mask = np.ones((size, size), dtype=bool)

    result = classify_brain_tissue(slice_hu, brain_mask)

    assert result.masks["CSF"][5:25, 5:25].all()
    assert not result.masks["White Matter"].any()


def test_classify_brain_tissue_never_labels_outside_the_brain_mask():
    size = 10
    slice_hu = np.full((size, size), 40.0, dtype=np.float64)  # all gray-matter-band HU
    brain_mask = np.zeros((size, size), dtype=bool)  # nothing is brain

    result = classify_brain_tissue(slice_hu, brain_mask)

    assert not result.masks["CSF"].any()
    assert not result.masks["White Matter"].any()
    assert not result.masks["Gray Matter"].any()


def test_classify_brain_tissue_merges_gray_matter_regions_separated_by_a_narrow_gap():
    # Two 8x8 gray-matter-band blocks separated by a 3px gap of neither-band
    # HU - regression test for the exact symptom reported 2026-09-24 ("gray
    # matter regions are still not big enough"): the closing step must
    # bridge a gap this narrow so both blocks end up as one connected
    # component, not two isolated ones.
    size = 25
    slice_hu = np.zeros((size, size), dtype=np.float64)
    slice_hu[5:13, 5:13] = 40.0
    slice_hu[5:13, 16:24] = 40.0
    brain_mask = np.ones((size, size), dtype=bool)

    result = classify_brain_tissue(slice_hu, brain_mask)

    from scipy import ndimage

    labeled, num_components = ndimage.label(result.masks["Gray Matter"])
    assert num_components == 1
    assert labeled[8, 8] == labeled[8, 19]  # both original blocks share one label


def test_classify_brain_tissue_removes_isolated_speckle_from_a_uniform_region():
    # A single noise-driven pixel (salt) inside an otherwise-uniform gray
    # matter block, and a single noise-driven hole (pepper) inside it -
    # regression test for the exact symptom reported 2026-09-24 ("regions
    # should be more united rather than isolated spots").
    size = 30
    slice_hu = np.full((size, size), 40.0, dtype=np.float64)  # gray matter band, everywhere
    slice_hu[15, 15] = 28.0  # one noise-driven pixel that alone reads as white matter
    slice_hu[10, 10] = -100.0  # one noise-driven pixel that alone reads as neither
    brain_mask = np.ones((size, size), dtype=bool)

    result = classify_brain_tissue(slice_hu, brain_mask)

    assert result.masks["Gray Matter"][15, 15]  # cleaned up, not an isolated white-matter speck
    assert not result.masks["White Matter"].any()
    assert result.masks["Gray Matter"][10, 10]  # the single-pixel hole gets filled back in


def test_classify_brain_tissue_drops_small_isolated_components():
    size = 30
    slice_hu = np.zeros((size, size), dtype=np.float64)
    slice_hu[5:7, 5:7] = 28.0  # a tiny 2x2 speck - below the default min component size
    brain_mask = np.ones((size, size), dtype=bool)

    result = classify_brain_tissue(slice_hu, brain_mask)

    assert not result.masks["White Matter"].any()


def test_classify_brain_tissue_masks_are_boolean_and_mutually_exclusive():
    slice_hu, _ = _make_head_slice()
    brain_mask = extract_brain_mask(slice_hu)

    result = classify_brain_tissue(slice_hu, brain_mask)

    csf, white_matter, gray_matter = result.masks["CSF"], result.masks["White Matter"], result.masks["Gray Matter"]
    assert csf.dtype == bool
    assert white_matter.dtype == bool
    assert gray_matter.dtype == bool
    assert white_matter.shape == slice_hu.shape
    assert not (white_matter & gray_matter).any()
    assert not (csf & white_matter).any()
    assert not (csf & gray_matter).any()
