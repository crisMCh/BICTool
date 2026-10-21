"""`region_seg` (Chan-Vese) on a synthetic bright line - the case behind the falx MTF failures."""

import numpy as np

from bictool.domain.segmentation.active_contour import region_seg


def _line_with_seed_inside(size=64, band=(26, 38)):
    rng = np.random.default_rng(0)
    image = np.full((size, size), 20.0) + rng.normal(0, 1.0, (size, size))
    image[:, band[0]:band[1]] = 60.0 + rng.normal(0, 1.0, (size, band[1] - band[0]))
    seed = np.zeros((size, size))
    seed[28:37, 28:37] = 1.0  # fully inside the bright band
    return image, seed, band


def test_a_seed_inside_a_bright_line_grows_along_it():
    image, seed, (left, right) = _line_with_seed_inside()

    region = region_seg(image, seed, 300, 0.2)

    assert region.sum() > 4 * seed.sum()                   # grew instead of collapsing
    assert region[:, left:right].mean() > 0.8               # along the whole band
    assert region[:, : left - 2].mean() < 0.05 and region[:, right + 2 :].mean() < 0.05  # not beside it


def test_the_original_octave_step_collapses_on_the_same_seed():
    image, seed, _band = _line_with_seed_inside()

    region = region_seg(image, seed, 300, 0.2, octave_step=True)

    assert region.sum() < seed.sum()  # the documented bug, kept only for the legacy recipe
