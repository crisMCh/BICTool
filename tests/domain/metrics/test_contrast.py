"""Regression test for `calc_contrast`, verified against real Octave's
`funcCalcContrast` on a small synthetic (non-patient) image.
"""

import numpy as np
import pytest

from bictool.domain.metrics.contrast import calc_contrast


def test_calc_contrast_matches_octave():
    img = np.arange(1, 17).reshape(4, 4)
    assert calc_contrast(img) == pytest.approx(0.54232614454664, rel=1e-9)
