"""Regression tests for `radial_average` and `nps_complete_roi`, verified
against real Octave's `radialAverage`/`funcCalcNPScompleteROI` on a small
synthetic (non-patient) image (digits of pi, chosen to avoid the
point-symmetry that would make e.g. `magic(7)` or an arithmetic-progression
matrix radially uniform and thus a weak test).
"""

import numpy as np
import pytest

from bictool.domain.metrics.nps import radial_average, nps_complete_roi

IMG = np.array(
    [
        [3, 1, 4, 1, 5, 9, 2],
        [6, 5, 3, 5, 8, 9, 7],
        [9, 3, 2, 3, 8, 4, 6],
        [2, 6, 4, 3, 3, 8, 3],
        [2, 7, 9, 5, 0, 2, 8],
        [8, 4, 1, 9, 7, 1, 6],
        [9, 3, 9, 9, 3, 7, 5],
    ],
    dtype=np.float64,
)


def test_radial_average_matches_octave():
    prof = radial_average(IMG, 4, 4, [0, 1, 2, 3])
    np.testing.assert_allclose(prof, [3.0, 4.25, 5.05, 5.0625], rtol=1e-9)


def test_nps_complete_roi_matches_octave():
    result = nps_complete_roi(IMG, px=0.5)
    assert result.var_measured == pytest.approx(7.6454081632653, rel=1e-9)
    assert result.var_nps == pytest.approx(7.43357952591326, rel=1e-9)
    np.testing.assert_allclose(result.f_avg, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        result.nps_measured_radagv,
        [1.00003716580507, 1.73589544756851, 2.12329672952305],
        rtol=1e-9,
    )
