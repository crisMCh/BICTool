"""Regression test for `ratio_nps_mtf`, verified against real Octave's
`funcRatioNpsMtf` on small synthetic (non-patient) MTF/PS curves.
"""

import numpy as np
import pytest

from bictool.domain.metrics.ps_mtf_ratio import ratio_nps_mtf

F_MTF = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
MTF = np.array(
    [
        [1.0, 1.0],
        [0.9, 0.85],
        [0.7, 0.65],
        [0.5, 0.45],
        [0.3, 0.25],
        [0.15, 0.12],
        [0.1, 0.08],
    ]
)
F_NPS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.5])
NPS = np.array([10, 8, 6, 4, 2, 1], dtype=np.float64)


def test_ratio_nps_mtf_matches_octave():
    result = ratio_nps_mtf(F_NPS, NPS, F_MTF, MTF)

    assert result.ratio_lf_hf == pytest.approx(1.08190498926431, rel=1e-9)
    assert result.slope_avg == pytest.approx(-0.071957671957672, rel=1e-9)
    assert result.loc_max == pytest.approx(0.75609756097561, rel=1e-9)
    assert len(result.f_int) == 41
    assert result.f_int[0] == pytest.approx(0.1)
    assert result.f_int[-1] == pytest.approx(0.5)
