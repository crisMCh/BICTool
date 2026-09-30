"""Wiring tests for `evaluate_slice_quality`.

This does NOT test numeric parity against a real Octave run end-to-end:
`calc_mtf` has no committed bit-exact-vs-Octave test (its validation lives
only in local, gitignored scratch scripts, because of a known and
deliberately-deferred bump-removal-stage discrepancy - see project memory /
`reference/octave/TRANSLATION_STATUS.md`'s "Known open issues"). So
`mtf_area`, `feature_vector[0]`, `predicted_label`, and `decision_value`
here are NOT expected to match a real Octave run bit-for-bit.

What this DOES test: that `evaluate_slice_quality` wires the
already-independently-validated component functions (`nps_complete_roi`,
`calc_contrast`, `ratio_nps_mtf`, `classify_quality`) together correctly -
in particular the ROI1-vs-ROI2 and `var_nps`-vs-`var_measured` feature
mappings, which are the highest-risk detail in the whole recipe.
"""

import numpy as np

from bictool.domain.metrics.contrast import calc_contrast
from bictool.domain.metrics.nps import nps_complete_roi
from bictool.domain.metrics.ps_mtf_ratio import ratio_nps_mtf
from bictool.domain.results.classification import classify_quality
from bictool.domain.results.study_quality import evaluate_slice_quality

# Small synthetic (non-patient) ROIs: digits-of-pi pattern (see
# tests/domain/metrics/test_nps.py) and a shifted variant, so ROI1 and ROI2
# are genuinely different - a ROI1/ROI2 mixup in the wiring would otherwise
# go undetected.
IMG_ROI1 = np.array(
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
IMG_ROI2 = IMG_ROI1 + 100.0  # distinct values, same shape
PIX_SPC = 0.5


def test_evaluate_slice_quality_wires_nps_and_contrast_correctly(monkeypatch):
    # calc_mtf needs a real edge to segment; stub it out since this test is
    # only about wiring the OTHER metrics and the feature vector, not MTF
    # itself (see module docstring).
    # Frequency domain must fully cover the NPS curve's domain (f_avg for
    # this ROI spans [0, 1]) or ratio_nps_mtf's interp1_linear correctly
    # returns NaN outside it - not a wiring bug, just a fixture-domain
    # mismatch, so use a wide-enough range here.
    fake_xaxis = np.array([0.0, 0.3, 0.6, 0.9, 1.2])
    fake_mtf = np.array([[1.0, 1.0], [0.8, 0.75], [0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])
    fake_mtf_area_per_edge = np.array([0.3, 0.4])

    import bictool.domain.metrics.mtf_seed as mtf_seed

    monkeypatch.setattr(mtf_seed, "calc_mtf", lambda *a, **k: (fake_xaxis, fake_mtf, fake_mtf_area_per_edge))

    result = evaluate_slice_quality(IMG_ROI1, IMG_ROI2, PIX_SPC)

    expected_nps_roi1 = nps_complete_roi(IMG_ROI1, PIX_SPC)
    expected_nps_roi2 = nps_complete_roi(IMG_ROI2, PIX_SPC)
    np.testing.assert_allclose(result.nps_roi1.nps_measured_radagv, expected_nps_roi1.nps_measured_radagv)
    np.testing.assert_allclose(result.nps_roi2.nps_measured_radagv, expected_nps_roi2.nps_measured_radagv)
    assert result.nps_roi1.var_measured == expected_nps_roi1.var_measured
    assert result.nps_roi1.var_nps == expected_nps_roi1.var_nps
    assert result.nps_roi2.var_nps == expected_nps_roi2.var_nps

    assert result.contrast_roi1 == calc_contrast(IMG_ROI1)
    assert result.contrast_roi2 == calc_contrast(IMG_ROI2)

    expected_ratio = ratio_nps_mtf(
        expected_nps_roi1.f_avg, expected_nps_roi1.nps_measured_radagv, fake_xaxis, fake_mtf
    )
    assert result.ratio.ratio_lf_hf == expected_ratio.ratio_lf_hf
    assert result.ratio.slope_avg == expected_ratio.slope_avg
    assert result.ratio.loc_max == expected_ratio.loc_max

    assert result.mtf_area == float(np.mean(fake_mtf_area_per_edge))

    # The feature vector is the single highest-risk detail: it must be
    # [mtfArea, npsHomo=ROI2.var_nps, npsStruct=ROI1.var_nps,
    #  varStruct=ROI1.var_measured, ratioLfHf, slopeAvg, locMax].
    expected_features = np.array([
        result.mtf_area,
        expected_nps_roi2.var_nps,
        expected_nps_roi1.var_nps,
        expected_nps_roi1.var_measured,
        expected_ratio.ratio_lf_hf,
        expected_ratio.slope_avg,
        expected_ratio.loc_max,
    ])
    np.testing.assert_allclose(result.feature_vector, expected_features)

    expected_label, expected_decision = classify_quality(expected_features)
    assert result.predicted_label == expected_label
    assert result.decision_value == expected_decision


def test_evaluate_slice_quality_retries_mtf_seeds_until_one_finds_edges(monkeypatch):
    import bictool.domain.metrics.mtf_seed as mtf_seed

    tried = []

    def fake_calc_mtf(img_roi, x, y, pix_spc):
        tried.append((x, y))
        if (x, y) != (3, 4):
            raise ValueError("No edge columns survived bump removal")
        return np.array([0.0, 0.5, 1.0]), np.ones((3, 2)), np.array([0.3, 0.4])

    monkeypatch.setattr(mtf_seed, "calc_mtf", fake_calc_mtf)

    result = evaluate_slice_quality(IMG_ROI1, IMG_ROI2, PIX_SPC, mtf_seeds=[(2, 2), (3, 4), (5, 5)])

    assert tried == [(2, 2), (3, 4)]
    assert result.mtf_seed == (3, 4)


def _stub_mtf(monkeypatch, mtf, areas):
    import bictool.domain.metrics.mtf_seed as mtf_seed

    xaxis = np.linspace(0.0, 1.2, mtf.shape[0])
    monkeypatch.setattr(mtf_seed, "calc_mtf", lambda *a, **k: (xaxis, mtf, areas))


def test_edges_with_a_non_finite_mtf_are_dropped_everywhere(monkeypatch):
    good = np.linspace(1.0, 0.1, 5)
    mtf = np.column_stack([good, good, np.full(5, np.inf)])
    _stub_mtf(monkeypatch, mtf, np.array([0.3, 0.5, np.nan]))

    result = evaluate_slice_quality(IMG_ROI1, IMG_ROI2, PIX_SPC)

    assert result.mtf.shape[1] == 2
    assert result.mtf_area == 0.4
    assert np.isfinite(result.feature_vector).all()


def test_slice_fails_when_most_edges_have_a_non_finite_mtf(monkeypatch):
    import pytest

    good = np.linspace(1.0, 0.1, 5)
    mtf = np.column_stack([good, np.full(5, np.inf), np.full(5, np.inf)])
    _stub_mtf(monkeypatch, mtf, np.array([0.3, np.nan, np.nan]))

    with pytest.raises(ValueError, match="2 of 3 edge points"):
        evaluate_slice_quality(IMG_ROI1, IMG_ROI2, PIX_SPC)
