"""Tests `head_ct_verdict.py` with a hand-built model - no patient data."""

import numpy as np
import pytest

from bictool.domain.results.head_ct_verdict import (
    ACCEPTABLE,
    CORE_FEATURES,
    MIN_SLICES,
    NOT_ACCEPTABLE,
    UNCERTAIN,
    HeadCtModel,
    evaluate_study,
    probability,
)

# One informative feature: nps_homo; lower noise = acceptable, P = 0.5 at 20.
_W = np.array([0.0, -1.0, 0.0, 0.0, 0.0])
_B = 20.0


def _model(boot_spread=0.0, n_boot=200):
    rng = np.random.default_rng(0)
    scale = 1 + rng.normal(0, boot_spread, n_boot)
    return HeadCtModel(
        version="test-1",
        feature_names=CORE_FEATURES,
        weights=_W,
        intercept=_B,
        boot_weights=np.outer(scale, _W),
        boot_intercepts=_B * scale,
        feature_mean=np.array([0.5, 20.0, 100.0, 90.0, 12.0]),
        good_low=np.array([0.49, 15.0, 50.0, 45.0, 8.0]),
        good_high=np.array([0.54, 19.0, 170.0, 125.0, 20.0]),
        artefact_good_high=0.8,
        plausibility_names=("mtf_area", "nps_f_av"),
        plausibility_low=np.array([-0.03, -0.02]),
        plausibility_high=np.array([0.03, 0.06]),
    )


def _slices(nps_homo, **overrides):
    nps_homo = np.asarray(nps_homo, dtype=float)
    n = len(nps_homo)
    values = {
        "mtf_area": np.full(n, 0.5),
        "nps_homo": nps_homo,
        "nps_struct": np.full(n, 100.0),
        "var_struct": np.full(n, 90.0),
        "ratio_lf_hf": np.full(n, 12.0),
        "nps_f_av": np.full(n, 0.2),
        "nps_f_peak": np.full(n, 0.1),
    }
    values.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    return values


def _steady(level, n=25, spread=0.3, seed=1):
    return np.random.default_rng(seed).normal(level, spread, n)


def test_quiet_study_is_acceptable_with_tight_interval():
    v = evaluate_study(_model(), _slices(_steady(15.0)))

    assert v.verdict == ACCEPTABLE
    assert v.probability == pytest.approx(probability(_model(), np.median(
        np.column_stack([_slices(_steady(15.0))[n] for n in CORE_FEATURES]), axis=0)))
    assert 0.5 < v.probability_low <= v.probability <= v.probability_high
    assert v.reasons == ()
    assert v.model_version == "test-1"


def test_noisy_study_is_not_acceptable_and_names_the_reason():
    v = evaluate_study(_model(), _slices(_steady(25.0)))

    assert v.verdict == NOT_ACCEPTABLE
    assert v.probability_high < 0.5
    assert [f.name for f in v.reasons] == ["nps_homo"]
    finding = v.reasons[0]
    assert finding.outside == "above" and finding.low > finding.good_high
    assert finding.contribution < 0


def test_study_at_the_boundary_or_with_uncertain_model_is_uncertain():
    assert evaluate_study(_model(), _slices(_steady(20.0, spread=2.0))).verdict == UNCERTAIN
    # A clear-ish study becomes uncertain when the bootstrap models disagree.
    assert evaluate_study(_model(boot_spread=2.0), _slices(_steady(18.0))).verdict == UNCERTAIN


def test_feature_outside_good_range_but_pushing_towards_acceptable_is_no_reason():
    v = evaluate_study(_model(), _slices(_steady(10.0)))  # quieter than every good study

    nps = next(f for f in v.features if f.name == "nps_homo")
    assert nps.outside == "below"
    assert v.reasons == ()


def test_too_few_usable_slices_give_no_verdict():
    values = _slices(_steady(15.0, n=MIN_SLICES + 1))
    values["mtf_area"][:2] = np.nan  # leaves MIN_SLICES - 1 usable

    assert evaluate_study(_model(), values) is None
    values["mtf_area"][1] = 0.5
    assert evaluate_study(_model(), values).n_slices == MIN_SLICES


def test_inconsistent_noise_raises_the_artefact_indicator():
    steady = evaluate_study(_model(), _slices(_steady(15.0)))
    jumpy = evaluate_study(_model(), _slices(np.geomspace(2.0, 200.0, 25)))  # IQR / median ~2.8

    assert not steady.artefact.above_good_range and not steady.artefact.flagged
    assert jumpy.artefact.above_good_range and jumpy.artefact.flagged
    assert jumpy.artefact.low <= jumpy.artefact.value <= jumpy.artefact.high


def test_slices_far_above_the_study_noise_are_listed_as_outliers():
    noise = _steady(15.0)
    noise[5] = 50.0  # > 3x the median

    v = evaluate_study(_model(), _slices(noise), slice_ids=np.arange(100, 125))

    assert v.outlier_slices == (105,)


def test_no_artefact_indicator_with_few_slices():
    assert evaluate_study(_model(), _slices(_steady(15.0, n=MIN_SLICES))).artefact is None


def test_plausibility_compares_septum_minus_falx_with_training_range():
    falx = _slices(_steady(15.0))
    septum = _slices(_steady(15.0, n=6), mtf_area=np.full(6, 0.6), nps_f_av=np.full(6, 0.21))

    v = evaluate_study(_model(), falx, septum)
    checks = {c.name: c for c in v.plausibility}

    assert checks["mtf_area"].difference == pytest.approx(0.1)
    assert not checks["mtf_area"].consistent
    assert checks["nps_f_av"].consistent
    assert evaluate_study(_model(), falx).plausibility == ()


def test_slice_probabilities_follow_the_usable_slices():
    values = _slices(_steady(15.0, n=8))
    values["var_struct"][3] = np.nan

    v = evaluate_study(_model(), values, slice_ids=np.arange(10, 18))

    assert v.slice_ids == (10, 11, 12, 14, 15, 16, 17)
    assert len(v.slice_probabilities) == 7


def test_evaluation_is_deterministic():
    a = evaluate_study(_model(boot_spread=0.3), _slices(_steady(19.0, spread=1.5)))
    b = evaluate_study(_model(boot_spread=0.3), _slices(_steady(19.0, spread=1.5)))

    assert (a.probability_low, a.probability_high) == (b.probability_low, b.probability_high)


def test_model_round_trips_through_plain_arrays():
    model = _model(boot_spread=0.1)
    again = HeadCtModel.from_arrays(model.to_arrays())

    assert again.version == model.version
    assert again.feature_names == model.feature_names
    assert again.plausibility_names == model.plausibility_names
    np.testing.assert_array_equal(again.boot_weights, model.boot_weights)
    assert again.intercept == model.intercept


def test_results_are_turned_into_per_slice_values_in_slice_order():
    from types import SimpleNamespace

    from bictool.domain.results.head_ct_verdict import evaluate_study_results, slice_values

    def result(mtf_area):
        return SimpleNamespace(
            feature_vector=np.array([mtf_area, 20.0, 100.0, 70.0, 11.0, -1.0, 0.1]), nps_f_av=0.2, nps_f_peak=0.1
        )

    results = {7: result(0.7), 2: result(0.2), 5: result(0.5)}
    values, ids = slice_values(results)

    assert ids == [2, 5, 7]
    np.testing.assert_allclose(values["mtf_area"], [0.2, 0.5, 0.7])
    assert set(values) == {*CORE_FEATURES, "nps_f_av", "nps_f_peak"}
    assert evaluate_study_results(_model(), results) is None  # 3 slices < MIN_SLICES
    assert evaluate_study_results(_model(), {}) is None
    assert evaluate_study_results(None, results) is None  # no model file
    many = {i: result(0.5) for i in range(10)}
    verdict = evaluate_study_results(_model(), many, many)
    assert verdict.slice_ids == tuple(range(10)) and len(verdict.plausibility) == 2
