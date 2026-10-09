"""Tests the verdict wording: plain basics for users, every number with its
interval in the details (no patient data)."""

import numpy as np

from bictool.domain.results.head_ct_verdict import CORE_FEATURES, HeadCtModel, evaluate_study
from bictool.presentation.viewmodels.head_ct_verdict_text import (
    TONE_BAD,
    TONE_GOOD,
    TONE_NEUTRAL,
    is_outlier,
    slice_hint,
    slice_score,
    verdict_text,
)


def _model():
    """One informative feature (nps_homo): lower noise = acceptable, P = 0.5 at 20."""
    w = np.array([0.0, -1.0, 0.0, 0.0, 0.0])
    return HeadCtModel(
        version="test-1", feature_names=CORE_FEATURES, weights=w, intercept=20.0,
        boot_weights=np.tile(w, (200, 1)), boot_intercepts=np.full(200, 20.0),
        feature_mean=np.array([0.5, 20.0, 100.0, 90.0, 12.0]),
        good_low=np.array([0.49, 15.0, 50.0, 45.0, 8.0]),
        good_high=np.array([0.54, 19.0, 170.0, 125.0, 20.0]),
        artefact_good_high=0.8,
        plausibility_names=("mtf_area", "nps_f_av"),
        plausibility_low=np.array([-0.03, -0.02]),
        plausibility_high=np.array([0.03, 0.06]),
    )


def _slices(nps_homo):
    n = len(nps_homo)
    values = {name: np.full(n, v) for name, v in
              (("mtf_area", 0.5), ("nps_struct", 100.0), ("var_struct", 90.0), ("ratio_lf_hf", 12.0),
               ("nps_f_av", 0.2), ("nps_f_peak", 0.1))}
    values["nps_homo"] = np.asarray(nps_homo, dtype=float)
    return values


def _steady(level, n=25, spread=0.3, seed=1):
    return np.random.default_rng(seed).normal(level, spread, n)


def test_acceptable_study_headline_has_probability_and_interval():
    verdict = evaluate_study(_model(), _slices(_steady(15.0)))

    text = verdict_text(verdict, 25)

    assert text.tone == TONE_GOOD
    assert text.headline.startswith("Study verdict: Acceptable - ")
    assert "likely acceptable (range " in text.headline
    assert "No measurement is outside the range of good studies." in text.basics
    assert "95 % interval" in text.details[0]


def test_not_acceptable_study_lists_each_reason_with_interval_and_good_range():
    text = verdict_text(evaluate_study(_model(), _slices(_steady(25.0))), 25)

    assert text.tone == TONE_BAD
    assert "• Noise in brain tissue is higher than in good studies." in text.basics
    reason = next(line for line in text.details if "Noise in homogeneous tissue" in line)
    assert "<- reason" in reason and "good " in reason


def test_uncertain_study_is_neutral_toned():
    text = verdict_text(evaluate_study(_model(), _slices(_steady(20.0, spread=2.0))), 25)

    assert text.tone == TONE_NEUTRAL and "Uncertain, leaning" in text.headline


def test_missing_verdict_and_missing_model_are_explained():
    few = verdict_text(None, 3)
    assert few.headline == "Study verdict: not available" and "this study has 3" in few.basics[0]
    assert "model file missing" in verdict_text(None, 30, model_available=False).headline


def test_artefact_levels_are_worded_differently():
    steady = verdict_text(evaluate_study(_model(), _slices(_steady(15.0))), 25)
    jumpy = verdict_text(evaluate_study(_model(), _slices(np.geomspace(2.0, 200.0, 25))), 25)

    assert "within the good range" in " ".join(steady.details)
    assert "above the good range, confirmed" in " ".join(jumpy.details)
    assert "provisional" in " ".join(jumpy.details)
    assert "Artefacts (noise across slices): no sign." in steady.basics
    assert "Artefacts (noise across slices): likely - the noise changes strongly from slice to slice." in jumpy.basics


def test_outlier_slices_are_listed_as_one_based_numbers_and_slice_lookup_works():
    noise = _steady(15.0)
    noise[5] = 50.0
    verdict = evaluate_study(_model(), _slices(noise), slice_ids=np.arange(100, 125))

    text = verdict_text(verdict, 25)

    assert any("above 3x the study median: 106" in line for line in text.details)
    assert "Unusually noisy slices: 106." in text.basics
    assert is_outlier(verdict, 105) and not is_outlier(verdict, 104) and not is_outlier(None, 105)
    assert 0.0 <= slice_score(verdict, 105) <= 1.0
    assert slice_score(verdict, 3) is None and slice_score(None, 105) is None


def test_slice_hint_uses_words_and_keeps_the_number_in_the_tooltip():
    assert slice_hint(0.98)[0] == "This slice resembles: acceptable studies"
    assert slice_hint(0.10)[0] == "This slice resembles: not-acceptable studies"
    assert slice_hint(0.50)[0] == "This slice resembles: unclear"
    assert "0.98 of 1" in slice_hint(0.98)[1] and "Only a hint" in slice_hint(0.98)[1]
    assert slice_hint(None) == ("", "")
