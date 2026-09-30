"""Tests the study-level summary (means, majority verdict)."""

from types import SimpleNamespace

import pytest

from bictool.domain.results.study_summary import (
    ACCEPTABLE,
    INCONCLUSIVE,
    UNACCEPTABLE,
    study_verdict,
    summarize_metrics,
)


def _result(mtf, nps=1.0, ratio=2.0, contrast=10.0, label=1, decision=0.5):
    return SimpleNamespace(
        mtf_area=mtf,
        nps_roi2=SimpleNamespace(var_nps=nps),
        ratio=SimpleNamespace(ratio_lf_hf=ratio),
        contrast_roi1=contrast,
        predicted_label=label,
        decision_value=decision,
    )


def test_means_each_metric_over_all_slices():
    summary = summarize_metrics([_result(0.2, nps=1.0, contrast=10.0), _result(0.4, nps=3.0, contrast=30.0)])

    assert summary["mtf_area"].mean == pytest.approx(0.3)
    assert summary["mtf_area"].std == pytest.approx(0.1)
    assert summary["nps_variance"].mean == pytest.approx(2.0)
    assert summary["contrast"].mean == pytest.approx(20.0)
    assert summary["contrast"].count == 2


def test_non_finite_values_are_skipped():
    summary = summarize_metrics([_result(0.2, ratio=float("nan")), _result(0.4, ratio=4.0)])

    assert summary["ps_mtf_ratio"].mean == 4.0
    assert summary["ps_mtf_ratio"].count == 1


def test_no_results_gives_an_empty_summary_and_no_verdict():
    assert summarize_metrics([]) == {}
    assert study_verdict([]) is None


@pytest.mark.parametrize(
    ("labels", "expected"),
    [([1, 1, -1], ACCEPTABLE), ([1, -1, -1], UNACCEPTABLE), ([1, -1], INCONCLUSIVE)],
)
def test_study_verdict_is_a_majority_vote(labels, expected):
    verdict = study_verdict([_result(0.3, label=label) for label in labels])

    assert verdict.verdict == expected
    assert verdict.acceptable == labels.count(1)
    assert verdict.total == len(labels)
