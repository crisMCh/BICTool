"""Study-level (global) QA over the per-slice results of one series
(FR-04's global half): each metric's mean/SD across slices (both QA
recipes) and, for the legacy (octave) recipe, the majority vote of the
chest-CT SVM's slice verdicts. The physical recipe's study verdict is the
head-CT model's (`head_ct_verdict`). Used by the Results page and the study
export alike.

Pure NumPy - no Qt, no file I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# Summarized metrics, in display order: name -> how to read it off a
# `SliceQualityResult` (the same values the per-slice views show).
METRICS = {
    "mtf_area": lambda result: result.mtf_area,
    "nps_variance": lambda result: result.nps_roi2.var_nps,
    "ps_mtf_ratio": lambda result: result.ratio.ratio_lf_hf,
    "contrast": lambda result: result.contrast_roi1,
    "decision_value": lambda result: result.decision_value,
}

ACCEPTABLE = "Acceptable"
UNACCEPTABLE = "Unacceptable"
INCONCLUSIVE = "Inconclusive (tie)"


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class StudyVerdict:
    verdict: str
    acceptable: int
    total: int


def summarize_metrics(results: Iterable) -> dict[str, MetricSummary]:
    """Mean/SD over slices for each of `METRICS`. Non-finite values are
    skipped; a metric with no finite value (or no results) is left out.
    """
    results = list(results)
    summary = {}
    for name, read in METRICS.items():
        values = np.asarray([read(result) for result in results], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            summary[name] = MetricSummary(float(values.mean()), float(values.std()), int(values.size))
    return summary


def study_verdict(results: Iterable) -> StudyVerdict | None:
    """Majority vote of the slices' verdicts (a tie is inconclusive); None
    without results. Failed slices have no result and so don't vote.
    """
    labels = [result.predicted_label for result in results]
    if not labels:
        return None
    acceptable = sum(1 for label in labels if label == 1)
    if acceptable * 2 > len(labels):
        verdict = ACCEPTABLE
    elif acceptable * 2 < len(labels):
        verdict = UNACCEPTABLE
    else:
        verdict = INCONCLUSIVE
    return StudyVerdict(verdict, acceptable, len(labels))
