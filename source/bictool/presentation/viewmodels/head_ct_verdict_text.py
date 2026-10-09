"""Text for the study-level head-CT verdict (`domain.results.head_ct_verdict`)
on the Results page, in two levels:

* **basics** - what a user needs: the verdict and where it leans, the main
  reasons in plain words, whether artefacts are possible, unusual slices;
* **details** (behind "Details") - every number with its 95 % interval or
  reference range, so no statement is made without its uncertainty
  (docs/qa_method/head_ct_classifier.md, D7-D9).

Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from bictool.domain.results.head_ct_verdict import (
    ACCEPTABLE,
    FEATURE_LABELS,
    MIN_SLICES,
    MIN_SLICES_ARTEFACT,
    NOT_ACCEPTABLE,
    OUTLIER_FACTOR,
    PROVISIONAL_NOTE,
    HeadCtVerdict,
)

__all__ = [
    "PROVISIONAL_NOTE", "PROVISIONAL_SHORT", "VerdictText", "is_outlier", "resembles", "slice_hint", "slice_score",
    "verdict_text",
]

# Tone of the headline -> the view picks the color.
TONE_GOOD = "good"
TONE_BAD = "bad"
TONE_NEUTRAL = "neutral"
_TONES = {ACCEPTABLE: TONE_GOOD, NOT_ACCEPTABLE: TONE_BAD}

PROVISIONAL_SHORT = "Provisional model - decision support only."
SEPTUM_NOTE = "The septum pair is measured but not used for the verdict (no quality signal in the training data)."
OUTLIER_TEXT = "Noise unusually high on this slice"
_PLAUSIBILITY_LABELS = {"mtf_area": "sharpness (MTF area)", "nps_f_av": "noise texture (f_av)"}
_PLAUSIBILITY_PLAIN = {"mtf_area": "sharpness", "nps_f_av": "noise texture"}
# Plain names of the model's features for the basic explanation.
_PLAIN = {
    "mtf_area": "Sharpness",
    "nps_homo": "Noise in brain tissue",
    "nps_struct": "Noise at the falx",
    "var_struct": "Signal variation at the falx",
    "ratio_lf_hf": "Structure-to-noise balance",
}
# Per-slice hint: the study model on one slice is only indicative, so it is
# shown as a word, not a number (bands of the model's P(acceptable)).
_HINT_HIGH = 0.75
_HINT_LOW = 0.25


@dataclass(frozen=True)
class VerdictText:
    headline: str
    tone: str
    basics: tuple[str, ...]
    details: tuple[str, ...]


def _num(x: float) -> str:
    return f"{x:.3g}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def _slices(indices) -> str:
    return ", ".join(str(i + 1) for i in indices)  # 0-based index -> slice number


def _headline(v: HeadCtVerdict) -> str:
    likely = f"{_pct(v.probability)} likely acceptable"
    span = f"{100 * v.probability_low:.0f}-{100 * v.probability_high:.0f} %"
    if v.verdict == ACCEPTABLE:
        return f"Study verdict: Acceptable - {likely} (range {span})"
    if v.verdict == NOT_ACCEPTABLE:
        return f"Study verdict: Not acceptable - only {likely} (range {span})"
    leaning = "acceptable" if v.probability >= 0.5 else "not acceptable"
    return f"Study verdict: Uncertain, leaning {leaning} - {likely} (could be {span})"


def _artefact_basic(v: HeadCtVerdict) -> str:
    a = v.artefact
    if a is None:
        return "Artefacts (noise across slices): cannot be judged (too few slices)."
    if a.flagged:
        return "Artefacts (noise across slices): likely - the noise changes strongly from slice to slice."
    if a.above_good_range:
        return "Artefacts (noise across slices): possible - the noise changes more between slices than in good studies."
    return "Artefacts (noise across slices): no sign."


def _details(v: HeadCtVerdict) -> list[str]:
    details = [
        f"P(acceptable) {v.probability:.2f}, 95 % interval {v.probability_low:.2f}-{v.probability_high:.2f}; "
        f"from {v.n_slices} falx slices; model {v.model_version} (provisional)."
    ]
    details.append("Measurements (falx median, 95 % interval; range of the good training studies):")
    reasons = {f.name for f in v.reasons}
    for f in v.features:
        mark = "  <- reason" if f.name in reasons else ""
        details.append(
            f"  • {FEATURE_LABELS[f.name]}: {_num(f.value)} ({_num(f.low)}-{_num(f.high)}; "
            f"good {_num(f.good_low)}-{_num(f.good_high)}){mark}"
        )
    a = v.artefact
    if a is None:
        details.append(f"Artefact indicator: needs at least {MIN_SLICES_ARTEFACT} falx slices.")
    else:
        level = (
            "above the good range, confirmed (whole interval above)" if a.flagged
            else "above the good range, not confirmed (interval overlaps)" if a.above_good_range
            else "within the good range"
        )
        details.append(
            f"Artefact indicator (provisional, 2 artefact studies): noise scatter across slices "
            f"{a.value:.2f} (95 % interval {a.low:.2f}-{a.high:.2f}; good studies ≤ {a.good_high:.2f}) - {level}."
        )
    if v.outlier_slices:
        details.append(f"Slices with noise above {OUTLIER_FACTOR:g}x the study median: {_slices(v.outlier_slices)}.")
    for check in v.plausibility:
        label = _PLAUSIBILITY_LABELS.get(check.name, check.name)
        state = "consistent with training" if check.consistent else "outside the training range - check the ROIs"
        details.append(
            f"Septum vs falx, {label}: difference {check.difference:+.3g} "
            f"(training {check.low:+.3g} to {check.high:+.3g}) - {state}."
        )
    return details


def verdict_text(verdict: HeadCtVerdict | None, falx_slices: int, *, model_available: bool = True) -> VerdictText:
    """Headline, basic lines and detail lines. `falx_slices`: falx results
    available (for the explanation when there is no verdict)."""
    if not model_available:
        return VerdictText("Study verdict: not available (model file missing)", TONE_NEUTRAL, (), ())
    if verdict is None:
        return VerdictText(
            "Study verdict: not available",
            TONE_NEUTRAL,
            (f"Needs at least {MIN_SLICES} measured falx slices; this study has {falx_slices}.",),
            (),
        )
    basics = []
    if verdict.reasons:
        basics += [
            f"• {_PLAIN[f.name]} is {'higher' if f.outside == 'above' else 'lower'} than in good studies."
            for f in verdict.reasons
        ]
    else:
        basics.append("No measurement is outside the range of good studies.")
    basics.append(_artefact_basic(verdict))
    if verdict.outlier_slices:
        basics.append(f"Unusually noisy slices: {_slices(verdict.outlier_slices)}.")
    disagree = [_PLAUSIBILITY_PLAIN.get(c.name, c.name) for c in verdict.plausibility if not c.consistent]
    if disagree:
        basics.append(f"Check the ROIs: falx and septum disagree on {' and '.join(disagree)}.")
    return VerdictText(_headline(verdict), _TONES.get(verdict.verdict, TONE_NEUTRAL), tuple(basics), tuple(_details(verdict)))


def slice_score(verdict: HeadCtVerdict | None, index: int) -> float | None:
    """The indicative per-slice P(acceptable) of falx slice `index`, if any."""
    if verdict is None or index not in verdict.slice_ids:
        return None
    return float(verdict.slice_probabilities[verdict.slice_ids.index(index)])


def resembles(score: float | None) -> str:
    """One word for a slice's hint: "acceptable", "not acceptable", "unclear" ("" without one)."""
    if score is None:
        return ""
    if score >= _HINT_HIGH:
        return "acceptable"
    if score <= _HINT_LOW:
        return "not acceptable"
    return "unclear"


def slice_hint(score: float | None) -> tuple[str, str]:
    """(text, tooltip) for one slice: what the study model says about it, in
    words - the number is only in the tooltip, because one slice is only a hint."""
    if score is None:
        return "", ""
    word = resembles(score)
    words = "unclear" if word == "unclear" else word.replace("not acceptable", "not-acceptable") + " studies"
    tooltip = (
        f"Quality hint {score:.2f} of 1. Only a hint: the study model was trained on whole studies, "
        "and the verdict uses all falx slices (Study summary tab)."
    )
    return f"This slice resembles: {words}", tooltip


def is_outlier(verdict: HeadCtVerdict | None, index: int) -> bool:
    return verdict is not None and index in verdict.outlier_slices
