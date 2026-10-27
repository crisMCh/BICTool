"""Study-level image-quality verdict for head CT (physical QA recipe).

Design and evidence: docs/qa_method/head_ct_classifier.md (decisions D3-D10).
In short:

* A study is described by the **median** of each core feature over its falx
  (dural folds) slices; a logistic regression turns the medians into the
  probability that a radiologist would rate the study acceptable.
* The probability comes with a **95 % interval** from a paired bootstrap:
  each of the model's bootstrap refits (resampled training studies = model
  uncertainty) is evaluated on a resample of this study's slices
  (measurement uncertainty). Verdict: Acceptable if the whole interval is
  above 0.5, Not acceptable if it is below, otherwise Uncertain.
* **Reasons**: features whose 95 % interval lies entirely outside the range
  of the rated-good training studies *and* that push the decision towards
  "not acceptable", ordered by how much they push.
* **Artefact indicator** (provisional, 2 artefact studies in training): the
  scatter of nps_homo across slices (IQR / median) with its interval,
  compared with the good training studies: `above_good_range` (the value)
  and `flagged` (the whole interval). No probability.
* **Outlier slices**: noise above 3x the study median - descriptive only.
* **Plausibility**: septum minus falx medians of scan properties
  (sharpness, noise texture) compared with the range seen in training.
* **Metal rule** (`apply_metal_rule`, after the model): metal streaks that
  reach the brain within the verdict's slice range make the study Not
  assessable. Streaks disturb the measurements themselves (the noise ROI
  avoids them, so the study can look better than it is), and metal
  artefacts do not make a study unacceptable as such - with implants CT is
  often the only option; the reader decides. The model's own result is kept
  in `model_verdict`.

The model itself (`HeadCtModel`) is trained by
`application/train_classifier.py fit-study`; the shipped one is
`_head_ct_model.npz` next to this module, read by
`integration/model_files.py`. Pure NumPy/SciPy - no Qt, no file I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import expit

# The model's input features (core5; head_ct_classifier.md D6), in order.
CORE_FEATURES = ("mtf_area", "nps_homo", "nps_struct", "var_struct", "ratio_lf_hf")
# Per-slice values a study is evaluated from: core5 + noise texture (for the
# plausibility check). Keys of the mappings passed to `evaluate_study`.
SLICE_FEATURES = (*CORE_FEATURES, "nps_f_av", "nps_f_peak")
FEATURE_LABELS = {
    "mtf_area": "MTF area (sharpness)",
    "nps_homo": "Noise in homogeneous tissue (ROI2 NPS)",
    "nps_struct": "Noise at the falx (ROI1 NPS)",
    "var_struct": "Variance at the falx (ROI1)",
    "ratio_lf_hf": "PS/MTF ratio (LF/HF)",
    "nps_f_av": "Noise texture f_av",
    "nps_f_peak": "Noise texture f_peak",
}

ACCEPTABLE = "Acceptable"
NOT_ACCEPTABLE = "Not acceptable"
UNCERTAIN = "Uncertain"
NOT_ASSESSABLE = "Not assessable"  # metal rule only - the model never gives it

# Shown with every verdict of the shipped model (GUI and exports) - see
# docs/qa_method/head_ct_model_card.md. User-facing text never states how many
# studies the model was trained or tested on (user decision 2026-10-27).
PROVISIONAL_NOTE = (
    "Provisional model: trained on head-CT studies rated by a radiologist; not yet confirmed on independent "
    "data. Decision support only - see docs/qa_method/head_ct_model_card.md."
)

CONFIDENCE = 0.95
# Fewer falx slices than this: no verdict - a bootstrap over 1-4 slices
# cannot express how uncertain their median is.
MIN_SLICES = 5
# Fewer slices than this: no artefact indicator - an interquartile range
# needs enough slices to mean anything (training studies: ~25-35 slices).
MIN_SLICES_ARTEFACT = 10
ARTEFACT_FEATURE = "nps_homo"
OUTLIER_FACTOR = 3.0
_SEED = 2026


@dataclass(frozen=True)
class HeadCtModel:
    """A fitted study-level model. Weights act on the raw study medians
    (standardisation folded in): logit = intercept + weights . medians."""

    version: str
    feature_names: tuple[str, ...]
    weights: np.ndarray            # (n_features,)
    intercept: float
    boot_weights: np.ndarray       # (n_boot, n_features) - refits on resampled training studies
    boot_intercepts: np.ndarray    # (n_boot,)
    feature_mean: np.ndarray       # training mean of the study medians - reference for contributions
    good_low: np.ndarray           # min / max of the rated-good training studies' medians
    good_high: np.ndarray
    artefact_good_high: float      # largest nps_homo scatter among good training studies
    plausibility_names: tuple[str, ...]
    plausibility_low: np.ndarray   # min / max over training studies of (septum - falx) medians
    plausibility_high: np.ndarray

    _ARRAYS = (
        "weights", "boot_weights", "boot_intercepts", "feature_mean", "good_low", "good_high",
        "plausibility_low", "plausibility_high",
    )

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Plain arrays for `np.savez` (no pickles)."""
        out = {name: np.asarray(getattr(self, name), dtype=np.float64) for name in self._ARRAYS}
        out["version"] = np.array(self.version)
        out["feature_names"] = np.array(self.feature_names)
        out["intercept"] = np.float64(self.intercept)
        out["artefact_good_high"] = np.float64(self.artefact_good_high)
        out["plausibility_names"] = np.array(self.plausibility_names)
        return out

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> HeadCtModel:
        return cls(
            version=str(arrays["version"]),
            feature_names=tuple(str(n) for n in arrays["feature_names"]),
            intercept=float(arrays["intercept"]),
            artefact_good_high=float(arrays["artefact_good_high"]),
            plausibility_names=tuple(str(n) for n in arrays["plausibility_names"]),
            **{name: np.asarray(arrays[name], dtype=np.float64) for name in cls._ARRAYS},
        )


@dataclass(frozen=True)
class FeatureFinding:
    name: str
    value: float          # study median
    low: float            # 95 % interval of the median (slice bootstrap)
    high: float
    good_low: float       # range of the rated-good training studies
    good_high: float
    contribution: float   # to the logit, relative to the training mean; < 0 = towards not acceptable

    @property
    def outside(self) -> str | None:
        """'above' / 'below' when the whole interval is outside the good range."""
        if self.low > self.good_high:
            return "above"
        if self.high < self.good_low:
            return "below"
        return None


@dataclass(frozen=True)
class ArtefactIndicator:
    value: float          # IQR / median of nps_homo across slices
    low: float
    high: float
    good_high: float
    flagged: bool         # whole interval above every good training study

    @property
    def above_good_range(self) -> bool:
        """The value itself is above every good training study - weaker than
        `flagged`: with ~25 slices the interval is wide (training: the two
        artefact studies are above, but neither interval clears the range)."""
        return self.value > self.good_high


@dataclass(frozen=True)
class PlausibilityCheck:
    name: str
    difference: float     # septum median - falx median
    low: float            # range seen in training
    high: float

    @property
    def consistent(self) -> bool:
        return self.low <= self.difference <= self.high


@dataclass(frozen=True)
class HeadCtVerdict:
    verdict: str
    probability: float
    probability_low: float
    probability_high: float
    n_slices: int
    features: tuple[FeatureFinding, ...]
    reasons: tuple[FeatureFinding, ...]           # subset of features, strongest first
    artefact: ArtefactIndicator | None
    outlier_slices: tuple[int, ...]               # entries of `slice_ids`
    plausibility: tuple[PlausibilityCheck, ...]   # empty without septum slices
    slice_ids: tuple[int, ...]                    # the usable slices, in input order
    slice_probabilities: np.ndarray               # indicative only - the model is fitted on study medians
    model_version: str
    metal_slices: tuple[int, ...] = ()            # slices with metal streaks in the brain (metal rule)
    model_verdict: str | None = None              # the model's verdict when the metal rule overrode it

    @property
    def metal_rule(self) -> bool:
        return bool(self.metal_slices)


def _sigmoid(z):
    return expit(np.asarray(z, dtype=np.float64))


def probability(model: HeadCtModel, medians) -> float:
    """P(acceptable) of the main model for a study's feature medians."""
    return float(_sigmoid(model.intercept + np.asarray(medians, dtype=np.float64) @ model.weights))


def _feature_matrix(values: Mapping[str, Sequence[float]], names: Sequence[str]) -> np.ndarray:
    return np.column_stack([np.asarray(values[name], dtype=np.float64) for name in names])


def _scatter(x: np.ndarray, axis=-1) -> np.ndarray:
    q1, med, q3 = np.percentile(x, [25, 50, 75], axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (q3 - q1) / med


def evaluate_study(
    model: HeadCtModel,
    falx: Mapping[str, Sequence[float]],
    septum: Mapping[str, Sequence[float]] | None = None,
    slice_ids: Sequence[int] | None = None,
    *,
    seed: int = _SEED,
) -> HeadCtVerdict | None:
    """Verdict for one study from its per-slice falx values (`falx[name]` =
    one value per slice, names from `SLICE_FEATURES`). Slices with a
    non-finite model feature are ignored. None with fewer than `MIN_SLICES`
    usable slices. Deterministic for a given `seed`.
    """
    names = model.feature_names
    X = _feature_matrix(falx, names)
    ids = np.arange(len(X)) if slice_ids is None else np.asarray(slice_ids)
    usable = np.all(np.isfinite(X), axis=1)
    X, ids = X[usable], ids[usable]
    n = len(X)
    if n < MIN_SLICES:
        return None

    rng = np.random.default_rng(seed)
    n_boot = len(model.boot_intercepts)
    picks = rng.integers(0, n, size=(n_boot, n))       # one slice resample per model refit
    boot_medians = np.median(X[picks], axis=1)          # (n_boot, n_features)
    medians = np.median(X, axis=0)

    p = probability(model, medians)
    boot_p = _sigmoid(model.boot_intercepts + np.einsum("ij,ij->i", boot_medians, model.boot_weights))
    alpha = (1 - CONFIDENCE) / 2
    p_low, p_high = np.quantile(boot_p, [alpha, 1 - alpha])
    if p_low > 0.5:
        verdict = ACCEPTABLE
    elif p_high < 0.5:
        verdict = NOT_ACCEPTABLE
    else:
        verdict = UNCERTAIN

    lows, highs = np.quantile(boot_medians, [alpha, 1 - alpha], axis=0)
    contributions = model.weights * (medians - model.feature_mean)
    features = tuple(
        FeatureFinding(
            name, float(medians[i]), float(lows[i]), float(highs[i]),
            float(model.good_low[i]), float(model.good_high[i]), float(contributions[i]),
        )
        for i, name in enumerate(names)
    )
    reasons = tuple(sorted(
        (f for f in features if f.outside is not None and f.contribution < 0),
        key=lambda f: f.contribution,
    ))

    artefact = None
    noise = X[:, names.index(ARTEFACT_FEATURE)]
    if n >= MIN_SLICES_ARTEFACT:
        boot_scatter = _scatter(noise[picks], axis=1)
        s_low, s_high = np.quantile(boot_scatter, [alpha, 1 - alpha])
        artefact = ArtefactIndicator(
            float(_scatter(noise)), float(s_low), float(s_high), model.artefact_good_high,
            bool(s_low > model.artefact_good_high),
        )
    outliers = tuple(int(i) for i in ids[noise > OUTLIER_FACTOR * np.median(noise)])

    plausibility = ()
    if septum is not None:
        checks = []
        for k, name in enumerate(model.plausibility_names):
            s = np.asarray(septum.get(name, ()), dtype=np.float64)
            f = np.asarray(falx.get(name, ()), dtype=np.float64)
            s, f = s[np.isfinite(s)], f[np.isfinite(f)]
            if s.size and f.size:
                checks.append(PlausibilityCheck(
                    name, float(np.median(s) - np.median(f)),
                    float(model.plausibility_low[k]), float(model.plausibility_high[k]),
                ))
        plausibility = tuple(checks)

    return HeadCtVerdict(
        verdict=verdict,
        probability=p,
        probability_low=float(p_low),
        probability_high=float(p_high),
        n_slices=n,
        features=features,
        reasons=reasons,
        artefact=artefact,
        outlier_slices=outliers,
        plausibility=plausibility,
        slice_ids=tuple(int(i) for i in ids),
        slice_probabilities=_sigmoid(model.intercept + X @ model.weights),
        model_version=model.version,
    )


def slice_values(results_by_slice: Mapping[int, object]) -> tuple[dict[str, np.ndarray], list[int]]:
    """`SliceQualityResult`s ({slice index: result}) -> the per-slice values
    `evaluate_study` takes, plus the slice indices in the same (sorted) order."""
    ids = sorted(results_by_slice)
    results = [results_by_slice[i] for i in ids]
    # feature_vector = [mtf_area, nps_homo, nps_struct, var_struct, ratio_lf_hf, slope_avg, loc_max]
    values = {
        name: np.array([float(r.feature_vector[k]) for r in results], dtype=np.float64)
        for k, name in enumerate(CORE_FEATURES)
    }
    values["nps_f_av"] = np.array([float(r.nps_f_av) for r in results], dtype=np.float64)
    values["nps_f_peak"] = np.array([float(r.nps_f_peak) for r in results], dtype=np.float64)
    return values, ids


def evaluate_study_results(
    model: HeadCtModel | None,
    falx_results: Mapping[int, object],
    septum_results: Mapping[int, object] | None = None,
) -> HeadCtVerdict | None:
    """`evaluate_study` on `SliceQualityResult`s keyed by 0-based slice index
    (outlier and per-slice ids are then those indices). None without a model
    or with too few usable falx slices."""
    if model is None or not falx_results:
        return None
    falx, ids = slice_values(falx_results)
    septum = slice_values(septum_results)[0] if septum_results else None
    return evaluate_study(model, falx, septum, ids)


def apply_metal_rule(verdict: HeadCtVerdict | None, streak_slices: Sequence[int]) -> HeadCtVerdict | None:
    """Not assessable if metal streaks reach the brain (`streak_slices`, e.g.
    `ArtefactFlags.metal_streak_slices`) on a slice within the range of the
    verdict's slices - failed slices in between count (streaks can make the
    measurement fail)."""
    if verdict is None or not verdict.slice_ids:
        return verdict
    first, last = min(verdict.slice_ids), max(verdict.slice_ids)
    hit = tuple(sorted(index for index in set(streak_slices) if first <= index <= last))
    if not hit:
        return verdict
    return replace(verdict, verdict=NOT_ASSESSABLE, metal_slices=hit, model_verdict=verdict.verdict)
