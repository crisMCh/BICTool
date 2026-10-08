"""Training the image-quality classifier (FR-04) on rated head-CT volumes.

Input: the per-slice table written by `batch_qa` (`dataset_qa_slices.csv`)
and a ratings file with one row per volume:

    Dataset,Volume,Rating,Split
    Siemens ER,<study folder>,good,train
    ...

Rating is `good`, `bad` or `bad_artefacts` (the latter two form the "not
acceptable" class); Split is `train` or `test`. Ratings files name patient
studies - keep them out of version control (e.g. under debugs/).

Commands:

    split  RATINGS --seed N    add a stratified train/test split (one volume per
                                scanner x rating stratum with >= 2 volumes goes
                                to test) to a ratings file without a Split column
    cv     SLICES RATINGS      leave-one-patient-out cross-validation of the
                                candidate feature sets on the TRAIN volumes only
    fit    SLICES RATINGS --features NAME --out MODEL.npz
                                train on all TRAIN volumes and export the model
    test   SLICES RATINGS --model MODEL.npz
                                the one-time evaluation on the TEST volumes
    fit-study  SLICES RATINGS --out MODEL.npz --card CARD.md
                                the study-level head-CT model (the one the app
                                uses): fit on the TRAIN studies, patient-wise
                                cross-validation of the full verdict, model card
    test-study SLICES RATINGS --model MODEL.npz --card CARD.md
                                its one-time evaluation on the locked TEST
                                studies; refuses to run twice

Study level (fit-study / test-study; docs/qa_method/head_ct_classifier.md):
a logistic regression on the medians of the core5 features over each
study's falx slices, with 1000 bootstrap refits for the uncertainty, the
range of the rated-good studies, the artefact-indicator threshold and the
falx/septum plausibility ranges - evaluated by
`domain.results.head_ct_verdict`.

Slice level (cv / fit / test; the legacy design, kept for comparison):
the model is an RBF SVM on min-max scaled features (libsvm via scikit-learn),
exported in the format `domain.results.classification` evaluates with NumPy
(support vectors, dual coefficients, gamma, rho, labels, feature min/max) plus
the feature names, so the app needs no scikit-learn. Validation is always per
patient (all slices of a volume are held out together): slices of one patient
are not independent, and a per-slice split would overstate accuracy.

Requires the optional `train` extra (scikit-learn).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GOOD = "good"
NOT_ACCEPTABLE = ("bad", "bad_artefacts")
RATINGS = (GOOD, *NOT_ACCEPTABLE)

# Candidate feature sets -> columns of batch_qa's per-slice table.
FEATURE_SETS: dict[str, list[str]] = {
    # The original 7 (Passand 2022, Table 4.2; physical recipe values).
    "original7": ["mtf_area", "nps_homo", "nps_struct", "var_struct", "ratio_lf_hf", "slope_avg", "loc_max"],
    # Without slope and max location, which are ~constant on head CT
    # (docs/qa_method/physical_qa_recipe.md, Sec. 7 limitation 2).
    "core5": ["mtf_area", "nps_homo", "nps_struct", "var_struct", "ratio_lf_hf"],
    # core5 + noise texture of ROI2 (AAPM TG-233 f_A / f_P).
    "core5_texture": [
        "mtf_area", "nps_homo", "nps_struct", "var_struct", "ratio_lf_hf",
        "NPS f_av (1/mm)", "NPS f_peak (1/mm)",
    ],
}


@dataclass
class Rated:
    dataset: str
    volume: str
    rating: str
    split: str

    @property
    def label(self) -> int:
        """1 = acceptable (good), 0 = not acceptable."""
        return 1 if self.rating == GOOD else 0


def read_ratings(path: Path) -> list[Rated]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        rating = row["Rating"].strip().lower()
        if rating not in RATINGS:
            raise ValueError(f"Unknown rating {row['Rating']!r} for {row['Volume']} - expected one of {RATINGS}")
        out.append(Rated(row["Dataset"], row["Volume"], rating, row.get("Split", "").strip().lower()))
    return out


def write_ratings(path: Path, rated: list[Rated]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Volume", "Rating", "Split"])
        for r in rated:
            writer.writerow([r.dataset, r.volume, r.rating, r.split])


def stratified_split(rated: list[Rated], seed: int) -> list[Rated]:
    """One volume per (dataset, acceptable/not) stratum with >= 2 volumes goes
    to test, chosen at random with `seed`; everything else is train - so the
    test set spans scanners and both classes while every stratum keeps at
    least one training volume."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, int], list[Rated]] = defaultdict(list)
    for r in rated:
        strata[(r.dataset, r.label)].append(r)
    test = set()
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: r.volume)
        if len(members) >= 2:
            test.add(members[int(rng.integers(len(members)))].volume)
    return [Rated(r.dataset, r.volume, r.rating, "test" if r.volume in test else "train") for r in rated]


def load_slices(slices_csv: Path, rated: list[Rated], split: str, features: list[str]):
    """(X, y, groups, datasets) for the slices of the rated volumes in `split`.
    Slices with a non-finite feature are skipped."""
    by_volume = {(r.dataset, r.volume): r for r in rated if r.split == split}
    X, y, groups, datasets = [], [], [], []
    with open(slices_csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            r = by_volume.get((row["Dataset"], row["Volume"]))
            if r is None:
                continue
            values = [float(row[name]) for name in features]
            if not np.all(np.isfinite(values)):
                continue
            X.append(values)
            y.append(r.label)
            groups.append(r.volume)
            datasets.append(r.dataset)
    return np.array(X), np.array(y), np.array(groups), np.array(datasets)


def _make_model(C: float = 1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC

    return make_pipeline(MinMaxScaler(), SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced"))


def _volume_metrics(y, pred, groups):
    """Per-volume majority vote (the legacy recipe's study verdict), accuracy over volumes."""
    volumes = sorted(set(groups))
    correct = 0
    for v in volumes:
        mask = groups == v
        vote = 1 if pred[mask].mean() > 0.5 else 0  # a tie counts as not acceptable
        correct += int(vote == y[mask][0])
    return correct, len(volumes)


def cross_validate(X, y, groups, make_model=_make_model) -> dict:
    """Leave-one-patient-out: per-slice balanced accuracy and AUC, per-volume accuracy."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y))
    for volume in sorted(set(groups)):
        held = groups == volume
        model = make_model()
        model.fit(X[~held], y[~held])
        pred[held] = model.predict(X[held])
        score[held] = model.decision_function(X[held])
    correct, total = _volume_metrics(y, pred, groups)
    return {
        "slice_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "slice_auc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan"),
        "volume_accuracy": correct / total,
        "volumes_correct": correct,
        "volumes": total,
        "slices": int(len(y)),
    }


def scanner_only_baseline(y, groups, datasets) -> dict:
    """The same LOPO evaluation for a model that only knows the scanner -
    if the real model does not beat this, it is mostly recognising scanners."""
    names = sorted(set(datasets))
    X = np.array([[1.0 if d == n else 0.0 for n in names] for d in datasets])
    return cross_validate(X, y, groups)


def export_model(model, features: list[str], path: Path) -> dict:
    """Write the fitted pipeline in `domain.results.classification`'s format."""
    scaler, svc = model.named_steps["minmaxscaler"], model.named_steps["svc"]
    # sklearn: decision = sum(dual_coef * K) + intercept, > 0 -> classes_[1].
    # classification.py: decision = sum(sv_coef * K) - rho, >= 0 -> Label[0].
    params = {
        "SVs": svc.support_vectors_.astype(np.float64),
        "sv_coef": svc.dual_coef_[0].astype(np.float64),
        "gamma": np.float64(svc._gamma),
        "rho": np.float64(-svc.intercept_[0]),
        "Label": np.array([svc.classes_[1], svc.classes_[0]], dtype=np.int64),
        "feature_min": scaler.data_min_.astype(np.float64),
        "feature_max": scaler.data_max_.astype(np.float64),
        "feature_names": np.array(features),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)
    return {"support_vectors": int(len(svc.support_vectors_)), "gamma": float(svc._gamma)}


# --- Study level -----------------------------------------------------------

FALX = "Dural Folds"
SEPTUM = "Septum Pellucidum"
# head_ct_verdict feature name -> batch_qa column (where they differ).
_COLUMNS = {"nps_f_av": "NPS f_av (1/mm)", "nps_f_peak": "NPS f_peak (1/mm)"}
# Scan properties on which falx and septum agree (head_ct_classifier.md 5.3:
# Spearman rho 0.77 and 0.71) - the plausibility check.
PLAUSIBILITY_FEATURES = ("mtf_area", "nps_f_av")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 2026
TEST_MARKER = "<!-- one-time test recorded -->"


@dataclass
class StudySlices:
    rated: Rated
    falx: dict[str, np.ndarray]
    septum: dict[str, np.ndarray]
    falx_slices: np.ndarray

    @property
    def label(self) -> int:
        return self.rated.label


def load_study_slices(slices_csv: Path, rated: list[Rated], split: str) -> list[StudySlices]:
    """Per-slice values (`head_ct_verdict.SLICE_FEATURES`) of each rated
    study in `split`, separately for falx and septum, in ratings order."""
    from bictool.domain.results.head_ct_verdict import SLICE_FEATURES

    wanted = {(r.dataset, r.volume): r for r in rated if r.split == split}
    rows: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {FALX: [], SEPTUM: []})
    with open(slices_csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            key = (row["Dataset"], row["Volume"])
            if key in wanted and row["Structure"] in (FALX, SEPTUM):
                rows[key][row["Structure"]].append(row)

    def columns(slice_rows):
        return {
            name: np.array([float(r[_COLUMNS.get(name, name)]) for r in slice_rows], dtype=np.float64)
            for name in SLICE_FEATURES
        }

    out = []
    for key, r in wanted.items():
        structures = rows[key]
        out.append(StudySlices(
            r, columns(structures[FALX]), columns(structures[SEPTUM]),
            np.array([int(row["Slice"]) for row in structures[FALX]], dtype=np.int64),
        ))
    return out


def _study_medians(studies: list[StudySlices], names) -> np.ndarray:
    """(studies x features) medians over each study's falx slices that have
    all `names` finite - the same slices `evaluate_study` uses."""
    out = []
    for s in studies:
        X = np.column_stack([s.falx[name] for name in names]) if len(s.falx_slices) else np.empty((0, len(names)))
        X = X[np.all(np.isfinite(X), axis=1)]
        out.append(np.median(X, axis=0) if len(X) else np.full(len(names), np.nan))
    return np.array(out)


def _fit_logistic(M: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """L2 logistic regression (C = 1, class-balanced) on standardised
    medians; returns weights and intercept on the raw medians."""
    from sklearn.linear_model import LogisticRegression

    mean, sd = M.mean(axis=0), M.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    lr = LogisticRegression(C=1.0, class_weight="balanced").fit((M - mean) / sd, y)
    weights = lr.coef_[0] / sd  # classes_ = [0, 1] -> logit of "acceptable"
    return weights, float(lr.intercept_[0] - weights @ mean)


def fit_head_ct_model(
    studies: list[StudySlices], *, version: str, n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
):
    """Fit `head_ct_verdict.HeadCtModel` on rated studies (both classes needed).
    Studies with fewer than `MIN_SLICES` usable falx slices are left out, as
    the app gives them no verdict either."""
    from bictool.domain.results.head_ct_verdict import (
        ARTEFACT_FEATURE, CORE_FEATURES, MIN_SLICES, HeadCtModel, _scatter,
    )

    names = CORE_FEATURES
    usable = [
        np.sum(np.all(np.isfinite(np.column_stack([s.falx[n] for n in names])), axis=1)) >= MIN_SLICES
        if len(s.falx_slices) else False
        for s in studies
    ]
    studies = [s for s, ok in zip(studies, usable) if ok]
    M = _study_medians(studies, names)
    y = np.array([s.label for s in studies])
    if len(set(y)) < 2:
        raise ValueError("Both classes are needed to fit the study-level model")
    weights, intercept = _fit_logistic(M, y)

    # Bootstrap refits: resample studies with replacement within each class,
    # so every refit sees both classes (stratified bootstrap; Efron &
    # Tibshirani 1993, An Introduction to the Bootstrap).
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    boot_w, boot_b = np.empty((n_bootstrap, len(names))), np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.concatenate([rng.choice(members, size=len(members)) for members in classes])
        boot_w[b], boot_b[b] = _fit_logistic(M[idx], y[idx])

    def finite(x):
        return x[np.isfinite(x)]

    good_scatter = [float(_scatter(finite(s.falx[ARTEFACT_FEATURE]))) for s in studies if s.label == 1]
    diffs = np.array([
        [
            np.median(finite(s.septum[n])) - np.median(finite(s.falx[n]))
            if finite(s.septum[n]).size and finite(s.falx[n]).size else np.nan
            for n in PLAUSIBILITY_FEATURES
        ]
        for s in studies
    ])
    good = M[y == 1]
    return HeadCtModel(
        version=version,
        feature_names=tuple(names),
        weights=weights,
        intercept=intercept,
        boot_weights=boot_w,
        boot_intercepts=boot_b,
        feature_mean=M.mean(axis=0),
        good_low=good.min(axis=0),
        good_high=good.max(axis=0),
        artefact_good_high=max(good_scatter),
        plausibility_names=PLAUSIBILITY_FEATURES,
        plausibility_low=np.nanmin(diffs, axis=0),
        plausibility_high=np.nanmax(diffs, axis=0),
    )


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for k successes in n trials
    (Clopper & Pearson 1934)."""
    from scipy.stats import beta

    if n == 0:
        return float("nan"), float("nan")
    a = (1 - confidence) / 2
    low = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return low, high


def evaluate_studies(model, studies: list[StudySlices]) -> list[dict]:
    """The app's verdict (`head_ct_verdict.evaluate_study`) for each study."""
    from bictool.domain.results.head_ct_verdict import evaluate_study

    return [
        {"study": s, "verdict": evaluate_study(model, s.falx, s.septum, s.falx_slices)}
        for s in studies
    ]


def cross_validate_studies(studies: list[StudySlices], n_bootstrap: int = N_BOOTSTRAP) -> list[dict]:
    """Leave-one-patient-out: fit on all other studies (with its own
    bootstrap), evaluate the full verdict on the held-out one."""
    out = []
    for i, held in enumerate(studies):
        model = fit_head_ct_model(studies[:i] + studies[i + 1:], version="lopo", n_bootstrap=n_bootstrap)
        out.extend(evaluate_studies(model, [held]))
    return out


def summarize_verdicts(results: list[dict]) -> dict:
    """Point accuracy (p > 0.5), AUC, and the three-way verdicts, each count
    with an exact 95 % interval."""
    from bictool.domain.results.head_ct_verdict import ACCEPTABLE, NOT_ACCEPTABLE, UNCERTAIN

    scored = [r for r in results if r["verdict"] is not None]
    y = np.array([r["study"].label for r in scored])
    p = np.array([r["verdict"].probability for r in scored])
    point_correct = int(np.sum((p > 0.5) == (y == 1)))
    expected = {1: ACCEPTABLE, 0: NOT_ACCEPTABLE}
    confident = [r for r in scored if r["verdict"].verdict != UNCERTAIN]
    confident_correct = int(sum(r["verdict"].verdict == expected[r["study"].label] for r in confident))
    pos, neg = p[y == 1], p[y == 0]
    auc = (
        float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
        if len(pos) and len(neg) else float("nan")
    )
    return {
        "studies": len(results),
        "scored": len(scored),
        "point_correct": point_correct,
        "point_ci": clopper_pearson(point_correct, len(scored)),
        "auc": auc,
        "uncertain": len(scored) - len(confident),
        "confident": len(confident),
        "confident_correct": confident_correct,
        "confident_ci": clopper_pearson(confident_correct, len(confident)),
    }


def scanner_only_study_baseline(studies: list[StudySlices]) -> dict:
    """LOPO accuracy of a logistic regression that only knows the scanner -
    what the real model has to beat to be more than scanner recognition."""
    names = sorted({s.rated.dataset for s in studies})
    X = np.array([[1.0 if s.rated.dataset == n else 0.0 for n in names] for s in studies])
    y = np.array([s.label for s in studies])
    correct = 0
    for i in range(len(studies)):
        keep = np.arange(len(studies)) != i
        w, b = _fit_logistic(X[keep], y[keep])
        correct += int(((X[i] @ w + b) > 0) == (y[i] == 1))
    return {"correct": correct, "studies": len(studies), "ci": clopper_pearson(correct, len(studies))}


# --- Model card --------------------------------------------------------------

_TEST_PLACEHOLDER = "Not run yet. `train_classifier test-study` runs it once and fills in this section."


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def _ci(ci) -> str:
    return f"{_pct(ci[0])}-{_pct(ci[1])}"


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def _verdict_rows(results: list[dict], prefix: str) -> list[str]:
    from bictool.domain.results.head_ct_verdict import FEATURE_LABELS

    rows = [
        "| Study | Scanner | Rating | Falx slices | P(acceptable) [95 % CI] | Verdict | Outside good range (towards not acceptable) | Noise scatter [95 % CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, r in enumerate(results, 1):
        s, v = r["study"], r["verdict"]
        head = f"| {prefix}{k} | {s.rated.dataset} | {s.rated.rating.replace('_', ' ')} "
        if v is None:
            rows.append(head + f"| {len(s.falx_slices)} | - | no verdict (too few slices) | | |")
            continue
        reasons = ", ".join(FEATURE_LABELS[f.name] for f in v.reasons) or "-"
        art = "-" if v.artefact is None else f"{v.artefact.value:.2f} [{v.artefact.low:.2f}, {v.artefact.high:.2f}]"
        rows.append(
            head + f"| {v.n_slices} | {v.probability:.2f} [{v.probability_low:.2f}, {v.probability_high:.2f}] "
            f"| {v.verdict} | {reasons} | {art} |"
        )
    return rows


def _summary_rows(summary: dict) -> list[str]:
    return [
        "| Measure | Result | Exact 95 % CI |",
        "|---|---|---|",
        f"| Point estimate correct (P > 0.5 vs rating) | {summary['point_correct']} / {summary['scored']} "
        f"| {_ci(summary['point_ci'])} |",
        f"| AUC of the probabilities | {summary['auc']:.2f} | - |",
        f"| Verdict Uncertain | {summary['uncertain']} / {summary['scored']} | - |",
        f"| Confident verdicts (Acceptable / Not acceptable) correct | "
        f"{summary['confident_correct']} / {summary['confident']} | {_ci(summary['confident_ci'])} |",
    ]


def model_card(model, studies: list[StudySlices], cv_results: list[dict], baseline: dict, *, date: str) -> str:
    """The model card (Mitchell et al. 2019) as Markdown. Studies appear only
    as scanner + rating - never by folder name."""
    from bictool.domain.results.head_ct_verdict import (
        CONFIDENCE, FEATURE_LABELS, MIN_SLICES, MIN_SLICES_ARTEFACT, OUTLIER_FACTOR,
    )

    names = model.feature_names
    M = _study_medians(studies, names)
    sd = M.std(axis=0)
    summary = summarize_verdicts(cv_results)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in studies:
        counts[s.rated.dataset][s.rated.rating] += 1
    n_slices = [len(s.falx_slices) for s in studies]
    n_good = sum(s.label for s in studies)

    lines = [
        "# Model card: head-CT study-level quality model",
        "",
        f"Generated by `python -m bictool.application.train_classifier fit-study` on {date}. "
        "Do not edit by hand - rerun the command. Background, analysis and decisions: "
        "`head_ct_classifier.md`; feature measurement: `physical_qa_recipe.md`.",
        "",
        "## 1. Model details",
        "",
        f"* **Version:** {model.version}",
        "* **File:** `source/bictool/domain/results/_head_ct_model.npz`; evaluated by "
        "`domain/results/head_ct_verdict.py` (NumPy only).",
        "* **Type:** logistic regression (L2, C = 1, class-balanced) on standardised features.",
        f"* **Input:** for one study, the median over its falx (dural folds) slices of {len(names)} "
        "features from the physical QA recipe: " + ", ".join(f"{n} ({FEATURE_LABELS[n]})" for n in names) + ".",
        f"* **Output:** the probability that the study is rated acceptable, with a {100 * CONFIDENCE:.0f} % interval; "
        "verdict **Acceptable** (whole interval > 0.5), **Not acceptable** (whole interval < 0.5) or "
        "**Uncertain**; features outside the range of the good training studies; a provisional artefact "
        "indicator; outlier slices; a falx/septum plausibility check.",
        f"* **Uncertainty:** {len(model.boot_intercepts)} refits on bootstrap resamples of the training "
        f"studies (stratified by class, seed {BOOTSTRAP_SEED}), each paired with a resample of the "
        "study's slices; the interval is the 2.5-97.5 percentile of the resulting probabilities.",
        "",
        "## 2. Intended use",
        "",
        "* Decision support for image-quality QA of routine adult head CT, soft-tissue head kernels, "
        "reconstructed in axial ~4 mm slices, measured with the physical QA recipe in BICTool.",
        "* Not for: other body regions, sharp/bone kernels, the legacy (octave) recipe, single slices, "
        "or any diagnostic purpose. The verdict replaces no radiologist.",
        f"* No verdict with fewer than {MIN_SLICES} usable falx slices; no artefact indicator with fewer "
        f"than {MIN_SLICES_ARTEFACT}.",
        "",
        "## 3. Training data",
        "",
        f"{len(studies)} studies ({n_good} rated good, {len(studies) - n_good} not acceptable), one "
        "radiologist, three scanners. Falx slices per study: "
        f"{min(n_slices)}-{max(n_slices)}. Four further studies are locked for the one-time test (Section 7).",
        "",
        "| Scanner | good | bad | bad with artefacts |",
        "|---|---|---|---|",
    ]
    for dataset in sorted(counts):
        c = counts[dataset]
        lines.append(f"| {dataset} | {c['good']} | {c['bad']} | {c['bad_artefacts']} |")
    lines += [
        "",
        "Ratings are confounded with the scanner (`head_ct_classifier.md` 3.2).",
        "",
        "## 4. Parameters",
        "",
        "Coefficients per 1 SD of the training medians (positive = more likely acceptable), and the "
        "range of the good training studies used for the explanations:",
        "",
        "| Feature | Coefficient per SD | Odds ratio per SD | Good range (min-max) |",
        "|---|---|---|---|",
    ]
    for i, n in enumerate(names):
        beta = model.weights[i] * sd[i]
        lines.append(
            f"| {FEATURE_LABELS[n]} | {beta:+.2f} | {np.exp(beta):.2f} "
            f"| {_fmt(model.good_low[i])} - {_fmt(model.good_high[i])} |"
        )
    lines += [
        "",
        "The coefficients are regularised (C = 1) and the features are correlated; read them as the "
        "model's direction, not as independent effect sizes.",
        "",
        f"* **Artefact indicator:** IQR / median of nps_homo across falx slices; largest value among the "
        f"good training studies: {model.artefact_good_high:.2f}. Reported as *above the good range* "
        "(value) and *confirmed* (whole interval above). Provisional: 2 artefact studies.",
        f"* **Outlier slices:** nps_homo above {OUTLIER_FACTOR:g}x the study median (descriptive).",
        "* **Plausibility (septum - falx medians), range in training:** " + "; ".join(
            f"{n}: {_fmt(model.plausibility_low[k])} to {_fmt(model.plausibility_high[k])}"
            for k, n in enumerate(model.plausibility_names)
        ) + ".",
        "",
        "## 5. Patient-wise cross-validation (training studies)",
        "",
        "Leave one study out, fit the whole model (including its bootstrap) on the rest, run the app's "
        "verdict on the held-out study. Repeated for every study.",
        "",
        *_summary_rows(summary),
        "",
        f"Scanner-only baseline (logistic regression on the scanner name, same procedure): "
        f"{baseline['correct']} / {baseline['studies']} correct ({_ci(baseline['ci'])}).",
        "",
        *_verdict_rows(cv_results, "T"),
        "",
        f"Reading: {len(studies) - 1} training studies per fold. Where an interval contains 0.5 the "
        "verdict is *Uncertain*. Both sources of uncertainty contribute: the model refits (only "
        f"{len(studies) - n_good - 1}-{len(studies) - n_good} not-acceptable studies per fold) and the "
        "slice sampling. A held-out study can show a feature outside the good range even when "
        "acceptable, because the range is taken from the other studies only.",
        "",
        "## 6. Limitations",
        "",
        "* One rater; no inter-reader variability known; rating criteria not documented.",
        "* Small sample: exact intervals above are wide, and the design was chosen after looking at the "
        "same training studies (`head_ct_classifier.md` 5.5) - cross-validation is optimistic.",
        "* Scanner-rating confounding; only one good Canon study.",
        "* Only soft-tissue head kernels; scope of the physical recipe.",
        "* The artefact indicator rests on 2 artefact studies; the test set has none.",
        "* Dose was not used (decision D2).",
        "",
        "## 7. One-time test (locked test studies)",
        "",
        _TEST_PLACEHOLDER,
        "",
    ]
    return "\n".join(lines)


def one_time_test_section(results: list[dict], *, model_version: str, date: str) -> str:
    summary = summarize_verdicts(results)
    return "\n".join([
        TEST_MARKER,
        f"Run once on {date} with model {model_version}. The test studies were not used for any "
        "design decision or fit. Reported as obtained.",
        "",
        *_summary_rows(summary),
        "",
        *_verdict_rows(results, "X"),
    ])


def _print_metrics(title: str, metrics: dict) -> None:
    print(
        f"{title:28s} slice balanced acc {metrics['slice_balanced_accuracy']:.2f}  "
        f"slice AUC {metrics['slice_auc']:.2f}  volumes {metrics['volumes_correct']}/{metrics['volumes']} "
        f"({metrics['slices']} slices)"
    )


def _card_has_test(card: Path) -> bool:
    return card.is_file() and TEST_MARKER in card.read_text(encoding="utf-8")


def fit_study_command(
    slices: Path, rated: list[Rated], out: Path, card: Path, version: str, n_bootstrap: int = N_BOOTSTRAP,
) -> None:
    """Fit the study-level model on the TRAIN studies, cross-validate it and
    write the model and its card. Refuses to overwrite a card that already
    holds the one-time test: refitting after seeing the test would leak it."""
    import datetime

    if _card_has_test(card):
        raise SystemExit(f"{card} already holds the one-time test result - refitting would leak the test set")
    studies = load_study_slices(slices, rated, "train")
    from bictool.integration.model_files import save_head_ct_model

    model = fit_head_ct_model(studies, version=version, n_bootstrap=n_bootstrap)
    save_head_ct_model(model, out)
    cv = cross_validate_studies(studies, n_bootstrap=n_bootstrap)
    baseline = scanner_only_study_baseline(studies)
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(model_card(model, studies, cv, baseline, date=datetime.date.today().isoformat()), encoding="utf-8")
    s = summarize_verdicts(cv)
    print(
        f"Model {version}: {len(studies)} training studies -> {out}\n"
        f"LOPO: point {s['point_correct']}/{s['scored']}, AUC {s['auc']:.2f}, uncertain {s['uncertain']}, "
        f"confident correct {s['confident_correct']}/{s['confident']}; scanner-only {baseline['correct']}/"
        f"{baseline['studies']}. Card: {card}"
    )


def test_study_command(slices: Path, rated: list[Rated], model_path: Path, card: Path) -> None:
    """The one-time test: evaluate the locked TEST studies and record the
    result in the model card. Runs only once per card."""
    import datetime

    if not card.is_file() or _TEST_PLACEHOLDER not in card.read_text(encoding="utf-8"):
        if _card_has_test(card):
            raise SystemExit(f"The one-time test was already run - see {card}")
        raise SystemExit(f"{card} is not a model card from fit-study")
    from bictool.integration.model_files import load_head_ct_model

    model = load_head_ct_model(model_path)
    results = evaluate_studies(model, load_study_slices(slices, rated, "test"))
    section = one_time_test_section(results, model_version=model.version, date=datetime.date.today().isoformat())
    card.write_text(card.read_text(encoding="utf-8").replace(_TEST_PLACEHOLDER, section), encoding="utf-8")
    print(section)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split")
    p.add_argument("ratings", type=Path)
    p.add_argument("--seed", type=int, required=True)
    for name in ("cv", "fit", "test"):
        p = sub.add_parser(name)
        p.add_argument("slices", type=Path)
        p.add_argument("ratings", type=Path)
        if name in ("fit",):
            p.add_argument("--features", choices=FEATURE_SETS, required=True)
            p.add_argument("--out", type=Path, required=True)
        if name == "test":
            p.add_argument("--model", type=Path, required=True)
    for name in ("fit-study", "test-study"):
        p = sub.add_parser(name)
        p.add_argument("slices", type=Path)
        p.add_argument("ratings", type=Path)
        p.add_argument("--card", type=Path, required=True, help="model card (Markdown)")
        if name == "fit-study":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--version", required=True, help="e.g. 'head-ct-1'")
            p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help=argparse.SUPPRESS)
        else:
            p.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "split":
        rated = stratified_split(read_ratings(args.ratings), args.seed)
        write_ratings(args.ratings, rated)
        for r in rated:
            print(f"{r.split:5s} {r.rating:13s} {r.dataset} / {r.volume}")
        return

    rated = read_ratings(args.ratings)
    if any(r.split not in ("train", "test") for r in rated):
        raise SystemExit("Ratings file has no complete Split column - run `split` first")

    if args.command == "cv":
        results = {}
        for name, features in FEATURE_SETS.items():
            X, y, groups, datasets = load_slices(args.slices, rated, "train", features)
            results[name] = cross_validate(X, y, groups)
            _print_metrics(name, results[name])
        _, y, groups, datasets = load_slices(args.slices, rated, "train", FEATURE_SETS["core5"])
        results["scanner_only_baseline"] = scanner_only_baseline(y, groups, datasets)
        _print_metrics("scanner-only baseline", results["scanner_only_baseline"])
        print(json.dumps(results, indent=2))
    elif args.command == "fit":
        features = FEATURE_SETS[args.features]
        X, y, groups, _ = load_slices(args.slices, rated, "train", features)
        model = _make_model()
        model.fit(X, y)
        info = export_model(model, features, args.out)
        print(f"Trained on {len(set(groups))} volumes / {len(y)} slices, features {args.features}: {info}")
    elif args.command == "fit-study":
        fit_study_command(args.slices, rated, args.out, args.card, args.version, args.bootstrap)
    elif args.command == "test-study":
        test_study_command(args.slices, rated, args.model, args.card)
    elif args.command == "test":
        from bictool.domain.results.classification import classify_quality
        from bictool.integration.model_files import load_svm_model

        model = load_svm_model(args.model)
        features = [str(n) for n in model["feature_names"]]
        X, y, groups, _ = load_slices(args.slices, rated, "test", features)
        pred = np.array([classify_quality(x, model)[0] for x in X])
        correct, total = _volume_metrics(y, pred, groups)
        from sklearn.metrics import balanced_accuracy_score

        print(
            f"TEST: slice balanced accuracy {balanced_accuracy_score(y, pred):.2f}, "
            f"volumes {correct}/{total} ({len(y)} slices)"
        )
        for volume in sorted(set(groups)):
            mask = groups == volume
            print(f"  {volume}: rated {'good' if y[mask][0] else 'not acceptable'}, "
                  f"{pred[mask].mean():.0%} of slices predicted acceptable")


if __name__ == "__main__":
    main()
