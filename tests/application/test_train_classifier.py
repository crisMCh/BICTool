"""Tests `train_classifier.py` on synthetic data - no patient data."""

import csv

import numpy as np
import pytest

pytest.importorskip("sklearn")

import bictool.application.train_classifier as tc
from bictool.application.train_classifier import (
    Rated,
    cross_validate,
    export_model,
    load_slices,
    read_ratings,
    stratified_split,
    write_ratings,
)
from bictool.domain.results.classification import classify_quality
from bictool.integration.model_files import load_svm_model


def _ratings():
    return [
        Rated("A", "a_good1", "good", ""),
        Rated("A", "a_good2", "good", ""),
        Rated("A", "a_bad", "bad", ""),
        Rated("B", "b_good", "good", ""),
        Rated("B", "b_bad1", "bad", ""),
        Rated("B", "b_bad2", "bad_artefacts", ""),
        Rated("B", "b_bad3", "bad", ""),
    ]


def test_split_puts_one_volume_per_large_stratum_in_test_and_is_reproducible():
    first = stratified_split(_ratings(), seed=7)
    second = stratified_split(_ratings(), seed=7)

    assert [r.split for r in first] == [r.split for r in second]
    test = [r for r in first if r.split == "test"]
    # strata with >= 2 volumes: (A, good) and (B, not acceptable) -> one each
    assert sorted((r.dataset, r.label) for r in test) == [("A", 1), ("B", 0)]
    # single-volume strata stay in training
    assert {r.volume for r in first if r.split == "train"} >= {"a_bad", "b_good"}


def test_ratings_file_round_trip(tmp_path):
    path = tmp_path / "ratings.csv"
    write_ratings(path, stratified_split(_ratings(), seed=1))

    assert [r.volume for r in read_ratings(path)] == [r.volume for r in _ratings()]


def test_unknown_rating_is_rejected(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("Dataset,Volume,Rating,Split\nA,v,meh,train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown rating"):
        read_ratings(path)


def _slices_csv(tmp_path, rated, rng):
    path = tmp_path / "slices.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Volume", "Structure", "Slice", *tc.FEATURE_SETS["core5_texture"], "slope_avg", "loc_max"])
        for r in rated:
            for k in range(6):
                centre = 1.0 if r.label else -1.0  # separable classes
                writer.writerow([r.dataset, r.volume, "Dural Folds", k, *(centre + rng.normal(0, 0.3, 9))])
    return path


def test_exported_model_reproduces_sklearn_decisions_in_numpy(tmp_path):
    rng = np.random.default_rng(0)
    rated = [Rated(r.dataset, r.volume, r.rating, "train") for r in _ratings()]
    features = tc.FEATURE_SETS["core5"]
    X, y, _, _ = load_slices(_slices_csv(tmp_path, rated, rng), rated, "train", features)
    model = tc._make_model()
    model.fit(X, y)

    export_model(model, features, tmp_path / "model.npz")
    exported = load_svm_model(tmp_path / "model.npz")

    for x in X:
        label, decision = classify_quality(x, exported)
        assert decision == pytest.approx(model.decision_function([x])[0], abs=1e-9)
        assert label == model.predict([x])[0]
    assert list(exported["feature_names"]) == features


def test_cross_validation_holds_out_whole_patients():
    rng = np.random.default_rng(1)
    groups = np.repeat(["p1", "p2", "p3", "p4"], 5)
    y = np.repeat([1, 1, 0, 0], 5)
    X = (np.where(y == 1, 1.0, -1.0)[:, None] + rng.normal(0, 0.2, (20, 3)))
    seen = []

    class Spy:
        def __init__(self):
            self.inner = tc._make_model()

        def fit(self, X_train, y_train):
            seen.append(len(X_train))
            self.inner.fit(X_train, y_train)

        def predict(self, X_test):
            return self.inner.predict(X_test)

        def decision_function(self, X_test):
            return self.inner.decision_function(X_test)

    metrics = cross_validate(X, y, groups, make_model=Spy)

    assert seen == [15, 15, 15, 15]  # each fold leaves out one patient's 5 slices
    assert metrics["volumes"] == 4
    assert metrics["volume_accuracy"] == 1.0


# --- Study level -------------------------------------------------------------

def _study_ratings(split="train"):
    return [
        Rated("A", "a_good1", "good", split),
        Rated("A", "a_good2", "good", split),
        Rated("A", "a_bad", "bad", split),
        Rated("B", "b_good1", "good", split),
        Rated("B", "b_good2", "good", split),
        Rated("B", "b_bad1", "bad", split),
        Rated("B", "b_bad2", "bad_artefacts", split),
    ]


def _study_csv(tmp_path, rated, seed=0, n_falx=12, n_septum=4):
    """Per-slice rows as batch_qa writes them: quieter falx for good studies."""
    rng = np.random.default_rng(seed)
    path = tmp_path / "slices.csv"
    columns = ["mtf_area", "nps_homo", "nps_struct", "var_struct", "ratio_lf_hf"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Volume", "Structure", "Slice", *columns, "NPS f_av (1/mm)", "NPS f_peak (1/mm)"])
        for k, r in enumerate(rated):
            noise = (20.0 if r.label else 30.0) + k * 0.3
            for structure, n in (("Dural Folds", n_falx), ("Septum Pellucidum", n_septum)):
                for i in range(n):
                    writer.writerow([
                        r.dataset, r.volume, structure, 10 + i,
                        0.5 + rng.normal(0, 0.005), noise + rng.normal(0, 1), 100 + rng.normal(0, 5),
                        (80.0 if r.label else 140.0) + rng.normal(0, 5), 12 + rng.normal(0, 1),
                        0.2 + rng.normal(0, 0.01), 0.1,
                    ])
    return path


def test_study_slices_are_split_into_falx_and_septum(tmp_path):
    rated = _study_ratings()
    rated[0] = Rated("A", "a_good1", "good", "test")
    studies = tc.load_study_slices(_study_csv(tmp_path, rated, n_falx=7, n_septum=3), rated, "train")

    assert [s.rated.volume for s in studies] == [r.volume for r in rated[1:]]
    assert len(studies[0].falx["nps_homo"]) == 7 and len(studies[0].septum["nps_homo"]) == 3
    assert list(studies[0].falx_slices) == list(range(10, 17))
    assert studies[0].falx["nps_f_av"][0] == pytest.approx(0.2, abs=0.05)


def test_fitted_study_model_separates_and_stores_good_reference_ranges(tmp_path):
    from bictool.domain.results.head_ct_verdict import ACCEPTABLE, NOT_ACCEPTABLE, evaluate_study

    # Clearly separated classes and enough studies for confident verdicts
    # (with only a handful of studies the honest answer is "Uncertain").
    rated = [Rated("A" if k % 2 else "B", f"s{k}", "good" if k < 10 else "bad", "train") for k in range(20)]
    studies = tc.load_study_slices(_study_csv(tmp_path, rated), rated, "train")
    model = tc.fit_head_ct_model(studies, version="t", n_bootstrap=100)

    assert model.boot_weights.shape == (100, 5)
    medians = tc._study_medians(studies, model.feature_names)
    good = medians[[s.label == 1 for s in studies]]
    np.testing.assert_allclose(model.good_low, good.min(axis=0))
    np.testing.assert_allclose(model.good_high, good.max(axis=0))
    noisy = model.feature_names.index("var_struct")
    assert model.weights[noisy] < 0  # more variance at the falx -> less likely acceptable
    verdicts = [evaluate_study(model, s.falx, s.septum).verdict for s in studies]
    assert verdicts == [ACCEPTABLE if s.label else NOT_ACCEPTABLE for s in studies]


def test_study_model_round_trips_through_npz(tmp_path):
    rated = _study_ratings()
    studies = tc.load_study_slices(_study_csv(tmp_path, rated), rated, "train")
    model = tc.fit_head_ct_model(studies, version="t", n_bootstrap=20)

    from bictool.integration.model_files import load_head_ct_model, save_head_ct_model

    save_head_ct_model(model, tmp_path / "m.npz")
    again = load_head_ct_model(tmp_path / "m.npz")

    np.testing.assert_array_equal(again.boot_weights, model.boot_weights)
    assert again.version == "t"


def test_study_cross_validation_holds_out_each_study(tmp_path, monkeypatch):
    rated = _study_ratings()
    studies = tc.load_study_slices(_study_csv(tmp_path, rated), rated, "train")
    fitted_on = []
    fit = tc.fit_head_ct_model
    monkeypatch.setattr(
        tc, "fit_head_ct_model",
        lambda train, **kw: fitted_on.append({s.rated.volume for s in train}) or fit(train, **kw),
    )

    results = tc.cross_validate_studies(studies, n_bootstrap=30)

    assert [r["study"].rated.volume for r in results] == [r.volume for r in rated]
    assert all(r.volume not in seen for r, seen in zip(rated, fitted_on))
    assert all(len(seen) == len(rated) - 1 for seen in fitted_on)
    summary = tc.summarize_verdicts(results)
    assert summary["point_correct"] == len(rated)


def test_clopper_pearson_matches_known_values():
    assert tc.clopper_pearson(11, 11) == pytest.approx((0.7151, 1.0), abs=1e-4)
    assert tc.clopper_pearson(4, 4) == pytest.approx((0.3976, 1.0), abs=1e-4)
    assert tc.clopper_pearson(0, 10) == pytest.approx((0.0, 0.3085), abs=1e-4)
    low, high = tc.clopper_pearson(5, 10)
    assert low == pytest.approx(0.1871, abs=1e-4) and high == pytest.approx(0.8129, abs=1e-4)


def test_fit_study_and_one_time_test_commands(tmp_path):
    rated = _study_ratings() + [
        Rated("A", "x_good", "good", "test"), Rated("B", "x_bad", "bad", "test"),
    ]
    ratings = tmp_path / "ratings.csv"
    write_ratings(ratings, rated)
    slices = _study_csv(tmp_path, rated)
    model, card = tmp_path / "model.npz", tmp_path / "card.md"

    fit = ["fit-study", str(slices), str(ratings), "--out", str(model), "--card", str(card), "--bootstrap", "30"]
    tc.main([*fit, "--version", "t-1"])

    text = card.read_text(encoding="utf-8")
    assert "t-1" in text and "Not run yet" in text
    assert not any(r.volume in text for r in rated)  # studies appear only as scanner + rating

    tc.main(["test-study", str(slices), str(ratings), "--model", str(model), "--card", str(card)])

    text = card.read_text(encoding="utf-8")
    assert tc.TEST_MARKER in text and "Not run yet" not in text
    assert "| X1 | A | good |" in text and "| X2 | B | bad |" in text
    with pytest.raises(SystemExit, match="already run"):
        tc.main(["test-study", str(slices), str(ratings), "--model", str(model), "--card", str(card)])
    with pytest.raises(SystemExit, match="leak"):
        tc.main([*fit, "--version", "t-2"])
