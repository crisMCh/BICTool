"""Tests `batch_qa.py`'s orchestration with fakes - no DICOM data, no CNN."""

from types import SimpleNamespace

import numpy as np

import bictool.application.batch_qa as batch_qa
from bictool.application.batch_qa import find_studies, run_batch


def test_find_studies_looks_through_single_folder_wrappers(tmp_path):
    root = tmp_path / "CanonPrime" / "CanonPrime"
    (root / "study b").mkdir(parents=True)
    (root / "study a").mkdir()
    (root / "notes.txt").write_text("x")

    assert [study.name for study in find_studies(tmp_path / "CanonPrime")] == ["study a", "study b"]


def test_a_broken_study_is_recorded_and_the_batch_continues(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "good").mkdir(parents=True)
    (dataset / "broken").mkdir()

    def fake_load(path):
        if path.name == "broken":
            raise ValueError("no DICOM series found")
        return SimpleNamespace(
            num_slices=2,
            pixel_spacing_mm=0.5,
            acquisition_date="20260101",
            series_description="Head 4mm",
            slice_thickness_mm=4.0,
            slice_paths=[],
            get_slice_hu=lambda i: np.zeros((8, 8)),
        )

    monkeypatch.setattr(batch_qa, "load_dicom_volume", fake_load)
    monkeypatch.setattr(batch_qa, "_segment_volume", lambda volume, get_model: ({0: {"A": np.ones((8, 8), dtype=bool)}}, {}))
    monkeypatch.setattr(batch_qa, "evaluate_volume_qa", lambda *a, **k: {})

    volumes = run_batch([dataset], tmp_path / "out", weights_path="unused")

    by_name = {volume.name: volume for volume in volumes}
    assert by_name["broken"].error == "ValueError: no DICOM series found"
    assert by_name["good"].error is None
    assert by_name["good"].segmented_slices == 1
    assert (tmp_path / "out" / "dataset_qa.xlsx").is_file()
    assert (tmp_path / "out" / "dataset_qa_summary.csv").is_file()
    assert (tmp_path / "out" / "dataset_qa_slices.csv").is_file()


def test_excluded_studies_are_skipped(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "20260131_171926 - bad motion artefacts").mkdir(parents=True)
    (dataset / "20260131_164257 - good").mkdir()
    seen = []
    monkeypatch.setattr(batch_qa, "run_study", lambda study_dir, *a: seen.append(study_dir.name) or SimpleNamespace(
        error=None, name=study_dir.name, group="Scanner", structures={}))
    monkeypatch.setattr(batch_qa, "export_dataset_excel", lambda *a: None)
    monkeypatch.setattr(batch_qa, "export_dataset_csv", lambda *a: None)

    run_batch([dataset], tmp_path / "out", weights_path="unused", exclude=["bad motion artefacts"])

    assert seen == ["20260131_164257 - good"]


def test_recipe_option_reaches_the_volume_qa(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "study a").mkdir(parents=True)  # two studies - a lone subfolder counts as a wrapper
    (dataset / "study b").mkdir()
    seen = []
    monkeypatch.setattr(batch_qa, "run_study", lambda study_dir, group, get_model, cache, recipe, tissue: seen.append((recipe, tissue))
                        or SimpleNamespace(error=None, name=study_dir.name, group=group, structures={}))
    monkeypatch.setattr(batch_qa, "export_dataset_excel", lambda *a: None)
    monkeypatch.setattr(batch_qa, "export_dataset_csv", lambda *a: None)

    batch_qa.main([str(dataset), "--out", str(tmp_path / "out"), "--weights", "unused", "--recipe", "octave",
                   "--tissue", "anchored"])

    assert seen == [("octave", "anchored"), ("octave", "anchored")]


def test_physical_recipe_studies_get_the_head_ct_verdict(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "study a").mkdir(parents=True)
    (dataset / "study b").mkdir()
    monkeypatch.setattr(batch_qa, "load_dicom_volume", lambda path: SimpleNamespace(
        num_slices=2, pixel_spacing_mm=0.5, acquisition_date=None, series_description="Head",
        slice_thickness_mm=4.0, slice_paths=[], get_slice_hu=lambda i: np.zeros((8, 8))))
    monkeypatch.setattr(batch_qa, "_segment_volume", lambda volume, get_model: ({0: {"A": np.ones((8, 8), dtype=bool)}}, {}))
    falx = SimpleNamespace(results={0: "falx result"})
    monkeypatch.setattr(batch_qa, "evaluate_volume_qa", lambda *a, **k: {"Dural Folds": falx})
    calls = []
    monkeypatch.setattr(batch_qa, "evaluate_study_results", lambda model, f, s: calls.append((f, s)) or "verdict")
    monkeypatch.setattr(batch_qa, "export_dataset_excel", lambda *a: None)
    monkeypatch.setattr(batch_qa, "export_dataset_csv", lambda *a: None)

    physical = run_batch([dataset], tmp_path / "out", weights_path="unused", recipe="physical")
    legacy = run_batch([dataset], tmp_path / "out2", weights_path="unused", recipe="octave")

    assert [volume.head_ct_verdict for volume in physical] == ["verdict", "verdict"]
    assert calls[0] == ({0: "falx result"}, None)  # falx results; no septum in this fake
    assert [volume.head_ct_verdict for volume in legacy] == [None, None]
