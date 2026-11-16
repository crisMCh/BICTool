"""Tests `batch_qa.py`'s orchestration with fakes - no DICOM data, no CNN."""

from types import SimpleNamespace

import numpy as np

import bictool.application.pipeline as steps
import bictool.application.batch_qa as batch_qa
from bictool.api.results import StudyResult
from bictool.application.qa_settings import METHOD_ANCHORED
from bictool.api.study import Study
from bictool.application.batch_qa import find_studies, run_batch


def test_find_studies_looks_through_single_folder_wrappers(tmp_path):
    root = tmp_path / "CanonPrime" / "CanonPrime"
    (root / "study b").mkdir(parents=True)
    (root / "study a").mkdir()
    (root / "notes.txt").write_text("x")

    assert [study.name for study in find_studies(tmp_path / "CanonPrime")] == ["study a", "study b"]


def _fake_volume(path):
    return SimpleNamespace(
        num_slices=2,
        pixel_spacing_mm=0.5,
        acquisition_date="20000101",
        series_description="Head 4mm",
        slice_thickness_mm=4.0,
        slice_paths=[],
        get_slice_hu=lambda i: np.zeros((8, 8)),
    )


def _fake_pipeline(monkeypatch, seen=None):
    """No DICOM, no CNN: a fake series, one segmented slice, nothing measured."""

    def segment(study, model=None, *, cache_file=None, **kwargs):
        study.set_segmentation({0: {"A": np.ones((8, 8), dtype=bool)}})
        return 1

    def measure(study, *a, **k):
        if seen is not None:
            seen.append(study.settings)
        return StudyResult(label=study.label, series=study.series, settings=study.settings, rois_reviewed=False, rois_edited=False,
                           measurements=(), segmented_slices=len(study.masks_by_slice))

    monkeypatch.setattr(steps, "load_anatomy_model", lambda weights_path=None: "model")
    monkeypatch.setattr(Study, "segment", segment)
    monkeypatch.setattr(Study, "measure", measure)


def test_a_broken_study_is_recorded_and_the_batch_continues(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "good").mkdir(parents=True)
    (dataset / "broken").mkdir()

    def fake_open(path):
        if path.name == "broken":
            raise ValueError("no DICOM series found")
        return _fake_volume(path)

    monkeypatch.setattr(steps, "open_series", fake_open)
    _fake_pipeline(monkeypatch)

    volumes = run_batch([dataset], tmp_path / "out", weights_path="unused")

    by_name = {volume.name: volume for volume in volumes}
    assert by_name["broken"].error == "ValueError: no DICOM series found"
    assert by_name["good"].error is None
    assert by_name["good"].segmented_slices == 1
    assert by_name["good"].series_label == "Head 4mm (2 slices, 4.0 mm)"
    assert (tmp_path / "out" / "dataset_qa.xlsx").is_file()
    assert (tmp_path / "out" / "dataset_qa_summary.csv").is_file()
    assert (tmp_path / "out" / "dataset_qa_slices.csv").is_file()


def test_excluded_studies_are_skipped(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "study 2 - bad motion artefacts").mkdir(parents=True)
    (dataset / "study 1 - good").mkdir()
    seen = []
    monkeypatch.setattr(batch_qa, "run_study", lambda study_dir, *a: seen.append(study_dir.name) or SimpleNamespace(
        error=None, name=study_dir.name, group="Scanner", structures={}))
    monkeypatch.setattr(batch_qa, "export_dataset_excel", lambda *a: None)
    monkeypatch.setattr(batch_qa, "export_dataset_csv", lambda *a: None)

    run_batch([dataset], tmp_path / "out", weights_path="unused", exclude=["bad motion artefacts"])

    assert seen == ["study 1 - good"]


def test_recipe_option_reaches_the_volume_qa(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "study a").mkdir(parents=True)  # two studies - a lone subfolder counts as a wrapper
    (dataset / "study b").mkdir()
    seen = []
    monkeypatch.setattr(batch_qa, "run_study", lambda study_dir, group, get_model, cache, recipe, tissue, exports, export_dir:
                        seen.append((recipe, tissue, exports, export_dir.name))
                        or SimpleNamespace(error=None, name=study_dir.name, group=group, structures={}))
    monkeypatch.setattr(batch_qa, "export_dataset_excel", lambda *a: None)
    monkeypatch.setattr(batch_qa, "export_dataset_csv", lambda *a: None)

    batch_qa.main([str(dataset), "--out", str(tmp_path / "out"), "--weights", "unused", "--recipe", "octave",
                   "--tissue", "anchored", "--plugin-export", "my-plugin.json"])

    assert seen == [("octave", "anchored", ("my-plugin.json",), "plugin_exports")] * 2


def test_the_recipe_and_wmgm_method_reach_the_study(monkeypatch, tmp_path):
    dataset = tmp_path / "Scanner"
    (dataset / "study a").mkdir(parents=True)
    (dataset / "study b").mkdir()
    monkeypatch.setattr(steps, "open_series", _fake_volume)
    seen = []
    _fake_pipeline(monkeypatch, seen)
    monkeypatch.setattr(batch_qa, "export_dataset_excel", lambda *a: None)
    monkeypatch.setattr(batch_qa, "export_dataset_csv", lambda *a: None)

    run_batch([dataset], tmp_path / "out", weights_path="unused", recipe="octave", tissue_method="anchored")

    assert [(s.recipe, s.tissue.method) for s in seen] == [("octave", METHOD_ANCHORED)] * 2
