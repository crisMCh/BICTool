"""The API from a study folder to the result, as a caller uses it: synthetic
DICOM files on disk, the real loader, ROI placement and measurement - only
the CNN is replaced (`fake_cnn`). Also `python -m bictool`. No patient data."""

import json

import pytest
from api_synthetic import write_synthetic_study

from bictool import api
from bictool.__main__ import main


def test_analyse_study_from_a_folder(tmp_path, fake_cnn):
    folder = write_synthetic_study(tmp_path / "study")

    result = api.analyse_study(folder, label="case 1", model="model")

    assert result.label == "case 1"
    assert result.series.slices == 3 and result.series.description == "Head 4.0 Hr40"
    assert result.series.series_instance_uid  # kept for integration ...
    assert "series_instance_uid" not in result.to_dict()["series"]  # ... but not exported by default
    assert sorted(result.measurements_for(api.SEPTUM)) == [0, 1, 2]
    m = result.measurements[0]
    assert m.slice.location_mm == 0.0 and m.slice.sop_instance_uid
    assert m.mtf_area > 0 and m.nps_homo >= 0
    assert result.segmentation_check is not None and result.artefacts.metal_slices == ()


def test_two_step_with_a_chosen_series(tmp_path, fake_cnn):
    folder = write_synthetic_study(tmp_path / "study")

    choices = api.list_series(folder)
    study = api.open_study(folder, series=choices[0].index)
    study.segment(model="model")
    rois = study.place_rois()
    rois.delete(1, api.roi_key(api.SEPTUM, "roi1"))
    result = study.measure(rois, reviewed=True)

    assert [c.slices for c in choices] == [3] and choices[0].axial
    assert sorted(result.measurements_for(api.SEPTUM)) == [0, 2]
    assert result.rois_reviewed and result.rois_edited


def test_folder_errors_are_series_errors(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(api.SeriesError, match="No DICOM"):
        api.open_study(tmp_path / "empty")
    with pytest.raises(api.SeriesError, match="does not exist"):
        api.open_study(write_synthetic_study(tmp_path / "study"), series=3)


def test_the_command_line(tmp_path, fake_cnn, capsys):
    folder = write_synthetic_study(tmp_path / "study")
    out = tmp_path / "result.json"

    assert main(["series", str(folder)]) == 0
    assert capsys.readouterr().out.startswith("0: CT")
    assert main(["analyse", str(folder), "--label", "case 1", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["label"] == "case 1" and len(data["measurements"]) == 3
    assert "sop_instance_uid" not in out.read_text(encoding="utf-8")
    assert main(["analyse", str(tmp_path / "nothing-here")]) == 2
    assert "error:" in capsys.readouterr().err
