"""Tests `api.Study` (ROIs, QA, study summary) and the result layout on
synthetic label maps and images - no patient data, no CNN."""

import json

import numpy as np
import pytest
from api_synthetic import fake_quality, fake_volume, septum_slice

import bictool.application.pipeline as pipeline
from bictool import api
from bictool.domain.roi_selection.structured_roi import SEPTUM_PELLUCIDUM

S1, S2 = api.roi_key(api.SEPTUM, "roi1"), api.roi_key(api.SEPTUM, "roi2")


def _study(masks_by_slice, num_slices=3, settings=None) -> api.Study:
    study = api.Study(fake_volume(masks_by_slice, num_slices), settings=settings)
    study.set_segmentation(masks_by_slice)
    return study


# ---- placing and measuring ----


def test_every_segmented_slice_gets_rois_and_qa(fake_measurement):
    result = _study({0: septum_slice(), 2: septum_slice()}).measure(head_ct_model=None)

    septum = result.measurements_for(api.SEPTUM)
    assert sorted(septum) == [0, 2]
    m = septum[0]
    assert m.roi1.width == m.roi2.width and m.roi2_region == api.VENTRICLE
    assert (m.mtf_area, m.nps_homo, m.nps_struct, m.var_struct, m.ratio_lf_hf) == (0.5, 2.0, 3.0, 4.0, 1.5)
    assert m.slice == api.SliceRef(0, 0.0, "1.2.3.0")
    (img1, _img2, _spacing), kwargs = fake_measurement[0]
    assert img1.shape == (m.roi1.width, m.roi1.width)
    assert kwargs["mtf_seeds"] and kwargs["recipe"] == "physical" and kwargs["mtf_structure"].any()
    assert result.measurements_for(api.FALX) == {}
    assert result.rois_reviewed is False and result.rois_edited is False and result.segmented_slices == 2


def test_a_failing_slice_is_recorded_and_the_rest_continue(monkeypatch):
    def evaluate(*a, **k):
        if not evaluate.failed:
            evaluate.failed = True
            raise ValueError("No edge columns survived bump removal")
        return fake_quality()

    evaluate.failed = False
    monkeypatch.setattr(pipeline, "evaluate_slice_quality", evaluate)

    result = _study({0: septum_slice(), 1: septum_slice()}, num_slices=2).measure(head_ct_model=None)

    assert result.failed == (api.Failure(api.SliceRef(0, 0.0, "1.2.3.0"), api.SEPTUM, "No edge columns survived bump removal"),)
    assert list(result.measurements_for(api.SEPTUM)) == [1]


def test_a_programming_error_is_raised_not_recorded(monkeypatch):
    monkeypatch.setattr(pipeline, "evaluate_slice_quality", lambda *a, **k: None.mtf)

    with pytest.raises(AttributeError):
        _study({0: septum_slice()}, num_slices=1).measure(head_ct_model=None)


def test_slices_with_too_little_brain_get_no_automatic_rois(fake_measurement):
    # A septum between small ventricles and no parenchyma around: the structure
    # is found, but the brain is ~8 % of slice 0's (skull base / vertex-like).
    small = {name: np.zeros(mask.shape, dtype=bool) for name, mask in septum_slice().items()}
    small["Ventricular System"][60:100, 60:100] = True
    small[SEPTUM_PELLUCIDUM][62:98, 79:81] = True
    small["Ventricular System"] &= ~small[SEPTUM_PELLUCIDUM]

    study = _study({0: septum_slice(), 1: small}, num_slices=2)
    rois = study.place_rois()
    result = study.measure(rois, head_ct_model=None)

    assert rois.slices() == [0]
    assert sorted(result.measurements_for(api.SEPTUM)) == [0]
    assert result.skipped[api.SEPTUM] == (1,)  # the septum was found there, but the slice has too little brain


def test_edited_rois_are_measured_and_marked(fake_measurement):
    study = _study({0: septum_slice(), 1: septum_slice()}, num_slices=2)
    rois = study.place_rois()
    roi1 = rois.effective(0)[S1]
    manual = api.RoiRect(roi1.x0 - 30, roi1.y0, roi1.x1 - 30, roi1.y1)
    rois.set_manual(0, S2, manual)
    rois.delete(1, S1)

    result = study.measure(rois, reviewed=True, head_ct_model=None)

    m = result.measurements[0]
    assert [m.slice.index for m in result.measurements] == [0]
    assert m.roi2 == manual and m.roi2_manual and m.roi2_region is None
    assert result.rois_reviewed is True and result.rois_edited is True


def test_a_manual_roi1_gets_a_new_automatic_roi2_as_in_the_program(fake_measurement):
    study = _study({0: septum_slice()}, num_slices=1)
    rois = study.place_rois()
    roi1, roi2 = rois.effective(0)[S1], rois.effective(0)[S2]
    moved = api.RoiRect(roi1.x0, roi1.y0 + 20, roi1.x1, roi1.y1 + 20)
    rois.set_manual(0, S1, moved)

    result = study.measure(rois, head_ct_model=None)

    m = result.measurements[0]
    assert m.roi1 == moved and m.roi1_manual
    assert m.roi2 != roi2 and not m.roi2_manual and m.roi2_region == api.VENTRICLE


@pytest.mark.parametrize(
    ("rect", "message"),
    [
        (lambda r: api.RoiRect(r.x0, r.y0, r.x1 + 3, r.y1), "square"),
        (lambda r: api.RoiRect(-4, r.y0, r.width - 4, r.y1), "outside"),
        (lambda r: api.RoiRect(r.x1, r.y0, r.x0, r.y1), "x0 < x1"),
        (lambda r: api.RoiRect(r.x0 + 0.5, r.y0, r.x1 + 0.5, r.y1), "whole pixels"),
    ],
)
def test_rois_that_cannot_be_measured_are_refused_before_measuring(fake_measurement, rect, message):
    study = _study({0: septum_slice()}, num_slices=1)
    rois = study.place_rois()
    rois.set_manual(0, S2, rect(rois.effective(0)[S2]))

    with pytest.raises(api.InvalidRois, match=message):
        study.measure(rois, head_ct_model=None)
    assert fake_measurement == []


def test_unknown_roi_slots_and_rois_of_another_series_are_refused(fake_measurement):
    study = _study({0: septum_slice()}, num_slices=1)
    rois = study.place_rois()
    with pytest.raises(ValueError, match="Unknown ROI slot"):
        rois.set_manual(0, "falx.roi3", api.RoiRect(0, 0, 10, 10))

    other = api.Study(fake_volume({0: septum_slice()}, num_slices=2))
    other.set_segmentation({0: septum_slice()})
    with pytest.raises(api.InvalidRois, match="another series"):
        other.measure(rois, head_ct_model=None)


def test_the_verdict_uses_the_falx_results_and_the_metal_rule(monkeypatch, fake_measurement):
    calls = []
    monkeypatch.setattr(pipeline, "evaluate_study_results", lambda model, falx, septum: calls.append((model, falx, septum)))
    rules = []
    monkeypatch.setattr(pipeline, "apply_metal_rule", lambda verdict, streaks: rules.append(tuple(streaks)) or verdict)
    model = object()

    physical = _study({0: septum_slice()}, num_slices=1).measure(head_ct_model=model)
    legacy = _study({0: septum_slice()}, num_slices=1, settings=api.Settings(recipe=api.RECIPE_OCTAVE)).measure()

    assert calls[0][0] is model and calls[0][1] == {} and sorted(calls[0][2]) == [0]
    assert rules == [()] and physical.verdict is None  # the fake model gives no verdict
    assert legacy.verdict is None and len(calls) == 1


def test_needs_a_segmentation_that_fits_the_series():
    study = api.Study(fake_volume({}, num_slices=1))
    with pytest.raises(api.NotSegmented):
        study.place_rois()
    with pytest.raises(api.InvalidSegmentation, match="does not exist"):
        study.set_segmentation({5: septum_slice()})
    with pytest.raises(api.InvalidSegmentation, match="do not match"):
        study.set_segmentation({0: {"Brain": np.ones((10, 10), dtype=bool)}})


# ---- segmentation: cache, cancel, resume ----


def test_segment_reads_and_writes_the_cache_of_its_own_series_only(fake_cnn, tmp_path):
    cache = tmp_path / "seg.npz"
    first = api.Study(fake_volume({0: septum_slice(), 1: septum_slice()}, num_slices=2))
    progress = []
    assert first.segment(model="model", cache_file=cache, progress=lambda *a: progress.append(a)) == 2
    assert progress == [("segmentation", 1, 2), ("segmentation", 2, 2)]

    fake_cnn.clear()
    second = api.Study(fake_volume({0: septum_slice(), 1: septum_slice()}, num_slices=2))
    assert second.segment(cache_file=cache) == 2 and fake_cnn == []

    other = api.Study(fake_volume({0: septum_slice()}, num_slices=3))
    with pytest.raises(api.InvalidSegmentation, match="another series"):
        other.segment(cache_file=cache)


def test_a_cancelled_segmentation_keeps_its_slices_and_resumes(fake_cnn):
    study = api.Study(fake_volume({i: septum_slice() for i in range(3)}, num_slices=3))
    calls = iter([False, True])

    with pytest.raises(api.Cancelled):
        study.segment(model="model", cancel=lambda: next(calls))
    assert sorted(study.masks_by_slice) == [0]

    assert study.segment(model="model") == 3
    assert fake_cnn == [0, 1, 2]  # slice 0 was not segmented twice


def test_measuring_can_be_cancelled(fake_measurement):
    with pytest.raises(api.Cancelled):
        _study({0: septum_slice()}, num_slices=1).measure(cancel=lambda: True, head_ct_model=None)


def test_without_weights_the_model_is_not_available(monkeypatch):
    import bictool.infrastructure.config as config

    monkeypatch.setattr(config, "load_config", lambda: {})
    with pytest.raises(api.ModelNotAvailable):
        api.load_anatomy_model()


# ---- the result layout ----

_TOP_KEYS = {
    "label", "series", "settings", "rois_reviewed", "rois_edited", "measurements", "failed", "skipped",
    "segmented_slices", "verdict", "artefacts", "basal_ganglia", "segmentation_check", "api_version",
    "schema_version", "units", "plugins", "plugin_values",
}
_MEASUREMENT_KEYS = {
    "slice", "pair", "roi1", "roi2", "roi1_manual", "roi2_manual", "roi2_region", "recipe", "mtf_area", "nps_homo",
    "nps_struct", "var_struct", "ratio_lf_hf", "nps_f_av", "nps_f_peak", "contrast_roi1", "contrast_roi2", "curves",
    "plugin_values",
}
_SERIES_KEYS = {
    "description", "slices", "slice_thickness_mm", "pixel_spacing_mm", "manufacturer_model", "reconstruction",
    "kvp", "tube_current_ma", "ctdi_vol_mgy", "dlp_mgy_cm", "acquisition_date",
}


def test_the_result_layout_is_pinned(fake_measurement):
    """Schema 1.x: changing these keys needs a SCHEMA_VERSION bump and a docs/api.md entry."""
    data = _study({0: septum_slice()}, num_slices=1).measure(head_ct_model=None).to_dict()

    assert api.SCHEMA_VERSION == "1.1"
    assert set(data) == _TOP_KEYS
    assert set(data["measurements"][0]) == _MEASUREMENT_KEYS
    assert set(data["series"]) == _SERIES_KEYS
    assert set(data["measurements"][0]["slice"]) == {"index", "location_mm"}
    assert set(data["settings"]) == {"recipe", "tissue_method", "tissue_parameters"}
    assert set(data["segmentation_check"]) == {"fragmentation", "slices", "worst_slices", "lobe_islands", "unreliable"}
    assert set(data["artefacts"]) == {
        "metal_slices", "metal_max_hu", "metal_streak_slices", "metal_streak_share", "brainstem_band_slices",
        "brainstem_band_depth_hu",
    }


def test_the_result_is_json_ready_without_uids_or_folder_name(fake_measurement):
    result = _study({0: septum_slice()}, num_slices=1).measure(head_ct_model=None, curves=True)
    data = json.loads(result.to_json())

    assert data["label"] == "study - Head 4.0 Hr40"
    assert data["settings"]["tissue_method"] == "adaptive"
    m = data["measurements"][0]
    assert m["pair"] == "septum" and m["roi2_region"] == "ventricle" and m["nps_f_peak"] is None
    assert m["curves"] is None and data["units"]["mtf_area"] == "1/mm"
    assert "sop_instance_uid" not in json.dumps(data) and "series_instance_uid" not in json.dumps(data)

    full = result.to_dict(include_curves=True, include_dicom_uids=True)
    assert full["measurements"][0]["slice"]["sop_instance_uid"] == "1.2.3.0"
    assert full["measurements"][0]["curves"]["mtf_per_edge"] == [[1.0] * 4, [1.0] * 4]


def test_a_result_reads_back_from_json(fake_measurement):
    settings = api.Settings(tissue=api.TissueSettings(method=api.METHOD_FCM))
    result = _study({0: septum_slice()}, num_slices=1, settings=settings).measure(head_ct_model=None, curves=True)

    back = api.StudyResult.from_json(result.to_json(include_curves=True, include_dicom_uids=True))

    assert back.to_dict(True, True) == result.to_dict(True, True)  # (NaN != NaN, so not back == result)
    assert back.measurements[0].roi1 == result.measurements[0].roi1 and back.measurements[0].slice.index == 0
    assert back.settings.tissue.method == api.METHOD_FCM
    with pytest.raises(ValueError, match="no internal measurements"):
        back.to_volume_qa()
    with pytest.raises(ValueError, match="schema"):
        api.StudyResult.from_dict({**result.to_dict(), "schema_version": "2.0"})


def test_to_volume_qa_feeds_the_exports(fake_measurement):
    result = _study({0: septum_slice()}, num_slices=1).measure(head_ct_model=None)

    qa = result.to_volume_qa("Scanner")

    roi1, roi2 = qa.structures[SEPTUM_PELLUCIDUM].rois[0]
    m = result.measurements[0]
    assert (roi1.col0, roi1.row0, roi1.size) == (m.roi1.x0, m.roi1.y0, m.roi1.width)
    assert roi2.structure == "Ventricular System"
    assert qa.series_label == "Head 4.0 Hr40 (1 slices, 4.0 mm)" and qa.group == "Scanner"


# ---- settings, errors, import ----


def test_settings_are_immutable_and_validated():
    settings = api.Settings()
    with pytest.raises(Exception):
        settings.tissue.adaptive.block_size = 999
    with pytest.raises(ValueError):
        api.Settings(recipe="nope")
    with pytest.raises(ValueError):
        api.TissueSettings(method="Adaptive Thresholding")  # labels are for display; ids select
    with pytest.raises(ValueError, match="plausible_low"):
        api.AdaptiveSettings(block_size=31, plausible_low=50, plausible_high=15, min_component_size=20)


def test_every_api_error_is_a_bictool_error():
    for name in ("SeriesError", "ModelNotAvailable", "NotSegmented", "InvalidSegmentation", "InvalidRois", "Cancelled"):
        assert issubclass(getattr(api, name), api.BictoolError)
    assert issubclass(api.InvalidRois, ValueError) and issubclass(api.ModelNotAvailable, FileNotFoundError)


def test_importing_the_api_loads_neither_tensorflow_nor_qt():
    import subprocess
    import sys

    code = (
        "import sys; import bictool.api; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'tensorflow', 'keras', 'PySide6'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout

    assert out.strip() == "[]"
