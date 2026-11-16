"""The plug-in system (FR-08): finding and checking plug-ins, isolation, the
metric and WM/GM-method extension points, plug-in values in results. Plug-ins
are written into temporary folders; the two shipped examples are loaded too.
Synthetic data only."""

import ast
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from api_synthetic import fake_volume, septum_slice

import bictool.application.pipeline as pipeline
from bictool import api
from bictool.api import plugins
from bictool.application import plugin_manager

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "plugins"
HEADER = 'from bictool.api import plugins\nBICTOOL_PLUGIN = plugins.PluginInfo(name="{name}", version="1.0", requires_api="{api}")\n'


def _plugin(folder: Path, name: str, body: str, api_range: str = ">=0.2,<1", file: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{file or name.replace('-', '_')}.py"
    path.write_text(HEADER.format(name=name, api=api_range) + textwrap.dedent(body), encoding="utf-8")
    return path


def _load(folder, **kwargs):
    loaded = api.load_plugins([folder], installed=False, **kwargs)
    return {p.name: p for p in loaded}


def _study(masks_by_slice, num_slices=1, settings=None):
    study = api.Study(fake_volume(masks_by_slice, num_slices), settings=settings)
    study.set_segmentation(masks_by_slice)
    return study


METRIC = """
@plugins.hookimpl
def bictool_pair_metrics(pair):
    return [plugins.MetricValue("roi2_sd", float(pair.roi2_pixels.std()), "HU")]
"""


# ---- finding and checking ----


def test_the_shipped_examples_load(tmp_path):
    loaded = {p.name: p for p in api.load_plugins([EXAMPLES], installed=False)}

    assert loaded["roi-stats"].status == "active"
    assert loaded["roi-stats"].hooks == ("bictool_pair_metrics", "bictool_study_metrics")
    assert loaded["otsu-wmgm"].status == "active" and loaded["otsu-wmgm"].hooks == ("bictool_tissue_methods",)
    assert "otsu-wmgm.otsu" in plugin_manager.current().tissue_methods


def test_the_shipped_measurement_example_runs(fake_measurement):
    api.load_plugins([EXAMPLES], installed=False)

    result = _study({0: septum_slice()}).measure(head_ct_model=None)

    values = {v.plugin: v for v in result.slice_plugin_values[0]}
    assert values["wm-box"].error is None and [v.name for v in values["wm-box"].values] == ["wm_mean", "wm_sd"]
    assert {v.plugin: v for v in result.plugin_values}["wm-box"].error is None
    assert all(h.error is None for h in result.plugin_hints)


def test_problems_are_reported_and_do_not_stop_the_others(tmp_path):
    _plugin(tmp_path, "good", METRIC)
    (tmp_path / "broken.py").write_text("raise RuntimeError('cannot start')\n", encoding="utf-8")
    (tmp_path / "no_info.py").write_text("x = 1\n", encoding="utf-8")
    _plugin(tmp_path, "too-new", METRIC, api_range=">=5")
    _plugin(tmp_path, "nothing", "x = 1\n")
    _plugin(tmp_path, "bad-args", "@plugins.hookimpl\ndef bictool_pair_metrics(pair, extra):\n    return []\n")
    _plugin(tmp_path, "good", METRIC, file="good_twice")
    _plugin(tmp_path, "off", METRIC)
    (tmp_path / "_private.py").write_text("raise SystemExit\n", encoding="utf-8")

    loaded = api.load_plugins([tmp_path], installed=False, disabled={"off"})
    by_source = {Path(p.source).stem: p for p in loaded}

    assert by_source["good"].status == "active"
    assert by_source["broken"].status == "error" and "RuntimeError: cannot start" in by_source["broken"].message
    assert by_source["no_info"].status == "error" and "BICTOOL_PLUGIN" in by_source["no_info"].message
    assert by_source["too_new"].status == "incompatible" and ">=5" in by_source["too_new"].message
    assert by_source["nothing"].status == "error" and "none of the hooks" in by_source["nothing"].message
    assert by_source["bad_args"].status == "error"
    assert by_source["good_twice"].status == "error" and "already" in by_source["good_twice"].message
    assert by_source["off"].status == "disabled"
    assert "_private" not in by_source
    assert plugin_manager.current().versions == {"good": "1.0"}


def test_unknown_and_unmarked_hook_functions_are_reported(tmp_path):
    _plugin(tmp_path, "newer", METRIC + "\n@plugins.hookimpl\ndef bictool_future_hook(study):\n    return []\n")
    _plugin(tmp_path, "only-future", "@plugins.hookimpl\ndef bictool_future_hook(study):\n    return []\n")
    _plugin(tmp_path, "unmarked", METRIC + "\ndef bictool_study_metrics(study):\n    return []\n")

    loaded = _load(tmp_path)

    assert loaded["newer"].status == "active" and loaded["newer"].hooks == ("bictool_pair_metrics",)
    assert "bictool_future_hook is not a hook of this BICTool" in loaded["newer"].message
    assert loaded["only-future"].status == "error"
    assert "none of the hooks" in loaded["only-future"].message and "bictool_future_hook" in loaded["only-future"].message
    assert loaded["unmarked"].status == "active"
    assert "bictool_study_metrics is not marked @plugins.hookimpl" in loaded["unmarked"].message


def test_installed_packages_are_found_by_entry_point(monkeypatch, tmp_path):
    module = SimpleNamespace(
        BICTOOL_PLUGIN=plugins.PluginInfo("pkg-metric", "2.1", ">=0.2,<1"),
        bictool_pair_metrics=plugins.hookimpl(lambda pair: [plugins.MetricValue("one", 1.0)]),
    )
    entry = SimpleNamespace(value="pkg_metric", dist=SimpleNamespace(name="bictool-pkg-metric"), load=lambda: module)
    import bictool.infrastructure.plugin_loader as loader

    monkeypatch.setattr(loader.importlib.metadata, "entry_points", lambda group: [entry] if group == "bictool.plugins" else [])

    loaded = api.load_plugins([], installed=True)

    assert [(p.name, p.status, p.source) for p in loaded] == [("pkg-metric", "active", "package bictool-pkg-metric: pkg_metric")]


# ---- metrics ----


def test_pair_and_study_values_are_in_the_result_and_json(tmp_path, fake_measurement):
    api.load_plugins([EXAMPLES], installed=False)

    result = _study({0: septum_slice()}).measure(head_ct_model=None)

    values = {v.plugin: v for v in result.measurements[0].plugin_values}
    stats = {m.name: m for m in values["roi-stats"].values}
    m = result.measurements[0]
    expected = fake_volume({0: septum_slice()}).get_slice_hu(0)[m.roi2.y0:m.roi2.y1, m.roi2.x0:m.roi2.x1]
    assert stats["roi2_mean"].value == pytest.approx(float(expected.mean())) and stats["roi2_mean"].unit == "HU"
    assert values["roi-stats"].error is None
    assert result.plugins == {"roi-stats": "1.0", "otsu-wmgm": "1.0", "wm-box": "1.0"}
    assert [v.plugin for v in result.plugin_values] == ["roi-stats", "wm-box"]  # NaN medians are still reported

    data = json.loads(result.to_json())
    assert data["measurements"][0]["plugin_values"][0]["plugin"] == "roi-stats"
    assert data["plugins"] == {"roi-stats": "1.0", "otsu-wmgm": "1.0", "wm-box": "1.0"}
    back = api.StudyResult.from_json(result.to_json())
    assert back.measurements[0].plugin_values[0].values[0].name == "roi1_mean"


def test_a_failing_or_misbehaving_plugin_is_recorded_and_the_study_goes_on(tmp_path, fake_measurement):
    _plugin(tmp_path, "fails", "@plugins.hookimpl\ndef bictool_pair_metrics(pair):\n    raise ZeroDivisionError('oops')\n")
    _plugin(tmp_path, "odd", """
        @plugins.hookimpl
        def bictool_pair_metrics():
            return [plugins.MetricValue("ok", 2), plugins.MetricValue("bad", "x"), "nonsense"]
    """)
    _plugin(tmp_path, "writes", """
        @plugins.hookimpl
        def bictool_pair_metrics(pair):
            pair.image[:] = 0
    """)
    api.load_plugins([tmp_path], installed=False)
    study = _study({0: septum_slice()})
    before = study.volume.get_slice_hu(0).copy()

    result = study.measure(head_ct_model=None)

    values = {v.plugin: v for v in result.measurements[0].plugin_values}
    assert values["fails"].error == "ZeroDivisionError: oops" and values["fails"].values == ()
    assert [v.name for v in values["odd"].values] == ["ok"] and "not a number" in values["odd"].error
    assert "read-only" in values["writes"].error
    np.testing.assert_array_equal(study.volume.get_slice_hu(0), before)  # BICTool's image is untouched
    assert len(result.measurements) == 1


def test_without_plugins_the_result_has_no_plugin_values(fake_measurement):
    result = _study({0: septum_slice()}).measure(head_ct_model=None)

    assert result.plugins == {} and result.measurements[0].plugin_values == () and result.plugin_values == ()


MEASURE = """
import numpy as np
SEEN = []

@plugins.hookimpl
def bictool_slice_metrics(ct_slice):
    brain = ct_slice.brain_mask()
    wm = ct_slice.tissue()[plugins.WHITE_MATTER]
    SEEN.append((ct_slice.slice_index, len(ct_slice.pairs), ct_slice.tissue_method, brain.shape, wm.dtype == bool))
    roi = np.zeros(ct_slice.image.shape, bool)
    roi[2:6, 3:9] = True
    return [plugins.MetricValue("box_mean", float(ct_slice.image[roi].mean()), "HU"),
            plugins.RoiMarker("box", roi, description="a test box")]

@plugins.hookimpl
def bictool_study_metrics(study):
    corner = np.zeros(study.image(0).shape, bool)
    corner[0, 0] = True
    return [plugins.MetricValue("measured", float(len(study.measured_slices))),
            plugins.RoiMarker("corner", corner, slice_index=0)]

@plugins.hookimpl
def bictool_artefact_hints(study):
    return [plugins.ArtefactHint("Ring-like pattern", (0,))]
"""


def test_slice_values_markers_and_hints_are_in_the_result_and_json(tmp_path, fake_measurement):
    import sys

    _plugin(tmp_path, "measure", MEASURE)
    api.load_plugins([tmp_path], installed=False)
    study = _study({0: septum_slice()})
    image = study.volume.get_slice_hu(0)

    result = study.measure(head_ct_model=None)

    (values,) = result.slice_plugin_values[0]
    assert values.plugin == "measure" and values.error is None
    assert values.values == (plugins.MetricValue("box_mean", pytest.approx(float(image[2:6, 3:9].mean())), "HU"),)
    assert sys.modules["bictool_plugins.measure"].SEEN == [(0, 1, "adaptive", image.shape, True)]
    box = next(m for m in result.plugin_markers(0) if m.name == "box")
    assert box.box == api.RoiRect(3, 2, 9, 6) and box.description == "a test box"
    expected = np.zeros(image.shape, bool)
    expected[2:6, 3:9] = True
    np.testing.assert_array_equal(box.full_mask(image.shape), expected)
    assert {m.name for m in result.plugin_markers()} == {"box", "corner"}
    assert [v.name for v in result.plugin_values[0].values] == ["measured"]
    assert result.plugin_hints == (plugins.PluginHints("measure", (plugins.ArtefactHint("Ring-like pattern", (0,)),)),)

    back = api.StudyResult.from_json(result.to_json())
    assert back.to_dict() == result.to_dict()
    np.testing.assert_array_equal(back.plugin_markers(0)[0].full_mask(image.shape), expected)

    rows = result.to_volume_qa().plugin_rows
    assert [1, "slice", "measure", "box_mean", values.values[0].value, "HU", ""] in rows
    assert ["1", "possible artefact", "measure", "Ring-like pattern", "", "", ""] in rows
    assert rows[-1] == ["", "plug-in input", "", "WM/GM method", "adaptive", "", ""]


def test_invalid_markers_and_hints_are_dropped_with_a_note(tmp_path, fake_measurement):
    _plugin(tmp_path, "bad", """
        import numpy as np

        @plugins.hookimpl
        def bictool_slice_metrics(ct_slice):
            shape = ct_slice.image.shape
            return [plugins.MetricValue("ok", 1.0),
                    plugins.RoiMarker("wrong shape", np.ones((3, 3), bool)),
                    plugins.RoiMarker("empty", np.zeros(shape, bool)),
                    plugins.RoiMarker("other slice", np.ones(shape, bool), slice_index=5)]

        @plugins.hookimpl
        def bictool_study_metrics(study):
            return [plugins.RoiMarker("no slice", np.ones(study.image(0).shape, bool))]

        @plugins.hookimpl
        def bictool_artefact_hints(study):
            return [plugins.ArtefactHint("x" * 300), plugins.ArtefactHint("bad slice", (99,)), "nonsense",
                    plugins.ArtefactHint("  fine \\n text ")]
    """)
    api.load_plugins([tmp_path], installed=False)

    result = _study({0: septum_slice()}).measure(head_ct_model=None)

    (values,) = result.slice_plugin_values[0]
    assert [v.name for v in values.values] == ["ok"] and values.markers == ()
    for problem in ("wrong shape", "is empty", "slice 5 is not allowed"):
        assert problem in values.error
    assert "slice_index is missing" in result.plugin_values[0].error
    (hints,) = result.plugin_hints
    assert hints.hints == (plugins.ArtefactHint("fine text"),)
    assert "longer than 200" in hints.error and "outside" in hints.error and "nonsense" in hints.error


def test_without_slice_or_study_plugins_nothing_extra_is_computed(tmp_path, fake_measurement, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the plug-in inputs must not be built")

    monkeypatch.setattr(pipeline, "PluginInputs", forbidden)
    plain = _study({0: septum_slice()}).measure(head_ct_model=None)
    _plugin(tmp_path, "pairs-only", METRIC)
    api.load_plugins([tmp_path], installed=False)
    with_pair_plugin = _study({0: septum_slice()}).measure(head_ct_model=None)

    assert plain.slice_plugin_values == {} and plain.plugin_hints == () and plain.plugin_values == ()
    assert with_pair_plugin.measurements[0].plugin_values[0].plugin == "pairs-only"
    assert with_pair_plugin.slice_plugin_values == {} and with_pair_plugin.plugin_hints == ()


# ---- export formats ----


def test_a_plugin_export_format_writes_the_result(tmp_path, fake_measurement):
    _plugin(tmp_path / "plugins", "fmt", """
        import json

        def write(result, path):
            path.write_text(json.dumps({"label": result.label, "pairs": len(result.measurements)}), encoding="utf-8")

        def fail(result, path):
            raise OSError("disk says no")

        @plugins.hookimpl
        def bictool_export_formats():
            return [plugins.ExportFormat("json", "Short JSON", ".json", write),
                    plugins.ExportFormat("broken", "Broken", ".txt", fail),
                    plugins.ExportFormat("bad", "Bad extension", "json", write),
                    plugins.ExportFormat("", "No id", ".x", write)]
    """)
    loaded = _load(tmp_path / "plugins")
    result = _study({0: septum_slice()}).measure(head_ct_model=None)

    written = api.write_export(result, tmp_path / "out" / "case", "fmt.json")

    assert api.export_formats() == {"fmt.broken": "Broken", "fmt.json": "Short JSON"}
    assert loaded["fmt"].status == "active" and "extension 'json'" in loaded["fmt"].message
    assert "not a valid ExportFormat" in loaded["fmt"].message
    assert written == tmp_path / "out" / "case.json"
    assert json.loads(written.read_text(encoding="utf-8")) == {"label": "study - Head 4.0 Hr40", "pairs": 1}
    with pytest.raises(api.PluginFailed, match="disk says no"):
        api.write_export(result, tmp_path / "out" / "case.txt", "fmt.broken")
    with pytest.raises(api.PluginFailed, match="not available"):
        api.write_export(result, tmp_path / "x", "gone.json")


# ---- WM/GM methods ----


def test_a_plugin_wmgm_method_is_selectable_and_used(tmp_path, fake_measurement):
    _plugin(tmp_path, "my-wmgm", """
        from dataclasses import dataclass
        import numpy as np
        CALLS = []

        @dataclass(frozen=True)
        class S:
            level: float = 30.0

        def classify(slice_hu, brain_mask, settings, context):
            CALLS.append((settings.level, context.slice_index, context.labels is not None))
            return {plugins.WHITE_MATTER: brain_mask & (slice_hu < settings.level),
                    plugins.GRAY_MATTER: brain_mask & (slice_hu >= settings.level),
                    plugins.CSF: np.zeros(slice_hu.shape, bool)}

        @plugins.hookimpl
        def bictool_tissue_methods():
            return [plugins.TissueMethod("level", "Level split", classify, S())]
    """)
    api.load_plugins([tmp_path], installed=False)
    module = __import__("sys").modules["bictool_plugins.my_wmgm"]
    settings = api.Settings(tissue=api.TissueSettings(method="my-wmgm.level", plugin=module.S(level=25.0)))

    classify = pipeline.study_tissue_classifier(settings.tissue, {0: septum_slice()}, fake_volume({0: septum_slice()}))
    image = fake_volume({0: septum_slice()}).get_slice_hu(0)
    result = classify(image, np.ones(image.shape, bool), 0)
    data = _study({0: septum_slice()}, settings=settings).measure(head_ct_model=None).to_dict()

    assert module.CALLS[0] == (25.0, 0, True)
    assert set(result.masks) >= {"White Matter", "Gray Matter", "CSF", "Brain Mask"}
    assert data["settings"] == {"recipe": "physical", "tissue_method": "my-wmgm.level", "tissue_parameters": {"level": 25.0}}
    back = api.StudyResult.from_dict(data)  # the plug-in's settings come back as a plain dict
    assert back.settings.tissue.method == "my-wmgm.level" and back.settings.tissue.plugin == {"level": 25.0}
    assert back.to_dict()["settings"] == data["settings"]
    rerun = pipeline.study_tissue_classifier(back.settings.tissue, {0: septum_slice()}, fake_volume({0: septum_slice()}))
    rerun(image, np.ones(image.shape, bool), 0)
    assert module.CALLS[-1] == (25.0, 0, True)  # the dict became the plug-in's settings again
    with pytest.raises(ValueError, match="do not fit"):
        bad = api.TissueSettings(method="my-wmgm.level", plugin={"nonsense": 1})
        pipeline.study_tissue_classifier(bad, {0: septum_slice()}, fake_volume({0: septum_slice()}))


def test_a_missing_plugin_method_is_a_clear_error():
    tissue = api.TissueSettings(method="gone.method")

    with pytest.raises(ValueError, match="not available"):
        pipeline.make_tissue_classifier(tissue, 0.5)


# ---- the contract ----


def test_the_label_names_match_the_cnn():
    source = (Path(pipeline.__file__).parents[1] / "domain" / "segmentation" / "head_ct_unet.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.AnnAssign | ast.Assign) and "_CLASS_NAMES" in ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
    )
    assert plugins.LABELS == tuple(names[i] for i in sorted(names))


def test_every_hook_is_documented():
    guide = (Path(__file__).resolve().parents[2] / "docs" / "developer" / "reference" / "hooks.md").read_text(encoding="utf-8")

    assert plugins.HOOKS == (
        "bictool_pair_metrics", "bictool_slice_metrics", "bictool_study_metrics", "bictool_artefact_hints",
        "bictool_tissue_methods", "bictool_export_formats",
    )
    for hook in plugins.HOOKS:
        assert f"`{hook}" in guide
