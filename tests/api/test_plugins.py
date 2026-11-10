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
    assert result.plugins == {"roi-stats": "1.0", "otsu-wmgm": "1.0"}
    assert [v.plugin for v in result.plugin_values] == ["roi-stats"]  # the falx median is NaN here (no falx pair)

    data = json.loads(result.to_json())
    assert data["measurements"][0]["plugin_values"][0]["plugin"] == "roi-stats"
    assert data["plugins"] == {"roi-stats": "1.0", "otsu-wmgm": "1.0"}
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
    guide = (Path(__file__).resolve().parents[2] / "docs" / "plugins.md").read_text(encoding="utf-8")

    assert plugins.HOOKS == ("bictool_pair_metrics", "bictool_study_metrics", "bictool_tissue_methods")
    for hook in plugins.HOOKS:
        assert f"`{hook}" in guide
