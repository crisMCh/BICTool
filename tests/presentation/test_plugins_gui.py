"""Plug-ins in the program: Settings > Plug-ins, a plug-in's WM/GM method in
the Method drop-down with its generated settings dialog, and plug-in values
from a QA run on the Results page and in the export rows. Plug-ins are
written into temporary folders; synthetic data only."""

import dataclasses
import textwrap
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt

import bictool.application.pipeline as pipeline
from bictool.api import plugins
from bictool.application import plugin_manager
from bictool.infrastructure.ui_settings import PLUGINS_DISABLED, UiSettings
from bictool.integration.results_export import PLUGIN_HEADER, plugin_rows
from bictool.presentation.viewmodels.plugins_viewmodel import PluginsViewModel, format_disabled, parse_disabled
from bictool.presentation.viewmodels.qa_run_viewmodel import QaRunViewModel
from bictool.presentation.viewmodels.results_viewmodel import ResultsViewModel
from bictool.presentation.viewmodels.roi_pairs import PAIR_LABELS
from bictool.presentation.viewmodels.roi_selection import RoiRect
from bictool.presentation.views.plugin_settings_dialog import PluginSettingsDialog
from bictool.presentation.views.plugins_dialog import PluginsDialog

HEADER = 'from bictool.api import plugins\nBICTOOL_PLUGIN = plugins.PluginInfo(name="{name}", version="1.0", requires_api=">=0.2,<1")\n'

METRICS = """
@plugins.hookimpl
def bictool_pair_metrics(pair):
    return [plugins.MetricValue("roi2_mean", float(pair.roi2_pixels.mean()), "HU")]

@plugins.hookimpl
def bictool_study_metrics(study):
    return [plugins.MetricValue("pairs", float(len(study.pairs)))]
"""

WMGM = """
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Level:
    level: float = 30.0
    smooth: bool = False
    passes: int = 2

def classify(slice_hu, brain_mask, settings, context):
    return {plugins.WHITE_MATTER: brain_mask & (slice_hu < settings.level),
            plugins.GRAY_MATTER: brain_mask & (slice_hu >= settings.level)}

@plugins.hookimpl
def bictool_tissue_methods():
    return [plugins.TissueMethod("level", "Level split", classify, Level())]
"""


def _load(tmp_path, **sources):
    for name, body in sources.items():
        (tmp_path / f"{name.replace('-', '_')}.py").write_text(HEADER.format(name=name) + textwrap.dedent(body), encoding="utf-8")
    manager = plugin_manager.load_plugins([tmp_path], entry_points=False)
    plugin_manager.use(manager)
    return manager


# ---- Settings > Plug-ins ----


def test_the_plugin_list_switches_plugins_off_for_the_next_start(tmp_path):
    (tmp_path / "broken.py").write_text("raise ValueError('bad')\n", encoding="utf-8")
    manager = _load(tmp_path, **{"metric-a": METRICS})
    vm = PluginsViewModel(manager.plugins, set(), [tmp_path])

    rows = {row.name: row for row in vm.rows()}
    dialog = PluginsDialog(vm)
    item = next(dialog._table.item(r, 0) for r in range(dialog._table.rowCount()) if dialog._table.item(r, 1).text() == "metric-a")
    item.setCheckState(Qt.CheckState.Unchecked)

    assert rows["metric-a"].status == "Active" and rows["metric-a"].switchable
    broken = next(row for row in rows.values() if row.status == "Error")
    assert "ValueError: bad" in broken.details and not broken.switchable
    assert vm.changed and vm.disabled == {"metric-a"}
    assert format_disabled({"b", "a"}) == "a,b" and parse_disabled(" a, ,b") == {"a", "b"}
    assert vm.summary().endswith("active.")


def test_disabled_plugins_are_not_loaded_at_start(tmp_path):
    UiSettings().set_str(PLUGINS_DISABLED, "metric-a")
    (tmp_path / "metric_a.py").write_text(HEADER.format(name="metric-a") + textwrap.dedent(METRICS), encoding="utf-8")

    manager = plugin_manager.load_plugins([tmp_path], entry_points=False, disabled=parse_disabled(UiSettings().get_str(PLUGINS_DISABLED)))

    assert [(p.name, p.status) for p in manager.plugins] == [("metric-a", "disabled")]
    assert not manager.has_pair_metrics


# ---- a plug-in WM/GM method ----


def test_a_plugin_method_is_in_the_drop_down_with_a_generated_settings_dialog(tmp_path):
    from bictool.presentation.views.main_window import MainWindowView

    _load(tmp_path, **{"my-wmgm": WMGM})
    window = MainWindowView()
    combo = window._brain_tissue_method_combo

    items = [combo.itemText(i) for i in range(combo.count())]
    combo.setCurrentText("Level split (plug-in)")

    assert "Level split (plug-in)" in items and window.view_model.brain_tissue.method == "my-wmgm.level"
    assert window._brain_tissue_settings_btn.isEnabled()
    settings = window.view_model.brain_tissue.tissue_settings()
    assert settings.method == "my-wmgm.level" and settings.plugin.level == 30.0

    defaults = settings.plugin
    dialog = PluginSettingsDialog("Level split", dataclasses.replace(defaults, level=25.0, smooth=True), defaults)
    assert set(dialog._inputs) == {"level", "smooth", "passes"}
    assert dialog.settings() == dataclasses.replace(defaults, level=25.0, smooth=True)
    dialog._on_restore_defaults_clicked()
    assert dialog.settings() == defaults

    image = np.full((32, 32), 35.0)
    image[:, :16] = 20.0
    result = window.view_model.brain_tissue.classifier()(image, np.ones((32, 32), bool), 0)
    assert result.masks["White Matter"][:, :16].all() and not result.masks["CSF"].any()


SLOW_WMGM = """
import time
SEEN = []

def classify(slice_hu, brain_mask, settings, context):
    SEEN.append((context.pixel_spacing_mm, None if context.labels is None else sorted(context.labels)))
    time.sleep(0.6)
    return {plugins.WHITE_MATTER: brain_mask}

@plugins.hookimpl
def bictool_tissue_methods():
    return [plugins.TissueMethod("slow", "Slow split", classify)]
"""


def test_a_slow_plugin_method_is_flagged_and_sees_the_cnn_labels(tmp_path):
    import sys

    from bictool.presentation.viewmodels.brain_tissue_viewmodel import BrainTissueViewModel

    _load(tmp_path, **{"slow-wmgm": SLOW_WMGM})
    labels = {"Brain Parenchyma": np.ones((16, 16), bool)}
    volume = SimpleNamespace(num_slices=1, pixel_spacing_mm=0.5, get_slice_hu=lambda i: np.full((16, 16), 30.0))
    vm = BrainTissueViewModel(lambda: volume, lambda: {0: SimpleNamespace(masks=labels)}, None, None, SimpleNamespace(smoothing_mm=0.0))
    texts = []
    vm.slow_text_changed.connect(texts.append)

    vm.set_method("slow-wmgm.slow")
    vm.result(0)
    vm.set_method("slow-wmgm.slow")  # a change clears the warning

    assert sys.modules["bictool_plugins.slow_wmgm"].SEEN == [(0.5, ["Brain Parenchyma"])]
    assert texts[0].startswith("⚠ Slow split (plug-in) needs") and texts[1] == "" and vm.slow_text == ""


# ---- plug-in values from a QA run ----


def _quality(*a, **k):
    return SimpleNamespace(
        mtf_area=0.5, predicted_label=1, decision_value=0.1, recipe="physical",
        feature_vector=np.zeros(7), nps_f_av=0.2, nps_f_peak=0.1, contrast_roi1=0.1, contrast_roi2=0.02,
    )


def test_plugin_values_reach_the_results_page_and_the_export_rows(tmp_path, monkeypatch):
    _load(tmp_path, **{"metric-a": METRICS, "fails": "@plugins.hookimpl\ndef bictool_pair_metrics():\n    raise KeyError('x')\n"})
    monkeypatch.setattr(pipeline, "evaluate_slice_quality", _quality)
    volume = SimpleNamespace(
        num_slices=4, pixel_spacing_mm=0.5, slice_thickness_mm=4.0, get_slice_hu=lambda i: np.full((64, 64), 30.0 + i),
    )
    rois = {"falx.roi1": RoiRect(5, 5, 25, 25), "falx.roi2": RoiRect(30, 30, 50, 50)}
    qa = QaRunViewModel(lambda job: job.run(), lambda: volume, lambda i: rois, lambda: {})
    results = ResultsViewModel(qa, lambda: None)

    qa.run([(0, "falx"), (3, "falx")])

    values = {v.plugin: v for v in qa.plugin_values[(3, "falx")]}
    assert values["metric-a"].values == (plugins.MetricValue("roi2_mean", 33.0, "HU"),)
    assert values["fails"].error == "KeyError: 'x'"
    assert qa.study_plugin_values == (plugins.PluginValues("metric-a", (plugins.MetricValue("pairs", 2.0),)),)
    assert "metric-a: roi2_mean 33 HU" in results.plugin_note(3, "falx")
    lines = results.plugin_lines()
    assert "Plug-in metric-a: pairs 2" in lines and any("fails: problem on 2 ROI pair(s)" in line for line in lines)

    rows = plugin_rows(qa.plugin_values, qa.study_plugin_values, PAIR_LABELS)
    assert len(PLUGIN_HEADER) == 7
    assert [4, "Falx", "metric-a", "roi2_mean", 33.0, "HU", ""] in rows
    assert [1, "Falx", "fails", "", "", "", "KeyError: 'x'"] in rows
    assert ["", "whole series", "metric-a", "pairs", 2.0, "", ""] in rows


SLICE_PLUGIN = """
import numpy as np

@plugins.hookimpl
def bictool_slice_metrics(ct_slice):
    roi = np.zeros(ct_slice.image.shape, bool)
    roi[10:20, 10:20] = True
    return [plugins.MetricValue("box_mean", float(ct_slice.image[roi].mean()), "HU"), plugins.RoiMarker("box", roi)]

@plugins.hookimpl
def bictool_artefact_hints(study):
    return [plugins.ArtefactHint("Ring-like pattern", (3,))]
"""


def _fake_study(num_slices=4):
    return SimpleNamespace(
        num_slices=num_slices, pixel_spacing_mm=0.5, slice_thickness_mm=4.0,
        get_slice_hu=lambda i: np.full((64, 64), 30.0 + i),
    )


def test_slice_values_hints_and_markers_reach_the_results_and_export(tmp_path, monkeypatch):
    _load(tmp_path, **{"slice-a": SLICE_PLUGIN})
    monkeypatch.setattr(pipeline, "evaluate_slice_quality", _quality)
    volume = _fake_study()
    rois = {"falx.roi1": RoiRect(5, 5, 25, 25), "falx.roi2": RoiRect(30, 30, 50, 50)}
    qa = QaRunViewModel(lambda job: job.run(), lambda: volume, lambda i: rois, lambda: {})
    results = ResultsViewModel(qa, lambda: None)

    qa.run([(0, "falx"), (3, "falx")])

    assert qa.slice_plugin_values[3][0].values == (plugins.MetricValue("box_mean", 33.0, "HU"),)
    assert [m.name for m in qa.plugin_markers(3)] == ["box"] and qa.has_plugin_markers()
    assert "slice-a: box_mean 33 HU" in results.plugin_note(3, "falx")
    assert "Possible artefact (plug-in slice-a): Ring-like pattern" in results.plugin_note(3, "falx")
    assert "Possible artefact" not in results.plugin_note(0, "falx")
    assert results.plugin_hint_lines() == ["Possible artefact (plug-in slice-a): Ring-like pattern (slice 4)"]
    assert qa.plugin_tissue_method is None  # the plug-in never asked for WM/GM masks

    rows = plugin_rows(
        qa.plugin_values, qa.study_plugin_values, PAIR_LABELS, slice_values=qa.slice_plugin_values, hints=qa.plugin_hints,
    )
    assert [4, "slice", "slice-a", "box_mean", 33.0, "HU", ""] in rows
    assert ["4", "possible artefact", "slice-a", "Ring-like pattern", "", "", ""] in rows

    qa.reset()
    assert qa.slice_plugin_values == {} and qa.plugin_hints == () and not qa.has_plugin_markers()


def test_the_window_draws_plugin_rois_on_request(tmp_path):
    from bictool.presentation.views.main_window import PLUGIN_MARKER_PREFIX, MainWindowView

    _load(tmp_path, **{"slice-a": SLICE_PLUGIN})
    window = MainWindowView()
    volume = _fake_study()
    qa = window.view_model.qa
    window.view_model.study.volume = volume
    window.view_model.study.current_slice = 3
    window._roi_viewport.set_image(volume.get_slice_hu(3))
    context = pipeline.slice_context(volume, 3, (), pipeline.PluginInputs(volume, {}, pipeline.TissueSettings()))
    qa.slice_plugin_values = {3: plugin_manager.current().slice_metrics(context)}

    window._refresh_plugin_markers()
    assert not window._plugin_markers_cb.isHidden() and window._plugin_marker_keys == []  # offered, not ticked
    window._plugin_markers_cb.setChecked(True)

    key = f"{PLUGIN_MARKER_PREFIX}0"
    assert window._plugin_marker_keys == [key]
    assert window._roi_viewport._overlay_masks[key][10:20, 10:20].all()
    assert window._roi_viewport._overlay_display_names[key] == "slice-a: box"
    qa.reset()  # a new run or study: the markers go
    assert window._plugin_marker_keys == [] and window._plugin_markers_cb.isHidden()
    window.view_model.study.volume = None


def test_without_plugins_no_plugin_job_runs(monkeypatch):
    monkeypatch.setattr(pipeline, "evaluate_slice_quality", _quality)
    volume = SimpleNamespace(num_slices=1, pixel_spacing_mm=0.5, get_slice_hu=lambda i: np.zeros((64, 64)))
    rois = {"falx.roi1": RoiRect(5, 5, 25, 25), "falx.roi2": RoiRect(30, 30, 50, 50)}
    jobs = []
    qa = QaRunViewModel(lambda job: jobs.append(job) or job.run(), lambda: volume, lambda i: rois, lambda: {})

    qa.run([(0, "falx")])

    assert len(jobs) == 1 and qa.plugin_values == {} and qa.study_plugin_values == ()
