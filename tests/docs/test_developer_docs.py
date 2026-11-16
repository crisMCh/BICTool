"""The developer guide (docs/developer) stays complete and correct: every public
name, hook, hook argument and field is in the reference; every link and anchor
resolves (with GitHub's and the build's heading ids); every page is in the
built guide; every Python example parses, and every example plug-in loads and
runs on a synthetic study without a problem. No real data needed."""

import ast
import dataclasses
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from markdown.extensions.toc import slugify

import bictool.application.pipeline as pipeline
from bictool import api
from bictool.api import plugins

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "docs" / "developer"
REF = DEV / "reference"
PAGES = sorted(DEV.rglob("*.md"))
POINTERS = [ROOT / "docs" / "api.md", ROOT / "docs" / "plugins.md"]

sys.path.insert(0, str(ROOT / "tests" / "api"))
from api_synthetic import fake_quality, fake_volume, septum_slice  # noqa: E402

TYPES = {
    "results.md": ["StudyResult", "SeriesInfo", "SeriesChoice", "SliceRef", "Measurement", "Curves", "Failure", "Verdict",
                   "FeatureFinding", "NoiseScatter", "SeptumCheck", "ArtefactHints", "BasalGangliaSlice",
                   "SegmentationCheck"],
    "settings.md": ["Settings", "TissueSettings", "AdaptiveSettings", "FcmSettings", "BrainTissueSettings"],
    "rois.md": ["RoiRect"],
    "plugin-types.md": ["PluginInfo", "MetricValue", "RoiMarker", "ArtefactHint", "PairContext", "SliceContext",
                        "StudyContext", "TissueContext", "TissueMethod", "ExportFormat", "PluginValues", "PluginMarker",
                        "PluginHints", "LoadedPlugin"],
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _without_code(text: str) -> str:
    return re.sub(r"^(```|~~~).*?^\1", "", text, flags=re.M | re.S)


def _entries(text: str) -> dict[str, str]:
    """`### \\`Name\\`` (or ##) heading -> the text up to the next such heading."""
    parts = re.split(r"^(#{2,3} .*)$", _without_code(text), flags=re.M)
    entries = {}
    for heading, body in zip(parts[1::2], parts[2::2]):
        match = re.match(r"#{2,3} `([\w.]+)`", heading)
        if match:
            entries[match.group(1)] = body
    return entries


def _public_methods(cls) -> list[str]:
    return [name for name, value in vars(cls).items()
            if not name.startswith("_") and (callable(value) or isinstance(value, (property, classmethod, staticmethod)))]


# ---- completeness ----


def test_every_public_api_name_is_in_the_reference():
    reference = "".join(_read(p) for p in REF.glob("*.md"))
    missing = [name for name in api.__all__ if f"`{name}`" not in reference and f"`{name}(" not in reference]
    assert not missing


def test_every_plugin_name_is_in_the_plugin_reference():
    text = _read(REF / "plugin-types.md")
    assert not [name for name in plugins.__all__ if f"`{name}`" not in text]


def test_every_hook_and_its_arguments_are_in_the_hook_reference():
    entries = _entries(_read(REF / "hooks.md"))
    for hook in plugins.HOOKS:
        assert hook in entries, hook
        arguments = [p for p in inspect.signature(getattr(plugins.HookSpecs, hook)).parameters if p != "self"]
        for argument in arguments:
            assert f"`{argument}`" in entries[hook], (hook, argument)


@pytest.mark.parametrize("file, names", TYPES.items())
def test_every_field_of_the_public_types_is_documented(file, names):
    entries = _entries(_read(REF / file))
    for name in names:
        assert name in entries, name
        cls = getattr(api, name, None) or getattr(plugins, name)
        fields = [f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")]
        fields += [m for m in _public_methods(cls) if not m.startswith("from_volume")]
        missing = [f for f in fields if f"`{f}" not in entries[name] and f"{f}(" not in entries[name]]
        assert not missing, (name, missing)


def test_every_method_of_study_and_roi_store_is_documented():
    functions, rois = _read(REF / "api-functions.md"), _entries(_read(REF / "rois.md"))
    assert not [m for m in _public_methods(api.Study) if f"`{m}" not in functions and f"Study.{m}" not in functions]
    assert not [m for m in _public_methods(api.RoiPairStore) if f"`{m}(" not in rois["RoiPairStore"]]


def test_every_page_is_in_the_built_guide():
    spec = importlib.util.spec_from_file_location("build_manual", ROOT / "docs" / "manual" / "build_manual.py")
    build = importlib.util.module_from_spec(spec)
    sys.modules["build_manual"] = build  # dataclasses look their module up there
    spec.loader.exec_module(build)

    assert sorted(build.DEVELOPER_PAGES) == sorted(p.relative_to(DEV).as_posix() for p in PAGES)
    body, tokens = build.developer_guide()
    ids = set(re.findall(r'id="([^"]+)"', body))
    links = re.findall(r'href="#([^"]+)"', body)
    assert links and not [link for link in links if link not in ids]  # every link inside the built guide resolves


# ---- links ----


def _github_slug(text: str) -> str:
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def _anchors(path: Path) -> tuple[set[str], set[str]]:
    """The heading ids of `path` as GitHub and as the build (python-markdown) make them."""
    headings = [re.sub(r"`", "", h) for h in re.findall(r"^#{1,6} (.*)$", _without_code(_read(path)), flags=re.M)]
    return {_github_slug(h) for h in headings}, {slugify(h, "-") for h in headings}


@pytest.mark.parametrize("page", PAGES + POINTERS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_links_and_anchors_resolve(page):
    for label, target in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", _without_code(_read(page))):
        if re.match(r"[a-z][a-z0-9+.-]*:", target):
            continue
        file, _, anchor = target.partition("#")
        destination = (page.parent / file).resolve() if file else page
        assert destination.is_file(), (page.name, target)
        if anchor and destination.suffix == ".md":
            github, built = _anchors(destination)
            assert anchor in github and anchor in built, (page.name, target)


# ---- examples ----


def _python_blocks() -> list[tuple[str, str]]:
    blocks = []
    for page in PAGES:
        for block in re.findall(r"^```python\n(.*?)^```", _read(page), flags=re.M | re.S):
            blocks.append((page.relative_to(DEV).as_posix(), block))
    return blocks


def test_every_python_example_parses():
    blocks = _python_blocks()
    assert len(blocks) > 30
    for page, block in blocks:
        ast.parse(block, filename=page)


def test_the_example_plugins_load_and_run(tmp_path, monkeypatch):
    examples = [block for _page, block in _python_blocks() if "BICTOOL_PLUGIN" in block]
    for number, block in enumerate(examples):
        (tmp_path / f"example_{number}.py").write_text(block, encoding="utf-8")
    loaded = api.load_plugins([tmp_path], installed=False)
    assert len(loaded) == len(examples) >= 5
    assert [(p.name, p.status, p.message) for p in loaded if p.status != "active" or p.message] == []

    monkeypatch.setattr(pipeline, "evaluate_slice_quality", lambda *a, **k: fake_quality())
    volume = fake_volume({0: septum_slice()}, 1)
    study = api.Study(volume)
    study.set_segmentation({0: septum_slice()})
    result = study.measure(head_ct_model=None)

    groups = [m.plugin_values for m in result.measurements] + list(result.slice_plugin_values.values())
    problems = [(v.plugin, v.error) for group in groups + [result.plugin_values] for v in group if v.error]
    problems += [(h.plugin, h.error) for h in result.plugin_hints if h.error]
    assert problems == []
    assert {v.plugin for v in result.measurements[0].plugin_values} == {"roi2-sd"}
    assert {v.plugin for v in result.slice_plugin_values[0]} == {"wm-disc"}
    for format_id in api.export_formats():
        assert api.write_export(result, tmp_path / "out" / "case", format_id).is_file()
    tissue = api.TissueSettings(method="level-split.level", plugin={"level_hu": 28.0})
    masks = pipeline.study_tissue_classifier(tissue, {0: septum_slice()}, volume)(
        volume.get_slice_hu(0), np.ones(volume.get_slice_hu(0).shape, bool), 0
    ).masks
    assert set(masks) >= {plugins.WHITE_MATTER, plugins.GRAY_MATTER, plugins.CSF}
