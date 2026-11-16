"""The plug-ins in use (FR-08): checks and registers what `plugin_loader` found,
calls their hooks with each plug-in isolated (an error is recorded, the study
goes on), and keeps the WM/GM methods they add.

`current()` is the manager the pipeline uses - the program sets it at start
(`use`), the API loads the installed and folder plug-ins on first use.
Plug-in values are descriptive only: nothing here feeds the study verdict.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import threading
from collections.abc import Callable, Iterable

import numpy as np
import pluggy
from pathlib import Path
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from bictool import API_VERSION
from bictool.application.plugin_types import (
    HOOKS,
    PROJECT,
    ArtefactHint,
    ExportFormat,
    HookSpecs,
    LoadedPlugin,
    MetricValue,
    PairContext,
    PluginHints,
    PluginInfo,
    PluginMarker,
    PluginValues,
    RoiMarker,
    SliceContext,
    StudyContext,
    TissueMethod,
)
from bictool.application.roi_store import RoiRect
from bictool.infrastructure.plugin_loader import FoundPlugin

logger = logging.getLogger(__name__)

ACTIVE, DISABLED, INCOMPATIBLE, ERROR = "active", "disabled", "incompatible", "error"
MAX_HINT_LENGTH = 200  # characters of one artefact hint
_EXTENSION = re.compile(r"^\.[A-Za-z0-9][A-Za-z0-9_-]{0,15}$")  # ".json", ".pdf", ...
ShapeOf = Callable[[int], "tuple[int, ...] | None"]  # slice index -> image shape, None if not allowed


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class PluginManager:
    """The checked, registered plug-ins and their hooks."""

    def __init__(self) -> None:
        self._pm = pluggy.PluginManager(PROJECT)
        self._pm.add_hookspecs(HookSpecs)
        self.plugins: list[LoadedPlugin] = []
        self.versions: dict[str, str] = {}               # active plug-in -> version
        self.tissue_methods: dict[str, TissueMethod] = {}  # "<plug-in>.<id>" -> method
        self.export_formats: dict[str, ExportFormat] = {}  # "<plug-in>.<id>" -> format

    # ---- loading ----

    def load(self, found: Iterable[FoundPlugin], disabled: Iterable[str] = ()) -> None:
        """Checks and registers plug-ins: each needs a `BICTOOL_PLUGIN` (`PluginInfo`)
        whose `requires_api` includes this API version; names must be unique."""
        disabled = set(disabled)
        for item in found:
            self.plugins.append(self._load_one(item, disabled))
        self.log_status()

    def log_status(self) -> None:
        """One log line per plug-in found (the program calls it again once its log file is set up)."""
        for plugin in self.plugins:
            logger.info("Plug-in %s %s (%s): %s %s", plugin.name, plugin.version, plugin.source, plugin.status, plugin.message)
        if not self.plugins:
            logger.info("No plug-ins found")

    def _load_one(self, item: FoundPlugin, disabled: set[str]) -> LoadedPlugin:
        if item.error is not None:
            return LoadedPlugin(item.source, "", item.source, ERROR, item.error)
        info = getattr(item.module, "BICTOOL_PLUGIN", None)
        if not isinstance(info, PluginInfo):
            return LoadedPlugin(item.source, "", item.source, ERROR, "no BICTOOL_PLUGIN = PluginInfo(...) in the module")
        name = info.name.strip()

        def status(state: str, message: str = "", hooks: tuple[str, ...] = ()) -> LoadedPlugin:
            return LoadedPlugin(name, info.version, item.source, state, message, info.description, hooks)

        if not name or "." in name:
            return status(ERROR, "the plug-in name must not be empty or contain '.'")
        if name in self.versions or any(p.name == name for p in self.plugins):
            return status(ERROR, f"another plug-in is already called {name!r}")
        try:
            compatible = SpecifierSet(info.requires_api).contains(API_VERSION, prereleases=True)
        except InvalidSpecifier:
            return status(ERROR, f"requires_api {info.requires_api!r} is not a version range like '>=0.2,<1'")
        if not compatible:
            return status(INCOMPATIBLE, f"needs API {info.requires_api}, this BICTool has {API_VERSION}")
        if name in disabled:
            return status(DISABLED, "switched off in Settings > Plug-ins")
        try:
            self._pm.register(item.module, name=name)
        except Exception as exc:  # e.g. a hook with unknown arguments (pluggy.PluginValidationError)
            return status(ERROR, _error_text(exc))
        hooks = tuple(h for h in HOOKS if any(i.plugin_name == name for i in getattr(self._pm.hook, h).get_hookimpls()))
        unused = _unused_functions(item.module)
        if not hooks:
            self._pm.unregister(name=name)
            return status(ERROR, "; ".join([f"implements none of the hooks {', '.join(HOOKS)}", *unused]))
        self.versions[name] = info.version
        message = self._add_tissue_methods(name) if "bictool_tissue_methods" in hooks else ""
        formats = self._add_export_formats(name) if "bictool_export_formats" in hooks else ""
        return status(ACTIVE, "; ".join(m for m in (*unused, message, formats) if m), hooks)

    def _add_tissue_methods(self, name: str) -> str:
        values = self._call(self._pm.hook.bictool_tissue_methods, name, {})
        if isinstance(values, str):
            return f"WM/GM methods not available - {values}"
        problems = []
        for method in values or []:
            if not isinstance(method, TissueMethod) or not callable(method.classify) or not method.id or "." in method.id:
                problems.append(f"not a valid TissueMethod: {method!r}")
                continue
            if method.settings is not None and not dataclasses.is_dataclass(method.settings):
                problems.append(f"settings of {method.id!r} must be a dataclass instance")
                continue
            self.tissue_methods[f"{name}.{method.id}"] = method
        return "; ".join(problems)

    def _add_export_formats(self, name: str) -> str:
        values = self._call(self._pm.hook.bictool_export_formats, name, {})
        if isinstance(values, str):
            return f"export formats not available - {values}"
        problems = []
        for fmt in values or []:
            if not isinstance(fmt, ExportFormat) or not callable(fmt.write) or not fmt.id or "." in fmt.id \
                    or not str(fmt.label).strip():
                problems.append(f"not a valid ExportFormat: {_short(fmt)}")
                continue
            if not isinstance(fmt.extension, str) or not _EXTENSION.match(fmt.extension):
                problems.append(f"export format {fmt.id!r}: extension {fmt.extension!r} is not like '.json'")
                continue
            self.export_formats[f"{name}.{fmt.id}"] = fmt
        return "; ".join(problems)

    def write_export(self, format_id: str, result, path) -> str | None:
        """Writes `result` with the plug-in format `format_id` to `path`; None on
        success, else what went wrong (a failing plug-in must not stop the export)."""
        fmt = self.export_formats.get(format_id)
        if fmt is None:
            return f"export format {format_id!r} is not available (plug-in missing or switched off)"
        try:
            fmt.write(result, Path(path))
        except Exception as exc:  # a plug-in must not stop the other files
            logger.warning("Plug-in export %s failed: %s", format_id, exc, exc_info=True)
            return _error_text(exc)
        return None

    # ---- calling hooks ----

    def _call(self, hook, plugin: str, kwargs: dict):
        """One plug-in's implementation of `hook`; an error comes back as its text."""
        for impl in hook.get_hookimpls():
            if impl.plugin_name != plugin:
                continue
            try:
                return impl.function(**{arg: kwargs[arg] for arg in impl.argnames})
            except Exception as exc:  # a plug-in must not stop the study
                logger.warning("Plug-in %s failed in %s: %s", plugin, hook.name, exc, exc_info=True)
                return _error_text(exc)
        return None

    def _metrics(self, hook, kwargs: dict, slice_index: int | None, shape_of: ShapeOf) -> tuple[PluginValues, ...]:
        out = []
        for impl in hook.get_hookimpls():
            name = impl.plugin_name
            result = self._call(hook, name, kwargs)
            if isinstance(result, str):
                out.append(PluginValues(name, error=result))
                continue
            values, markers, problem = _check_values(result, name, slice_index, shape_of)
            out.append(PluginValues(name, values, problem, markers))
        return tuple(sorted(out, key=lambda v: v.plugin))

    @property
    def has_pair_metrics(self) -> bool:
        return bool(self._pm.hook.bictool_pair_metrics.get_hookimpls())

    @property
    def has_slice_metrics(self) -> bool:
        return bool(self._pm.hook.bictool_slice_metrics.get_hookimpls())

    @property
    def has_study_metrics(self) -> bool:
        return bool(self._pm.hook.bictool_study_metrics.get_hookimpls())

    @property
    def has_artefact_hints(self) -> bool:
        return bool(self._pm.hook.bictool_artefact_hints.get_hookimpls())

    @property
    def has_measurement_hooks(self) -> bool:
        """Does any plug-in want the measured pairs (a metric or hint hook)?"""
        return self.has_pair_metrics or self.has_slice_metrics or self.has_study_metrics or self.has_artefact_hints

    def pair_metrics(self, context: PairContext) -> tuple[PluginValues, ...]:
        shape = context.image.shape
        return self._metrics(
            self._pm.hook.bictool_pair_metrics, {"pair": context}, context.slice_index,
            lambda index: shape if index == context.slice_index else None,
        )

    def slice_metrics(self, context: SliceContext) -> tuple[PluginValues, ...]:
        shape = context.image.shape
        return self._metrics(
            self._pm.hook.bictool_slice_metrics, {"ct_slice": context}, context.slice_index,
            lambda index: shape if index == context.slice_index else None,
        )

    def study_metrics(self, context: StudyContext) -> tuple[PluginValues, ...]:
        def shape_of(index):
            return context.image(index).shape if 0 <= index < context.num_slices else None

        return self._metrics(self._pm.hook.bictool_study_metrics, {"study": context}, None, shape_of)

    def artefact_hints(self, context: StudyContext) -> tuple[PluginHints, ...]:
        hook, out = self._pm.hook.bictool_artefact_hints, []
        for impl in hook.get_hookimpls():
            name = impl.plugin_name
            result = self._call(hook, name, {"study": context})
            if isinstance(result, str):
                out.append(PluginHints(name, error=result))
                continue
            hints, problem = _check_hints(result, context.num_slices)
            out.append(PluginHints(name, hints, problem))
        return tuple(sorted(out, key=lambda v: v.plugin))

    def tissue_method(self, method_id: str) -> TissueMethod | None:
        return self.tissue_methods.get(method_id)


def _unused_functions(module) -> list[str]:
    """Notes on `bictool_*` functions BICTool will never call (pluggy ignores them
    silently): a hook this version doesn't have (a typo, or made for a newer
    BICTool), or a hook without the `@hookimpl` mark."""
    notes = []
    for attr in sorted(dir(module)):
        if not attr.startswith("bictool_") or not callable(getattr(module, attr, None)):
            continue
        marked = getattr(getattr(module, attr), f"{PROJECT}_impl", None) is not None
        if attr not in HOOKS and marked:
            notes.append(f"{attr} is not a hook of this BICTool (API {API_VERSION}) - ignored")
        elif attr in HOOKS and not marked:
            notes.append(f"{attr} is not marked @plugins.hookimpl - ignored")
    return notes


def _check_values(
    result, plugin: str = "", slice_index: int | None = None, shape_of: ShapeOf = lambda index: None
) -> tuple[tuple[MetricValue, ...], tuple[PluginMarker, ...], str | None]:
    """The plug-in's values and markers, keeping only valid ones (with a note on the rest)."""
    if result is None:
        return (), (), None
    if not isinstance(result, (list, tuple)):
        return (), (), f"returned {type(result).__name__}, not a list of MetricValue"
    values, markers, problems, names = [], [], [], set()
    for item in result:
        if isinstance(item, RoiMarker):
            marker, problem = _check_marker(item, plugin, slice_index, shape_of)
            if problem:
                problems.append(problem)
            else:
                markers.append(marker)
            continue
        if not isinstance(item, MetricValue) or not item.name or item.name in names:
            problems.append(f"invalid or repeated value {_short(item)}")
            continue
        try:
            value = float(item.value)
        except (TypeError, ValueError):
            problems.append(f"{item.name}: value {item.value!r} is not a number")
            continue
        names.add(item.name)
        values.append(dataclasses.replace(item, value=value if math.isfinite(value) else float("nan")))
    return tuple(values), tuple(markers), ("; ".join(problems) or None)


def _short(item) -> str:
    text = repr(item)
    return text if len(text) <= 80 else text[:77] + "..."


def _check_marker(
    item: RoiMarker, plugin: str, slice_index: int | None, shape_of: ShapeOf
) -> tuple[PluginMarker | None, str | None]:
    """A `RoiMarker` checked and cropped to its bounding box."""
    if not isinstance(item.name, str) or not item.name.strip():
        return None, "a marker needs a name"
    index = item.slice_index if item.slice_index is not None else slice_index
    if index is None:
        return None, f"marker {item.name!r}: slice_index is missing"
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        return None, f"marker {item.name!r}: slice_index {index!r} is not a whole number"
    shape = shape_of(int(index))
    if shape is None:
        return None, f"marker {item.name!r}: slice {index} is not allowed here"
    try:
        mask = np.asarray(item.mask)
    except Exception:  # an object NumPy can't convert
        return None, f"marker {item.name!r}: the mask is not an array"
    if mask.shape != tuple(shape) or not (mask.dtype == bool or np.issubdtype(mask.dtype, np.number)):
        return None, f"marker {item.name!r}: the mask must be a bool array of shape {tuple(shape)}"
    mask = mask.astype(bool)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None, f"marker {item.name!r}: the mask is empty"
    box = RoiRect(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    crop = mask[box.y0 : box.y1, box.x0 : box.x1].copy()
    crop.flags.writeable = False
    return PluginMarker(plugin, item.name.strip(), int(index), box, crop, str(item.description or "")), None


def _check_hints(result, num_slices: int) -> tuple[tuple[ArtefactHint, ...], str | None]:
    """The plug-in's artefact hints, keeping only valid ones: one line of text
    (at most MAX_HINT_LENGTH characters) and existing slices."""
    if result is None:
        return (), None
    if not isinstance(result, (list, tuple)):
        return (), f"returned {type(result).__name__}, not a list of ArtefactHint"
    hints, problems = [], []
    for item in result:
        if not isinstance(item, ArtefactHint) or not isinstance(item.text, str) or not item.text.strip():
            problems.append(f"invalid hint {_short(item)}")
            continue
        text = " ".join(item.text.split())
        if len(text) > MAX_HINT_LENGTH:
            problems.append(f"hint longer than {MAX_HINT_LENGTH} characters: {text[:40]}...")
            continue
        try:
            slices = tuple(sorted({int(i) for i in item.slices}))
        except (TypeError, ValueError):
            problems.append(f"hint {text[:40]!r}: slices must be whole numbers")
            continue
        if any(not 0 <= i < num_slices for i in slices):
            problems.append(f"hint {text[:40]!r}: a slice outside 0..{num_slices - 1}")
            continue
        hints.append(ArtefactHint(text, slices))
    return tuple(hints), ("; ".join(problems) or None)


# ---- the manager the pipeline uses ----

_current: PluginManager | None = None
_lock = threading.Lock()


def load_plugins(folders=None, entry_points: bool = True, disabled: Iterable[str] = ()) -> PluginManager:
    """A manager with the plug-ins found in `folders` (default: `app_paths.plugin_folders()`)
    and, with `entry_points`, the installed ones."""
    from bictool.infrastructure.app_paths import plugin_folders
    from bictool.infrastructure.plugin_loader import find_plugins

    manager = PluginManager()
    manager.load(find_plugins(plugin_folders() if folders is None else folders, entry_points), disabled)
    return manager


def use(manager: PluginManager | None) -> None:
    """The manager the pipeline uses from now on (None: load the default on next use)."""
    global _current
    with _lock:
        _current = manager


def current() -> PluginManager:
    """The manager in use - on first use, every installed and folder plug-in."""
    global _current
    with _lock:
        if _current is None:
            _current = load_plugins()
        return _current
