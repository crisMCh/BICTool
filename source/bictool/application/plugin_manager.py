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
import threading
from collections.abc import Iterable

import pluggy
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from bictool import API_VERSION
from bictool.application.plugin_types import (
    HOOKS,
    PROJECT,
    HookSpecs,
    LoadedPlugin,
    MetricValue,
    PairContext,
    PluginInfo,
    PluginValues,
    StudyContext,
    TissueMethod,
)
from bictool.infrastructure.plugin_loader import FoundPlugin

logger = logging.getLogger(__name__)

ACTIVE, DISABLED, INCOMPATIBLE, ERROR = "active", "disabled", "incompatible", "error"


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
        if not hooks:
            self._pm.unregister(name=name)
            return status(ERROR, f"implements none of the hooks {', '.join(HOOKS)}")
        self.versions[name] = info.version
        message = self._add_tissue_methods(name) if "bictool_tissue_methods" in hooks else ""
        return status(ACTIVE, message, hooks)

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

    def _metrics(self, hook, kwargs: dict) -> tuple[PluginValues, ...]:
        out = []
        for impl in hook.get_hookimpls():
            name = impl.plugin_name
            result = self._call(hook, name, kwargs)
            if isinstance(result, str):
                out.append(PluginValues(name, error=result))
                continue
            values, problem = _check_values(result)
            out.append(PluginValues(name, values, problem))
        return tuple(sorted(out, key=lambda v: v.plugin))

    @property
    def has_pair_metrics(self) -> bool:
        return bool(self._pm.hook.bictool_pair_metrics.get_hookimpls())

    @property
    def has_study_metrics(self) -> bool:
        return bool(self._pm.hook.bictool_study_metrics.get_hookimpls())

    def pair_metrics(self, context: PairContext) -> tuple[PluginValues, ...]:
        return self._metrics(self._pm.hook.bictool_pair_metrics, {"pair": context})

    def study_metrics(self, context: StudyContext) -> tuple[PluginValues, ...]:
        return self._metrics(self._pm.hook.bictool_study_metrics, {"study": context})

    def tissue_method(self, method_id: str) -> TissueMethod | None:
        return self.tissue_methods.get(method_id)


def _check_values(result) -> tuple[tuple[MetricValue, ...], str | None]:
    """The plug-in's values, keeping only valid ones (with a note on the rest)."""
    if result is None:
        return (), None
    if not isinstance(result, (list, tuple)):
        return (), f"returned {type(result).__name__}, not a list of MetricValue"
    values, problems, names = [], [], set()
    for item in result:
        if not isinstance(item, MetricValue) or not item.name or item.name in names:
            problems.append(f"invalid or repeated value {item!r}")
            continue
        try:
            value = float(item.value)
        except (TypeError, ValueError):
            problems.append(f"{item.name}: value {item.value!r} is not a number")
            continue
        names.add(item.name)
        values.append(dataclasses.replace(item, value=value if math.isfinite(value) else float("nan")))
    return tuple(values), ("; ".join(problems) or None)


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
