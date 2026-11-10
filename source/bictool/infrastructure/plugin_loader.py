"""Finds plug-ins and imports their code: Python packages registered under the
entry point group `bictool.plugins`, and `.py` files or package folders in the
plug-in folders (`app_paths.plugin_folders`). A plug-in that fails to import
is reported, not raised - it must never stop the program.

Plug-in code runs inside BICTool with BICTool's rights: only code someone
deliberately installed or copied into a plug-in folder is loaded.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "bictool.plugins"
_MODULE_PREFIX = "bictool_plugins"


@dataclass(frozen=True)
class FoundPlugin:
    """A plug-in's code, or why it could not be imported."""

    source: str                      # a file, a folder or "package <distribution>: <entry point>"
    module: object | None = None     # the imported module (or object an entry point names)
    error: str | None = None


def _error_text(exc: BaseException) -> str:
    last = traceback.extract_tb(exc.__traceback__)[-1:] if exc.__traceback__ else []
    where = f" ({Path(last[0].filename).name}, line {last[0].lineno})" if last else ""
    return f"{type(exc).__name__}: {exc}{where}"


def _import_file(path: Path) -> ModuleType:
    """Imports a `.py` file or a package folder under a private module name."""
    name = f"{_MODULE_PREFIX}.{path.stem if path.is_file() else path.name}"
    if path.is_dir():
        spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    else:
        spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _folder_candidates(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        entry for entry in folder.iterdir()
        if not entry.name.startswith(("_", "."))
        and ((entry.is_file() and entry.suffix == ".py") or (entry.is_dir() and (entry / "__init__.py").is_file()))
    )


def find_plugins(folders: list[Path] | None = None, entry_points: bool = True) -> list[FoundPlugin]:
    """Every plug-in in `folders` and (with `entry_points`) every installed one."""
    found = []
    if entry_points:
        for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            source = f"package {entry.dist.name if entry.dist else '?'}: {entry.value}"
            try:
                found.append(FoundPlugin(source, module=entry.load()))
            except Exception as exc:  # a broken plug-in must not stop BICTool
                logger.warning("Plug-in %s could not be loaded: %s", source, exc)
                found.append(FoundPlugin(source, error=_error_text(exc)))
    for folder in folders or []:
        for path in _folder_candidates(Path(folder)):
            try:
                found.append(FoundPlugin(str(path), module=_import_file(path)))
            except Exception as exc:  # a broken plug-in must not stop BICTool
                logger.warning("Plug-in %s could not be loaded: %s", path, exc)
                found.append(FoundPlugin(str(path), error=_error_text(exc)))
    return found
