"""Remembered user-interface choices (e.g. "show trace log", "ROI details
open"), per Windows/OS user, via `QSettings` (INI format, user scope).

Only view preferences live here - nothing that changes a measurement.
Tests redirect the storage location (`QSettings.setPath`, see
`tests/presentation/conftest.py`) so they never touch a real user's settings.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSettings

ORGANIZATION = "BfS"
APPLICATION = "BICTool"

SHOW_TRACE_LOG = "ui/show_trace_log"
ROI_DETAILS_OPEN = "ui/roi_details_open"
VERDICT_DETAILS_OPEN = "ui/verdict_details_open"
MAIN_SPLITTER = "ui/main_splitter"        # viewer | sidebar widths
TRACE_SPLITTER = "ui/trace_splitter"      # content | trace log heights


def section_key(name: str) -> str:
    """Expanded/collapsed state of a sidebar section (default: expanded)."""
    return f"ui/section/{name}"


_DEFAULTS = {
    SHOW_TRACE_LOG: False,       # end users see a clean window; developers switch it on once
    ROI_DETAILS_OPEN: False,
    VERDICT_DETAILS_OPEN: False,
}


class UiSettings:
    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, ORGANIZATION, APPLICATION
        )

    def get_bool(self, key: str, default: bool | None = None) -> bool:
        value = self._settings.value(key, _DEFAULTS.get(key, False) if default is None else default)
        if isinstance(value, str):  # INI files store booleans as text
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    def set_bool(self, key: str, value: bool) -> None:
        self._settings.setValue(key, bool(value))
        self._settings.sync()

    def get_str(self, key: str, default: str = "") -> str:
        value = self._settings.value(key, default)
        return str(value) if value is not None else default

    def set_str(self, key: str, value: str) -> None:
        self._settings.setValue(key, str(value))
        self._settings.sync()

    def get_bytes(self, key: str) -> QByteArray | None:
        value = self._settings.value(key)
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def set_bytes(self, key: str, value: QByteArray) -> None:
        self._settings.setValue(key, value)
        self._settings.sync()
