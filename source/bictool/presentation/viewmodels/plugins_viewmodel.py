"""PluginsViewModel: what Settings > Plug-ins shows and changes.

The plug-ins found at start (`plugin_manager`), their status in words, and
which are switched off. Switching takes effect when BICTool is restarted
(plug-in code is loaded once, at start). Pure Python - no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUS_TEXT = {
    "active": "Active",
    "disabled": "Switched off",
    "incompatible": "Not compatible",
    "error": "Error",
}


@dataclass(frozen=True)
class PluginRow:
    name: str
    version: str
    status: str          # in words
    details: str         # the problem, else the description
    source: str
    switchable: bool     # a name to switch on/off (a plug-in that could not be read has none)
    enabled: bool


def parse_disabled(text: str) -> set[str]:
    """The stored list ("a,b") as names."""
    return {name.strip() for name in text.split(",") if name.strip()}


def format_disabled(names: set[str]) -> str:
    return ",".join(sorted(names))


class PluginsViewModel:
    def __init__(self, plugins, disabled: set[str], folders) -> None:
        """`plugins` = `LoadedPlugin`s found at start; `disabled` = the names switched
        off; `folders` = where plug-in files are looked for (the first is shown)."""
        self._plugins = list(plugins)
        self.disabled = set(disabled)
        self._initial = set(disabled)
        self.folders = list(folders)

    def rows(self) -> list[PluginRow]:
        rows = []
        for p in sorted(self._plugins, key=lambda p: p.name.lower()):
            switchable = p.status in ("active", "disabled") or (p.status == "incompatible" and bool(p.version))
            rows.append(PluginRow(
                name=p.name, version=p.version, status=STATUS_TEXT.get(p.status, p.status),
                details=p.message or p.description, source=p.source,
                switchable=switchable, enabled=p.name not in self.disabled,
            ))
        return rows

    def set_enabled(self, name: str, enabled: bool) -> None:
        if enabled:
            self.disabled.discard(name)
        else:
            self.disabled.add(name)

    @property
    def changed(self) -> bool:
        return self.disabled != self._initial

    @property
    def folder(self):
        """The plug-in folder of this installation (where to copy a plug-in)."""
        return self.folders[0] if self.folders else None

    def summary(self) -> str:
        active = sum(1 for p in self._plugins if p.status == "active")
        if not self._plugins:
            return "No plug-ins found."
        return f"{len(self._plugins)} plug-in(s) found, {active} active."
