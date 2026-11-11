"""Writing a BICTool plug-in (FR-08) - everything a plug-in imports.

A plug-in is a Python module (or package) with a `BICTOOL_PLUGIN` description
and one or more hook functions marked with `@hookimpl`:

    from bictool.api import plugins

    BICTOOL_PLUGIN = plugins.PluginInfo(
        name="roi-stats", version="1.0", requires_api=">=0.3,<1",
        description="Mean and SD of both ROIs",
    )

    @plugins.hookimpl
    def bictool_pair_metrics(pair):
        return [plugins.MetricValue("roi2_sd", float(pair.roi2_pixels.std()), "HU")]

BICTool finds plug-ins installed as Python packages (entry point group
`bictool.plugins`) and modules copied into a plug-in folder (`plugins/` next
to BICTool.exe; `~/.bictool/plugins` or `BICTOOL_PLUGIN_PATH` when run from
source). The extension points ("hooks") are listed in `HOOKS`; docs/developer (reference/hooks.md)
describes them. New hooks may be added in later versions - a plug-in only
implements those it needs.

Rules: images and masks handed to a plug-in are read-only; hooks may run on
a background thread; a hook that raises is recorded in the result and does not
stop the study; plug-in values never change the study verdict.
"""

from bictool.application.plugin_types import (
    ArtefactHint,
    CSF,
    ENTRY_POINT_GROUP,
    ExportFormat,
    GRAY_MATTER,
    hookimpl,
    HOOKS,
    hookspec,
    HookSpecs,
    LABELS,
    LoadedPlugin,
    MetricValue,
    PairContext,
    PluginHints,
    PluginInfo,
    PluginMarker,
    PluginValues,
    PROJECT,
    RoiMarker,
    SliceContext,
    StudyContext,
    TissueContext,
    TissueMethod,
    WHITE_MATTER,
)
from bictool.application.roi_store import RoiRect

__all__ = [
    "PROJECT",
    "ENTRY_POINT_GROUP",
    "hookimpl",
    "hookspec",
    "LABELS",
    "WHITE_MATTER",
    "GRAY_MATTER",
    "CSF",
    "PluginInfo",
    "MetricValue",
    "RoiMarker",
    "ArtefactHint",
    "RoiRect",
    "PairContext",
    "SliceContext",
    "StudyContext",
    "TissueContext",
    "TissueMethod",
    "ExportFormat",
    "PluginValues",
    "PluginMarker",
    "PluginHints",
    "LoadedPlugin",
    "HookSpecs",
    "HOOKS",
]
