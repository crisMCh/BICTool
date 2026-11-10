"""Plug-in types and extension points (hook specifications) - internal home;
plug-in authors import them from `bictool.api.plugins` (see its docstring and
docs/plugins.md). Pure Python, no Qt."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pluggy

from bictool.application.roi_store import RoiRect
from bictool.infrastructure.plugin_loader import ENTRY_POINT_GROUP  # noqa: F401 - re-exported

PROJECT = "bictool"
hookimpl = pluggy.HookimplMarker(PROJECT)
hookspec = pluggy.HookspecMarker(PROJECT)

# The anatomy CNN's labels (keys of `PairContext.labels`).
LABELS = (
    "Brain Parenchyma", "Subarachnoid Space", "Dural Folds", "Septum Pellucidum", "Cerebellum", "Caudate Nucleus",
    "Lentiform Nucleus", "Insular Cortex", "Internal Capsule", "Ventricular System", "Central Sulcus",
    "Frontal Lobe", "Parietal Lobe", "Occipital Lobe", "Temporal Lobe", "Thalamus",
)
# The masks a WM/GM method returns (`TissueMethod.classify`).
WHITE_MATTER, GRAY_MATTER, CSF = "White Matter", "Gray Matter", "CSF"


@dataclass(frozen=True)
class PluginInfo:
    """A plug-in's description (module attribute `BICTOOL_PLUGIN`). `requires_api`
    is a version specifier on `bictool.api.API_VERSION`, e.g. ">=0.2,<1"; a plug-in
    whose range does not include the running API is not loaded."""

    name: str
    version: str
    requires_api: str
    description: str = ""
    author: str = ""


@dataclass(frozen=True)
class MetricValue:
    """One value a metric hook reports. `name` is unique within the plug-in;
    BICTool shows and exports it as "<plug-in>: <name>"."""

    name: str
    value: float
    unit: str = ""
    description: str = ""


@dataclass(frozen=True)
class PairContext:
    """One measured ROI pair (input of `bictool_pair_metrics`). Arrays are read-only."""

    slice_index: int                     # 0-based
    pair: str                            # "falx" or "septum"
    image: np.ndarray                    # the slice in HU (rows, columns)
    pixel_spacing_mm: float
    roi1: RoiRect                        # on the structure
    roi2: RoiRect                        # in homogeneous tissue next to it
    labels: Mapping[str, np.ndarray] | None   # the CNN's masks (`LABELS`); None if the slice is not segmented
    bictool_values: Mapping[str, float]  # BICTool's own measurement (mtf_area, nps_homo, ..., `api.Measurement` names)

    @property
    def roi1_pixels(self) -> np.ndarray:
        return self.image[self.roi1.y0 : self.roi1.y1, self.roi1.x0 : self.roi1.x1]

    @property
    def roi2_pixels(self) -> np.ndarray:
        return self.image[self.roi2.y0 : self.roi2.y1, self.roi2.x0 : self.roi2.x1]


@dataclass(frozen=True)
class StudyContext:
    """The whole series after the measurement (input of `bictool_study_metrics`).
    `image(i)` / `labels(i)` decode on demand; arrays are read-only."""

    num_slices: int
    pixel_spacing_mm: float
    slice_thickness_mm: float | None
    segmented_slices: tuple[int, ...]
    pairs: tuple[PairContext, ...]       # every measured ROI pair
    image: Callable[[int], np.ndarray]
    labels: Callable[[int], Mapping[str, np.ndarray] | None]


@dataclass(frozen=True)
class TissueContext:
    """Extra input of a WM/GM method: where the slice is and its CNN masks (None if not segmented)."""

    slice_index: int | None
    pixel_spacing_mm: float | None
    labels: Mapping[str, np.ndarray] | None


@dataclass(frozen=True)
class TissueMethod:
    """A WM/GM method a plug-in adds (`bictool_tissue_methods`). It appears in the
    program's Method drop-down as `label`; its settings dialog is built from the
    fields of `settings` (a frozen dataclass instance = the defaults; int, float
    and bool fields get an input each). `classify(slice_hu, brain_mask, settings,
    context)` returns {WHITE_MATTER: mask, GRAY_MATTER: mask, CSF: mask}, each a
    bool array of the slice's shape inside `brain_mask`.

    `id` is unique and is what `api.TissueSettings(method=...)` takes; BICTool
    prefixes it with the plug-in's name ("<plug-in>.<id>")."""

    id: str
    label: str
    classify: Callable[..., Mapping[str, np.ndarray]]
    settings: object = None
    description: str = ""


@dataclass(frozen=True)
class PluginValues:
    """What one plug-in reported for one measurement or study."""

    plugin: str
    values: tuple[MetricValue, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class LoadedPlugin:
    """A plug-in BICTool found (Settings > Plug-ins): `status` is "active",
    "disabled", "incompatible" or "error" (then `message` says why)."""

    name: str
    version: str
    source: str                  # where it came from: a file, a folder or a package
    status: str
    message: str = ""
    description: str = ""
    hooks: tuple[str, ...] = field(default=())


class HookSpecs:
    """The extension points. A plug-in implements any of them as a module-level
    function with the same name (it may omit arguments it does not need)."""

    @hookspec
    def bictool_pair_metrics(self, pair: PairContext) -> list[MetricValue] | None:
        """Extra values for one measured ROI pair (called once per slice and pair)."""

    @hookspec
    def bictool_study_metrics(self, study: StudyContext) -> list[MetricValue] | None:
        """Extra values for the whole series (called once, after all pairs are measured)."""

    @hookspec
    def bictool_tissue_methods(self) -> list[TissueMethod] | None:
        """WM/GM methods this plug-in adds (called once when the plug-in is loaded)."""


HOOKS = tuple(name for name in vars(HookSpecs) if name.startswith("bictool_"))
