"""Plug-in types and extension points (hook specifications) - internal home;
plug-in authors import them from `bictool.api.plugins` (see its docstring and
docs/developer/reference/plugin-types.md). Pure Python, no Qt."""

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
class RoiMarker:
    """Where a plug-in measured - BICTool draws it on the image (Run QA > "Plug-in ROIs").
    A metric hook returns markers next to its `MetricValue`s. `mask` is a bool array
    of the slice's shape (any shape of ROI); `slice_index` is needed only in
    `bictool_study_metrics` (the other hooks use their slice)."""

    name: str
    mask: np.ndarray
    slice_index: int | None = None
    description: str = ""


@dataclass(frozen=True)
class PluginMarker:
    """A checked `RoiMarker` as BICTool keeps it: the mask cropped to `box`."""

    plugin: str
    name: str
    slice_index: int
    box: RoiRect                         # x0, y0, x1, y1 of the cropped mask
    mask: np.ndarray                     # bool, (box rows, box columns)
    description: str = ""

    def full_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """The marker as a bool array of the slice's `shape`."""
        out = np.zeros(shape, dtype=bool)
        out[self.box.y0 : self.box.y1, self.box.x0 : self.box.x1] = self.mask
        return out


@dataclass(frozen=True)
class ArtefactHint:
    """A possible artefact a plug-in noticed (`bictool_artefact_hints`): one neutral
    line of text and the slices (0-based) it concerns. Descriptive - never a score,
    never part of the verdict."""

    text: str
    slices: tuple[int, ...] = ()


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
class SliceContext:
    """One measured slice (input of `bictool_slice_metrics`), after its ROI pairs.
    Arrays are read-only. `brain_mask()` and `tissue()` are computed on first use:
    BICTool's brain mask of the slice, and its WM/GM/CSF masks
    ({WHITE_MATTER: mask, GRAY_MATTER: mask, CSF: mask}) from the WM/GM method
    `tissue_method` - the one chosen in the program or in `api.Settings`."""

    slice_index: int                     # 0-based
    image: np.ndarray                    # the slice in HU (rows, columns)
    pixel_spacing_mm: float
    labels: Mapping[str, np.ndarray] | None   # the CNN's masks (`LABELS`); None if the slice is not segmented
    pairs: tuple[PairContext, ...]       # BICTool's ROI pairs measured on this slice
    brain_mask: Callable[[], np.ndarray]
    tissue: Callable[[], Mapping[str, np.ndarray]]
    tissue_method: str


@dataclass(frozen=True)
class StudyContext:
    """The whole series after the measurement (input of `bictool_study_metrics` and
    `bictool_artefact_hints`). `image(i)`, `labels(i)`, `brain_mask(i)` and
    `tissue(i)` compute on demand (as in `SliceContext`); arrays are read-only."""

    num_slices: int
    pixel_spacing_mm: float
    slice_thickness_mm: float | None
    segmented_slices: tuple[int, ...]
    pairs: tuple[PairContext, ...]       # every measured ROI pair
    image: Callable[[int], np.ndarray]
    labels: Callable[[int], Mapping[str, np.ndarray] | None]
    brain_mask: Callable[[int], np.ndarray]
    tissue: Callable[[int], Mapping[str, np.ndarray]]
    tissue_method: str
    measured_slices: tuple[int, ...] = ()   # slices with at least one measured ROI pair


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
class ExportFormat:
    """A file format a plug-in writes a study result in (`bictool_export_formats`).
    The program offers it in the Export dialog (whole study) as `label`; the API
    calls it with `api.write_export`. `write(result, path)` writes `result` (an
    `api.StudyResult`) to `path` (a `pathlib.Path` ending with `extension`, e.g.
    ".json", chosen by the user). `id` is unique within the plug-in; BICTool
    prefixes it with the plug-in's name ("<plug-in>.<id>")."""

    id: str
    label: str
    extension: str
    write: Callable[..., None]
    description: str = ""


@dataclass(frozen=True)
class PluginValues:
    """What one plug-in reported for one ROI pair, slice or study."""

    plugin: str
    values: tuple[MetricValue, ...] = ()
    error: str | None = None
    markers: tuple[PluginMarker, ...] = ()


@dataclass(frozen=True)
class PluginHints:
    """The possible artefacts one plug-in noticed in a study."""

    plugin: str
    hints: tuple[ArtefactHint, ...] = ()
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
    function with the same name (it may omit arguments it does not need).
    The metric hooks may return `RoiMarker`s among their values."""

    @hookspec
    def bictool_pair_metrics(self, pair: PairContext) -> list[MetricValue | RoiMarker] | None:
        """Extra values for one measured ROI pair (called once per slice and pair)."""

    @hookspec
    def bictool_slice_metrics(self, ct_slice: SliceContext) -> list[MetricValue | RoiMarker] | None:
        """Extra values for one measured slice, e.g. from the plug-in's own ROIs
        (called once per slice with a measured ROI pair, after its pairs)."""

    @hookspec
    def bictool_study_metrics(self, study: StudyContext) -> list[MetricValue | RoiMarker] | None:
        """Extra values for the whole series (called once, after all pairs and slices)."""

    @hookspec
    def bictool_artefact_hints(self, study: StudyContext) -> list[ArtefactHint] | None:
        """Possible artefacts the plug-in noticed (called once, with the study metrics)."""

    @hookspec
    def bictool_tissue_methods(self) -> list[TissueMethod] | None:
        """WM/GM methods this plug-in adds (called once when the plug-in is loaded)."""

    @hookspec
    def bictool_export_formats(self) -> list[ExportFormat] | None:
        """File formats this plug-in writes study results in (called once when the plug-in is loaded)."""


HOOKS = tuple(name for name in vars(HookSpecs) if name.startswith("bictool_"))
