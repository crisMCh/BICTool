"""Settings of one QA run: the QA recipe and the WM/GM method with its parameters.

The WM/GM method only decides where the falx pair's ROI2 lands (white
matter next to the falx). Methods are identified by short stable ids
("adaptive", ...; a plug-in's method is "<plug-in>.<id>"); `METHOD_LABELS`
gives what the program shows for the built-in ones. The three
parameter dataclasses are what the settings dialogs edit (immutable - a
change is a new object); `default_*()` give the program's defaults.
Public through `bictool.api`. Pure Python.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass

from bictool.domain.results.study_quality import DEFAULT_RECIPE, RECIPES
from bictool.domain.segmentation import brain_tissue_adaptive as adaptive
from bictool.domain.segmentation import brain_tissue_fcm as fcm
from bictool.domain.segmentation.brain_tissue import (
    DEFAULT_AIR_HU_CUTOFF,
    DEFAULT_BONE_HU_CUTOFF,
    DEFAULT_CSF_RANGE,
    DEFAULT_GRAY_MATTER_RANGE,
    DEFAULT_MIN_COMPONENT_SIZE,
    DEFAULT_SMOOTHING_SIZE,
    DEFAULT_WHITE_MATTER_RANGE,
)

METHOD_FCM = "fcm"
METHOD_THRESHOLD = "threshold"
METHOD_ADAPTIVE = "adaptive"
METHOD_ANCHORED = "anchored"
METHODS = [METHOD_FCM, METHOD_THRESHOLD, METHOD_ADAPTIVE, METHOD_ANCHORED]
METHOD_LABELS = {
    METHOD_FCM: "Fuzzy C-Means",
    METHOD_THRESHOLD: "HU Thresholding",
    METHOD_ADAPTIVE: "Adaptive Thresholding",
    METHOD_ANCHORED: "Anchored (prototype)",
}
DEFAULT_METHOD = METHOD_ADAPTIVE  # no method wins on every scan; this one works on most


def is_plugin_method(method: str) -> bool:
    """A plug-in's WM/GM method id ("<plug-in>.<id>")."""
    plugin, _, own = method.partition(".")
    return bool(plugin and own)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class BrainTissueSettings:
    """Every tunable parameter `brain_tissue.extract_brain_mask`/
    `classify_brain_tissue` accept (the HU Thresholding method)."""

    air_hu_cutoff: float
    bone_hu_cutoff: float
    csf_low: float
    csf_high: float
    white_matter_low: float
    white_matter_high: float
    gray_matter_low: float
    gray_matter_high: float
    smoothing_size: int
    min_component_size: int

    def __post_init__(self) -> None:
        _require(self.air_hu_cutoff < self.bone_hu_cutoff, "air_hu_cutoff must be below bone_hu_cutoff")
        for name in ("csf", "white_matter", "gray_matter"):
            _require(getattr(self, f"{name}_low") < getattr(self, f"{name}_high"), f"{name}_low must be below {name}_high")
        _require(self.smoothing_size >= 1, "smoothing_size must be at least 1")
        _require(self.min_component_size >= 0, "min_component_size must not be negative")


@dataclass(frozen=True)
class FcmSettings:
    fuzziness: float
    max_iter: int
    spatial_weight: float
    min_component_size: int

    def __post_init__(self) -> None:
        _require(self.fuzziness > 1, "fuzziness must be above 1")
        _require(self.max_iter >= 1, "max_iter must be at least 1")
        _require(self.spatial_weight >= 0, "spatial_weight must not be negative")
        _require(self.min_component_size >= 0, "min_component_size must not be negative")


@dataclass(frozen=True)
class AdaptiveSettings:
    block_size: int
    plausible_low: float           # fixed range - used when no study range is available (or it's off)
    plausible_high: float
    min_component_size: int
    smoothing_mm: float = 0.0      # Gaussian sigma before classifying (app default 1 mm)
    use_study_range: bool = False  # plausible range from the study's basal-ganglia references (app default on)

    def __post_init__(self) -> None:
        _require(self.block_size >= 3, "block_size must be at least 3")
        _require(self.plausible_low < self.plausible_high, "plausible_low must be below plausible_high")
        _require(self.min_component_size >= 0, "min_component_size must not be negative")
        _require(self.smoothing_mm >= 0, "smoothing_mm must not be negative")


def default_threshold_settings() -> BrainTissueSettings:
    return BrainTissueSettings(
        air_hu_cutoff=DEFAULT_AIR_HU_CUTOFF,
        bone_hu_cutoff=DEFAULT_BONE_HU_CUTOFF,
        csf_low=DEFAULT_CSF_RANGE[0],
        csf_high=DEFAULT_CSF_RANGE[1],
        white_matter_low=DEFAULT_WHITE_MATTER_RANGE[0],
        white_matter_high=DEFAULT_WHITE_MATTER_RANGE[1],
        gray_matter_low=DEFAULT_GRAY_MATTER_RANGE[0],
        gray_matter_high=DEFAULT_GRAY_MATTER_RANGE[1],
        smoothing_size=DEFAULT_SMOOTHING_SIZE,
        min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
    )


def default_fcm_settings() -> FcmSettings:
    return FcmSettings(
        fuzziness=fcm.DEFAULT_FUZZINESS,
        max_iter=fcm.DEFAULT_MAX_ITER,
        spatial_weight=fcm.DEFAULT_SPATIAL_WEIGHT,
        min_component_size=fcm.DEFAULT_MIN_COMPONENT_SIZE,
    )


def default_adaptive_settings() -> AdaptiveSettings:
    return AdaptiveSettings(
        block_size=adaptive.DEFAULT_BLOCK_SIZE,
        plausible_low=adaptive.DEFAULT_PLAUSIBLE_RANGE[0],
        plausible_high=adaptive.DEFAULT_PLAUSIBLE_RANGE[1],
        min_component_size=adaptive.DEFAULT_MIN_COMPONENT_SIZE,
        smoothing_mm=adaptive.DEFAULT_SMOOTHING_MM,
        use_study_range=adaptive.DEFAULT_USE_STUDY_RANGE,
    )


@dataclass(frozen=True)
class TissueSettings:
    """The WM/GM method and every method's parameters (only the chosen method's are
    used). A plug-in's method ("<plug-in>.<id>") takes its settings from `plugin`
    (None: the plug-in's defaults)."""

    method: str = DEFAULT_METHOD
    threshold: BrainTissueSettings = field(default_factory=default_threshold_settings)
    fcm: FcmSettings = field(default_factory=default_fcm_settings)
    adaptive: AdaptiveSettings = field(default_factory=default_adaptive_settings)
    plugin: object = None

    def __post_init__(self) -> None:
        _require(
            self.method in METHODS or is_plugin_method(self.method),
            f"Unknown WM/GM method {self.method!r} - one of {METHODS} or a plug-in's '<plug-in>.<id>'",
        )

    def parameters(self):
        """The chosen method's parameters (the prototype uses Adaptive's where it has no model)."""
        if is_plugin_method(self.method):
            return self.plugin
        return {
            METHOD_THRESHOLD: self.threshold,
            METHOD_FCM: self.fcm,
            METHOD_ADAPTIVE: self.adaptive,
            METHOD_ANCHORED: self.adaptive,
        }[self.method]


@dataclass(frozen=True)
class Settings:
    """Everything a QA run can be tuned with. The defaults are the program's."""

    recipe: str = DEFAULT_RECIPE
    tissue: TissueSettings = field(default_factory=TissueSettings)

    def __post_init__(self) -> None:
        _require(self.recipe in RECIPES, f"Unknown QA recipe {self.recipe!r} - one of {RECIPES}")

    def to_dict(self) -> dict:
        """Plain values (the chosen WM/GM method's parameters only)."""
        params = self.tissue.parameters()
        if is_dataclass(params):
            params = asdict(params)
        return {"recipe": self.recipe, "tissue_method": self.tissue.method, "tissue_parameters": dict(params or {})}
