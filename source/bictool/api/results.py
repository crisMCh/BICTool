"""What a QA run returns: `StudyResult` and the types it is made of.

These are the API's own frozen dataclasses - they are the result layout
(`SCHEMA_VERSION`), independent of the program's internal classes, so the
internals can change without changing what callers get. `to_dict()` /
`from_dict()` convert them field by field.

Conventions: slice indices are 0-based (the program shows index + 1); values
that could not be computed are NaN in Python and null in JSON; categorical
values are short ids ("falx", "white_matter", "not_acceptable", ...); units
are in `UNITS`. A result carries no patient name, ID or birth date. The
DICOM UIDs (study, series, image) are kept for integration but left out of
`to_dict()` unless asked for.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import typing
from dataclasses import dataclass, field

import numpy as np

from bictool.api._version import API_VERSION
from bictool.application.plugin_types import MetricValue, PluginValues  # noqa: F401 - type hints
from bictool.application.qa_settings import (
    AdaptiveSettings,
    BrainTissueSettings,
    FcmSettings,
    Settings,
    TissueSettings,
)
from bictool.application.roi_store import PAIR_STRUCTURES, PAIRS, RoiRect
from bictool.domain.results import head_ct_verdict as verdicts
from bictool.domain.results.volume_qa import StructureQa, VolumeQa
from bictool.domain.roi_selection import homogeneous_roi
from bictool.domain.roi_selection.structured_roi import STRUCTURES, RoiCandidate

# The minor number grows when keys are added, the major number when a key is renamed or removed.
# 1.1 (API 0.2): plug-in values.
SCHEMA_VERSION = "1.1"

ACCEPTABLE, NOT_ACCEPTABLE, UNCERTAIN, NOT_ASSESSABLE = "acceptable", "not_acceptable", "uncertain", "not_assessable"
_VERDICT_IDS = {
    verdicts.ACCEPTABLE: ACCEPTABLE,
    verdicts.NOT_ACCEPTABLE: NOT_ACCEPTABLE,
    verdicts.UNCERTAIN: UNCERTAIN,
    verdicts.NOT_ASSESSABLE: NOT_ASSESSABLE,
}
VENTRICLE, WHITE_MATTER, FALLBACK = "ventricle", "white_matter", "fallback"
_REGION_IDS = {homogeneous_roi.VENTRICLE: VENTRICLE, homogeneous_roi.WHITE_MATTER: WHITE_MATTER, homogeneous_roi.FALLBACK: FALLBACK}

UNITS = {
    "mtf_area": "1/mm",
    "nps_homo": "HU^2*mm^2",
    "nps_struct": "HU^2*mm^2",
    "var_struct": "HU^2",
    "ratio_lf_hf": "1",
    "nps_f_av": "1/mm",
    "nps_f_peak": "1/mm",
    "contrast_roi1": "1",
    "contrast_roi2": "1",
    "probability": "1",
    "location_mm": "mm",
    "frequency": "1/mm",
    "hu": "HU",
}
_UID_KEYS = ("study_instance_uid", "series_instance_uid", "sop_instance_uid")


# ---- the result types ----


@dataclass(frozen=True)
class SeriesInfo:
    """The measured series: acquisition parameters and DICOM references, no patient identifiers."""

    description: str | None
    slices: int
    slice_thickness_mm: float | None
    pixel_spacing_mm: float
    manufacturer_model: str | None = None
    reconstruction: str | None = None
    kvp: float | None = None
    tube_current_ma: float | None = None
    ctdi_vol_mgy: float | None = None
    dlp_mgy_cm: float | None = None
    acquisition_date: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None

    @classmethod
    def from_volume(cls, volume) -> SeriesInfo:
        return cls(
            description=volume.series_description,
            slices=volume.num_slices,
            slice_thickness_mm=volume.slice_thickness_mm,
            pixel_spacing_mm=volume.pixel_spacing_mm,
            manufacturer_model=getattr(volume, "manufacturer_model", None),
            reconstruction=getattr(volume, "reconstruction", None),
            kvp=getattr(volume, "kvp", None),
            tube_current_ma=getattr(volume, "tube_current_ma", None),
            ctdi_vol_mgy=getattr(volume, "ctdi_vol_mgy", None),
            dlp_mgy_cm=getattr(volume, "dlp_mgy_cm", None),
            acquisition_date=getattr(volume, "acquisition_date", None),
            study_instance_uid=getattr(volume, "study_instance_uid", None),
            series_instance_uid=getattr(volume, "series_instance_uid", None),
        )

    @property
    def label(self) -> str:
        """'Head 4.0 Hr40 (37 slices, 4.0 mm)' - the program's and exports' series label."""
        return f"{self.description or ''} ({self.slices} slices, {self.slice_thickness_mm} mm)".strip()


@dataclass(frozen=True)
class SeriesChoice:
    """One series of a study folder (`list_series`); `index` is what `open_study(series=...)` takes."""

    index: int
    label: str          # modality, slices, matrix, kernel, thickness, "axial"
    slices: int
    axial: bool | None


@dataclass(frozen=True)
class SliceRef:
    """Which image: the 0-based index in the series, its position and its DICOM SOP Instance UID."""

    index: int
    location_mm: float | None = None
    sop_instance_uid: str | None = None


@dataclass(frozen=True)
class Curves:
    """The curves behind one measurement (frequencies in 1/mm)."""

    mtf_frequency: tuple[float, ...]
    mtf_per_edge: tuple[tuple[float, ...], ...]     # [edge][frequency]
    nps_frequency: tuple[float, ...]
    nps_roi1: tuple[float, ...]                     # radial average, HU^2*mm^2
    nps_roi2: tuple[float, ...]
    ratio_frequency: tuple[float, ...]
    ps_mtf_ratio: tuple[float, ...]


@dataclass(frozen=True)
class Measurement:
    """One ROI pair measured on one slice. The five model inputs (`mtf_area`,
    `nps_homo`, `nps_struct`, `var_struct`, `ratio_lf_hf`) keep the model's names."""

    slice: SliceRef
    pair: str                   # "falx" or "septum"
    roi1: RoiRect
    roi2: RoiRect
    roi1_manual: bool
    roi2_manual: bool
    roi2_region: str | None     # "ventricle" / "white_matter" / "fallback"; None for a manual ROI2
    recipe: str
    mtf_area: float
    nps_homo: float             # NPS variance of ROI2 (homogeneous tissue)
    nps_struct: float           # NPS variance of ROI1 (on the structure)
    var_struct: float           # pixel variance of ROI1
    ratio_lf_hf: float
    nps_f_av: float
    nps_f_peak: float
    contrast_roi1: float
    contrast_roi2: float
    curves: Curves | None = None
    plugin_values: tuple[PluginValues, ...] = ()   # what each plug-in reported for this pair
    _quality: object = field(default=None, repr=False, compare=False)  # internal SliceQualityResult


@dataclass(frozen=True)
class Failure:
    slice: SliceRef
    pair: str
    message: str


@dataclass(frozen=True)
class FeatureFinding:
    """One model feature over the study: median, its 95 % interval, the range of rated-good training studies."""

    name: str
    label: str
    unit: str
    value: float
    low: float
    high: float
    good_low: float
    good_high: float
    contribution: float         # to the model's logit vs the training mean; < 0 = towards not acceptable
    outside: str | None         # "above" / "below" when the whole interval is outside the good range


@dataclass(frozen=True)
class NoiseScatter:
    """Spread of the ROI2 noise across slices (IQR/median) - a possible-artefact indicator."""

    value: float
    low: float
    high: float
    good_high: float
    flagged: bool               # the whole interval is above every good training study


@dataclass(frozen=True)
class SeptumCheck:
    """Septum median minus falx median of one value, against the range seen in training."""

    name: str
    difference: float
    low: float
    high: float
    consistent: bool


@dataclass(frozen=True)
class Verdict:
    """The study verdict (physical recipe; provisional model - decision support only)."""

    verdict: str                        # "acceptable" / "not_acceptable" / "uncertain" / "not_assessable"
    probability: float                  # of "acceptable"
    probability_low: float              # 95 % interval
    probability_high: float
    slices_used: int
    features: tuple[FeatureFinding, ...]
    reasons: tuple[str, ...]            # feature names behind the verdict, strongest first
    noise_scatter: NoiseScatter | None
    outlier_slices: tuple[int, ...]
    septum_checks: tuple[SeptumCheck, ...]
    slice_scores: dict[int, float]      # indicative only - the model is fitted on study medians
    model_version: str
    metal_slices: tuple[int, ...] = ()  # metal streaks in the brain (metal rule)
    model_verdict: str | None = None    # the model's own verdict when the metal rule replaced it


@dataclass(frozen=True)
class ArtefactHints:
    """Possible artefacts (descriptive; only metal streaks in the brain change the verdict)."""

    metal_slices: tuple[int, ...]
    metal_max_hu: float | None
    metal_streak_slices: tuple[int, ...]
    metal_streak_share: dict[int, float]    # metal slice -> share of nearby brain disturbed
    brainstem_band_slices: tuple[int, ...]
    brainstem_band_depth_hu: dict[int, float]


@dataclass(frozen=True)
class BasalGangliaSlice:
    """Border contrast of the basal ganglia against the internal capsule (provisional)."""

    border_hu: float
    differences_hu: dict[str, float]    # nucleus -> HU difference
    plausible: bool


@dataclass(frozen=True)
class SegmentationCheck:
    """Study-level check of the anatomy segmentation (label fragmentation, provisional)."""

    fragmentation: float
    slices: int
    worst_slices: tuple[int, ...]
    lobe_islands: int
    unreliable: bool


@dataclass(frozen=True)
class _Raw:
    """The internal objects behind a result - only for the program's own exports."""

    verdict: object = None
    artefacts: object = None
    basal_ganglia: object = None
    segmentation_check: object = None


@dataclass(frozen=True)
class StudyResult:
    """Everything one QA run found for one series of a study."""

    label: str
    series: SeriesInfo
    settings: Settings
    rois_reviewed: bool                 # the caller says a person checked the ROIs
    rois_edited: bool                   # some ROI was moved, drawn or deleted
    measurements: tuple[Measurement, ...]
    failed: tuple[Failure, ...] = ()
    skipped: dict[str, tuple[int, ...]] = field(default_factory=dict)  # pair -> slices with too little brain
    segmented_slices: int = 0
    verdict: Verdict | None = None      # physical recipe; None for the legacy recipe or too few slices
    artefacts: ArtefactHints | None = None
    basal_ganglia: dict[int, BasalGangliaSlice] = field(default_factory=dict)
    segmentation_check: SegmentationCheck | None = None
    plugins: dict[str, str] = field(default_factory=dict)        # the active plug-ins: name -> version
    plugin_values: tuple[PluginValues, ...] = ()                 # each plug-in's values for the whole series
    api_version: str = API_VERSION
    schema_version: str = SCHEMA_VERSION
    _raw: _Raw = field(default_factory=_Raw, repr=False, compare=False)

    def measurements_for(self, pair: str) -> dict[int, Measurement]:
        """{slice index: measurement} of one pair ("falx" or "septum")."""
        return {m.slice.index: m for m in self.measurements if m.pair == pair}

    # ---- conversion ----

    def to_dict(self, include_curves: bool = False, include_dicom_uids: bool = False) -> dict:
        """Plain, JSON-ready values. `include_curves` keeps each measurement's
        curves (if measured with them); `include_dicom_uids` keeps the study,
        series and image UIDs - left out by default, as they link to the patient in the PACS."""
        data = _to_plain(self)
        data["units"] = dict(UNITS)
        if not include_curves:
            for m in data["measurements"]:
                m["curves"] = None
        if not include_dicom_uids:
            _drop_uids(data)
        return data

    def to_json(self, include_curves: bool = False, include_dicom_uids: bool = False, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(include_curves, include_dicom_uids), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> StudyResult:
        """A result back from `to_dict()` (same schema major version). It has
        no internal objects, so `to_volume_qa` is not available on it."""
        major = str(data.get("schema_version", "")).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"Result schema {data.get('schema_version')!r} - this API reads {SCHEMA_VERSION}")
        return _from_plain(cls, {k: v for k, v in data.items() if k != "units"})

    @classmethod
    def from_json(cls, text: str) -> StudyResult:
        return cls.from_dict(json.loads(text))

    def to_volume_qa(self, group: str = "") -> VolumeQa:
        """The internal container the program's Excel/CSV exports read (`integration.results_export`)."""
        if self.measurements and self.measurements[0]._quality is None:
            raise ValueError("A result read back with from_dict has no internal measurements to export")
        raw = self._raw
        qa = VolumeQa(
            group=group,
            name=self.label,
            series_label=self.series.label,
            slice_count=self.series.slices,
            segmented_slices=self.segmented_slices,
            pixel_spacing_mm=self.series.pixel_spacing_mm,
            acquisition_date=self.series.acquisition_date,
            head_ct_verdict=raw.verdict,
            artefact_flags=raw.artefacts,
            basal_ganglia=raw.basal_ganglia,
            segmentation_plausibility=raw.segmentation_check,
        )
        domain_regions = {v: k for k, v in _REGION_IDS.items()}
        # In `structured_roi.STRUCTURES` order (septum first), as the exports have always listed them.
        for pair in sorted(PAIRS, key=lambda p: STRUCTURES.index(PAIR_STRUCTURES[p])):
            structure = PAIR_STRUCTURES[pair]
            sq = StructureQa(structure, skipped=list(self.skipped.get(pair, ())))
            for m in self.measurements:
                if m.pair == pair:
                    region = domain_regions.get(m.roi2_region, "manual")
                    sq.rois[m.slice.index] = (_candidate(m.roi1, structure), _candidate(m.roi2, region))
                    sq.results[m.slice.index] = m._quality
            sq.errors = {f.slice.index: f.message for f in self.failed if f.pair == pair}
            qa.structures[structure] = sq
        from bictool.integration.results_export import plugin_rows

        qa.plugin_rows = plugin_rows(
            {(m.slice.index, m.pair): m.plugin_values for m in self.measurements if m.plugin_values},
            self.plugin_values, {"falx": "Falx", "septum": "Septum"},
        )
        return qa


def _candidate(rect: RoiRect, structure: str) -> RoiCandidate:
    return RoiCandidate(rect.y0, rect.x0, rect.width, structure)


# ---- building results from the internal objects ----


def _floats(values) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(values, dtype=np.float64).ravel())


def region_id(domain_region: str | None) -> str | None:
    return None if domain_region is None else _REGION_IDS.get(domain_region, domain_region)


def make_curves(quality) -> Curves:
    mtf = np.asarray(quality.mtf, dtype=np.float64)
    per_edge = mtf.T if mtf.ndim == 2 else mtf[np.newaxis, :]
    return Curves(
        mtf_frequency=_floats(quality.mtf_xaxis),
        mtf_per_edge=tuple(_floats(edge) for edge in per_edge),
        nps_frequency=_floats(quality.nps_roi2.f_avg),
        nps_roi1=_floats(quality.nps_roi1.nps_measured_radagv),
        nps_roi2=_floats(quality.nps_roi2.nps_measured_radagv),
        ratio_frequency=_floats(quality.ratio.f_int),
        ps_mtf_ratio=_floats(quality.ratio.ratio_nps_mtf),
    )


def make_measurement(
    slice_ref, pair, roi1, roi2, roi1_manual, roi2_manual, region, quality, curves: bool, plugin_values=()
) -> Measurement:
    model_inputs = dict(zip(verdicts.CORE_FEATURES, (float(v) for v in quality.feature_vector)))
    return Measurement(
        slice=slice_ref, pair=pair, roi1=roi1, roi2=roi2, roi1_manual=roi1_manual, roi2_manual=roi2_manual,
        roi2_region=None if roi2_manual else region_id(region),
        recipe=quality.recipe,
        nps_f_av=float(quality.nps_f_av), nps_f_peak=float(quality.nps_f_peak),
        contrast_roi1=float(quality.contrast_roi1), contrast_roi2=float(quality.contrast_roi2),
        curves=make_curves(quality) if curves else None,
        plugin_values=tuple(plugin_values),
        _quality=quality,
        **model_inputs,
    )


def make_verdict(v) -> Verdict | None:
    if v is None:
        return None

    def finding(f) -> FeatureFinding:
        return FeatureFinding(
            name=f.name, label=verdicts.FEATURE_LABELS.get(f.name, f.name), unit=UNITS.get(f.name, ""),
            value=f.value, low=f.low, high=f.high, good_low=f.good_low, good_high=f.good_high,
            contribution=f.contribution, outside=f.outside,
        )

    a = v.artefact
    return Verdict(
        verdict=_VERDICT_IDS[v.verdict],
        probability=v.probability, probability_low=v.probability_low, probability_high=v.probability_high,
        slices_used=v.n_slices,
        features=tuple(finding(f) for f in v.features),
        reasons=tuple(f.name for f in v.reasons),
        noise_scatter=None if a is None else NoiseScatter(a.value, a.low, a.high, a.good_high, a.flagged),
        outlier_slices=tuple(int(i) for i in v.outlier_slices),
        septum_checks=tuple(SeptumCheck(c.name, c.difference, c.low, c.high, c.consistent) for c in v.plausibility),
        slice_scores={int(i): float(p) for i, p in zip(v.slice_ids, v.slice_probabilities)},
        model_version=v.model_version,
        metal_slices=tuple(int(i) for i in v.metal_slices),
        model_verdict=_VERDICT_IDS.get(v.model_verdict) if v.model_verdict else None,
    )


def make_artefacts(flags) -> ArtefactHints | None:
    if flags is None:
        return None
    return ArtefactHints(
        metal_slices=tuple(int(i) for i in flags.metal_slices),
        metal_max_hu=flags.metal_max_hu,
        metal_streak_slices=tuple(int(i) for i in flags.metal_streak_slices),
        metal_streak_share={int(i): float(s) for i, s in flags.metal_streak_shares.items()},
        brainstem_band_slices=tuple(int(i) for i in flags.band_slices),
        brainstem_band_depth_hu={int(i): float(d) for i, d in flags.band_depths.items()},
    )


def make_basal_ganglia(by_slice) -> dict[int, BasalGangliaSlice]:
    return {
        int(i): BasalGangliaSlice(float(s.border), {k: float(d) for k, d in s.differences.items()}, bool(s.plausible))
        for i, s in (by_slice or {}).items()
    }


def make_segmentation_check(check) -> SegmentationCheck | None:
    if check is None:
        return None
    return SegmentationCheck(
        float(check.fragmentation), int(check.slices), tuple(int(i) for i in check.worst_slices),
        int(check.lobe_islands), bool(check.unreliable),
    )


# ---- plain values <-> the result types ----


def _settings_from_plain(data: dict) -> Settings:
    method = data["tissue_method"]
    params = data.get("tissue_parameters") or {}
    slot, kind = {"threshold": ("threshold", BrainTissueSettings), "fcm": ("fcm", FcmSettings)}.get(
        method, ("adaptive", AdaptiveSettings)
    )
    tissue = TissueSettings(method=method, **({slot: kind(**params)} if params else {}))
    return Settings(recipe=data["recipe"], tissue=tissue)


def _to_plain(value):
    if isinstance(value, Settings):
        return value.to_dict()
    if isinstance(value, RoiRect):
        return [int(value.x0), int(value.y0), int(value.x1), int(value.y1)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in dataclasses.fields(value) if not f.name.startswith("_")}
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _drop_uids(data) -> None:
    if isinstance(data, dict):
        for key in _UID_KEYS:
            data.pop(key, None)
        for value in data.values():
            _drop_uids(value)
    elif isinstance(data, list):
        for value in data:
            _drop_uids(value)


def _from_plain(kind, value):
    """`value` (from `_to_plain`) as an instance of the annotated type `kind`."""
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        options = [a for a in args if a is not type(None)]
        return _from_plain(options[0], value) if len(options) == 1 else value
    if value is None:
        return float("nan") if kind is float else None
    if kind is Settings:
        return _settings_from_plain(value)
    if kind is RoiRect:
        return RoiRect(*value)
    if dataclasses.is_dataclass(kind):
        hints = typing.get_type_hints(kind)
        return kind(**{
            f.name: _from_plain(hints[f.name], value[f.name])
            for f in dataclasses.fields(kind) if not f.name.startswith("_") and f.name in value
        })
    if origin is tuple:
        return tuple(_from_plain(args[0], v) for v in value)
    if origin is dict:
        key_type, item = args
        return {(int(k) if key_type is int else k): _from_plain(item, v) for k, v in value.items()}
    if kind is float:
        return float(value)
    return value
