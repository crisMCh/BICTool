"""The pipeline's steps - the only place that strings the domain functions
together. The GUI's ViewModels (incrementally, in background jobs), the
public API (`bictool.api.Study`) and the batch tool all call these, so a
study gives the same numbers whichever way it is run.

Internal: not part of the public API - the steps take internal objects and
change with the program. Every step is stateless: per slice (segment, WM/GM,
ROIs, measurement) or per study (tissue references, hints, verdict). Masks
are in DICOM pixel space. No Qt, no threads; only `list_series`,
`open_series`, `load_anatomy_model` and `prepare_segmentation` read files.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from bictool.application.roi_store import PAIR_STRUCTURES, RoiRect, candidate_to_rect, crop_roi
from bictool.application import plugin_manager
from bictool.application.plugin_types import (
    CSF,
    GRAY_MATTER,
    WHITE_MATTER,
    PairContext,
    PluginHints,
    PluginValues,
    SliceContext,
    StudyContext,
    TissueContext,
)
from bictool.application.qa_settings import (
    METHOD_ADAPTIVE,
    METHOD_ANCHORED,
    METHOD_FCM,
    METHOD_LABELS,
    TissueSettings,
    is_plugin_method,
)
from bictool.domain.artefacts.artefact_flags import ArtefactFlags, evaluate_artefact_flags
from bictool.domain.metrics.basal_ganglia import BasalGangliaSlice, evaluate_basal_ganglia
from bictool.domain.metrics.mtf_seed import mtf_seed_candidates
from bictool.domain.results.head_ct_verdict import (
    CORE_FEATURES,
    HeadCtModel,
    HeadCtVerdict,
    apply_metal_rule,
    evaluate_study_results,
)
from bictool.domain.results.study_quality import SliceQualityResult, evaluate_slice_quality
from bictool.domain.roi_selection.homogeneous_roi import find_roi2_candidate
from bictool.domain.roi_selection.slice_eligibility import eligible_slices
from bictool.domain.roi_selection.structured_roi import (
    STRUCTURES,
    RoiCandidate,
    estimate_midline_direction,
    find_roi1_candidate,
)
from bictool.domain.segmentation.brain_tissue import BrainTissueResult, classify_brain_tissue, extract_brain_mask
from bictool.domain.segmentation.brain_tissue_adaptive import classify_brain_tissue_adaptive
from bictool.domain.segmentation.brain_tissue_anchored import (
    AnchoredModel,
    classify_brain_tissue_anchored,
    fit_anchored_model,
)
from bictool.domain.segmentation.brain_tissue_fcm import classify_brain_tissue_fcm
from bictool.domain.segmentation.segmentation_brain_mask import brain_mask_from_segmentation
from bictool.domain.segmentation.segmentation_plausibility import SegmentationPlausibility, assess_plausibility
from bictool.domain.segmentation.tissue_references import TissueReferences, estimate_tissue_references
from bictool.integration import model_files
from bictool.integration.dicom_file_loader import DicomVolume, SeriesCandidate, discover, load_series

Masks = dict[str, np.ndarray]
# (slice_hu, brain_mask, slice_index=None) -> BrainTissueResult (+ "Brain Mask" in `.masks`)
TissueClassifier = Callable[..., object]


# ---- series ----


def list_series(path: str | Path) -> list[SeriesCandidate]:
    """Every series in a study folder (or the one file), best first: CT, axial,
    closest to 4 mm - the first is the program's default choice."""
    return discover(path)


def open_series(path: str | Path, series: int = 0) -> DicomVolume:
    """Series number `series` of `list_series(path)` (default: the program's choice)."""
    candidates = list_series(path)
    if not 0 <= series < len(candidates):
        raise IndexError(f"Series {series} does not exist - the study has {len(candidates)} series")
    return load_series(candidates[series])


# ---- anatomy segmentation (CNN) ----


def load_anatomy_model(weights_path: str | Path | None = None):
    """The head-CT U-Net with its weights loaded - from `weights_path`, else the
    configured or bundled ones (`infrastructure.config`). Imports TensorFlow.
    Raises FileNotFoundError when no weights are configured or the file is missing."""
    from bictool.domain.segmentation import head_ct_unet
    from bictool.infrastructure.config import load_config

    path = weights_path or load_config().get("segmentation", {}).get("anatomy_weights_path")
    if not path:
        raise FileNotFoundError(
            "No anatomy segmentation weights configured (BICTOOL_HEADCT_WEIGHTS or "
            "segmentation.anatomy_weights_path in ~/.bictool/config.json)"
        )
    model = head_ct_unet.build_model()
    model_files.load_anatomy_weights(model, path)
    return model


def prepare_segmentation(volume: DicomVolume):
    """The series as the CNN needs it (DICOM -> NIfTI, once per series; seconds)."""
    from bictool.integration.dicom_to_nifti import build_dicom_nifti_volume

    return build_dicom_nifti_volume(volume.slice_paths, volume.get_slice_hu)


def segment_slice(model, nifti_volume, index: int):
    """Slice `index`'s `SegmentationResult` in DICOM pixel space, or None if the
    slice can't be segmented (e.g. a scout sharing the series' geometry)."""
    from bictool.domain.segmentation.head_ct_unet import segment_slice as segment_nifti_slice

    if not nifti_volume.is_segmentable(index):
        return None
    return nifti_volume.remap_result_to_dicom_space(segment_nifti_slice(model, nifti_volume.get_nifti_slice_hu(index)))


def segmentation_check(masks_by_slice: Mapping[int, Masks], pixel_spacing_mm: float) -> SegmentationPlausibility:
    """The study-level segmentation check (label fragmentation, provisional)."""
    return assess_plausibility(dict(masks_by_slice), pixel_spacing_mm)


# ---- WM/GM (decides where the falx pair's ROI2 goes) ----


def brain_masks(masks_by_slice: Mapping[int, Masks], num_slices: int) -> dict[int, np.ndarray]:
    """Per segmented slice: the brain (merged CNN labels, largest 3D component)."""
    return brain_mask_from_segmentation(dict(masks_by_slice), num_slices)


def uses_references(tissue: TissueSettings) -> bool:
    """Does the method need the study's tissue references (from the basal ganglia)?"""
    return tissue.method == METHOD_ANCHORED or (tissue.method == METHOD_ADAPTIVE and tissue.adaptive.use_study_range)


def tissue_references(masks_by_slice: Mapping[int, Masks], get_slice_hu) -> TissueReferences | None:
    """Gray/white/CSF reference HU of the study (None until the basal ganglia and ventricles are segmented)."""
    return estimate_tissue_references(dict(masks_by_slice), get_slice_hu)


def fit_anchored(
    masks_by_slice: Mapping[int, Masks], volume, references: TissueReferences | None
) -> AnchoredModel | None:
    """The Anchored prototype's per-series model (None without references)."""
    if references is None or not masks_by_slice:
        return None
    masks = dict(masks_by_slice)
    brains = brain_masks(masks, volume.num_slices)
    return fit_anchored_model(masks, volume.get_slice_hu, brains, volume.pixel_spacing_mm, references)


def make_tissue_classifier(
    tissue: TissueSettings,
    pixel_spacing_mm: float | None,
    references: TissueReferences | None = None,
    anchored_model: AnchoredModel | None = None,
    masks_by_slice: Mapping[int, Masks] | None = None,
) -> TissueClassifier:
    """`(slice_hu, brain_mask, slice_index=None) -> BrainTissueResult` for the
    method and settings, snapshotted - safe to run on another thread. The
    prototype needs the slice index and its model; without them it falls back to Adaptive.
    A plug-in's method comes from the plug-ins in use (`plugin_manager.current()`) and
    sees the CNN masks of `masks_by_slice` (`TissueContext.labels`)."""
    if is_plugin_method(tissue.method):
        return _plugin_tissue_classifier(tissue, pixel_spacing_mm, masks_by_slice)
    method, threshold, fcm, adaptive = tissue.method, tissue.threshold, tissue.fcm, tissue.adaptive
    spacing = pixel_spacing_mm if adaptive.smoothing_mm > 0 else None
    sigma_px = adaptive.smoothing_mm / spacing if spacing else 0.0
    study_range = references.plausible_range() if references is not None and adaptive.use_study_range else None

    def classify(slice_hu: np.ndarray, brain_mask: np.ndarray, slice_index: int | None = None):
        if method == METHOD_ANCHORED and anchored_model is not None and slice_index is not None:
            result = classify_brain_tissue_anchored(slice_hu, brain_mask, anchored_model, slice_index)
        elif method == METHOD_FCM:
            result = classify_brain_tissue_fcm(
                slice_hu, brain_mask, fuzziness=fcm.fuzziness, max_iter=fcm.max_iter,
                spatial_weight=fcm.spatial_weight, min_component_size=fcm.min_component_size,
            )
        elif method in (METHOD_ADAPTIVE, METHOD_ANCHORED):
            result = classify_brain_tissue_adaptive(
                slice_hu, brain_mask, block_size=adaptive.block_size,
                plausible_range=study_range or (adaptive.plausible_low, adaptive.plausible_high),
                min_component_size=adaptive.min_component_size,
                smoothing_sigma_px=sigma_px,
            )
        else:
            result = classify_brain_tissue(
                slice_hu, brain_mask,
                csf_range=(threshold.csf_low, threshold.csf_high),
                white_matter_range=(threshold.white_matter_low, threshold.white_matter_high),
                gray_matter_range=(threshold.gray_matter_low, threshold.gray_matter_high),
                smoothing_size=threshold.smoothing_size,
                min_component_size=threshold.min_component_size,
            )
        # Not a tissue class, but the same `masks` dict carries it for the
        # "Brain Mask" overlay.
        result.masks["Brain Mask"] = brain_mask
        return result

    return classify


def _plugin_tissue_classifier(tissue: TissueSettings, pixel_spacing_mm: float | None, masks_by_slice=None):
    method = plugin_manager.current().tissue_method(tissue.method)
    if method is None:
        raise ValueError(f"WM/GM method {tissue.method!r} is not available (plug-in missing or switched off)")
    settings = tissue.plugin if tissue.plugin is not None else method.settings
    if isinstance(settings, Mapping) and dataclasses.is_dataclass(method.settings):
        # A dict (e.g. read back from a JSON result) = fields changed from the plug-in's defaults.
        try:
            settings = dataclasses.replace(method.settings, **settings)
        except TypeError as exc:
            raise ValueError(f"WM/GM method {tissue.method!r}: settings {dict(settings)} do not fit - {exc}") from exc

    def classify(slice_hu: np.ndarray, brain_mask: np.ndarray, slice_index: int | None = None):
        labels = None
        if masks_by_slice is not None and slice_index in masks_by_slice:
            labels = read_only_masks(masks_by_slice[slice_index])
        context = TissueContext(slice_index, pixel_spacing_mm, labels)
        try:
            masks = method.classify(read_only(slice_hu), read_only(brain_mask), settings, context)
        except Exception as exc:
            raise ValueError(f"WM/GM method {tissue.method!r} failed: {type(exc).__name__}: {exc}") from exc
        empty = np.zeros(slice_hu.shape, dtype=bool)
        result = {"White Matter": empty, "Gray Matter": empty, "CSF": empty}  # what a method left out is empty
        for name, mask in dict(masks or {}).items():
            mask = np.asarray(mask)
            if mask.shape != slice_hu.shape:
                raise ValueError(f"WM/GM method {tissue.method!r} returned a {mask.shape} mask for a {slice_hu.shape} slice")
            result[name] = mask.astype(bool) & brain_mask
        result["Brain Mask"] = brain_mask
        return BrainTissueResult(masks=result)

    return classify


def tissue_method_label(method: str) -> str:
    """The WM/GM method's name as the program shows it ("Adaptive Thresholding", "Otsu (plug-in)")."""
    if is_plugin_method(method):
        found = plugin_manager.current().tissue_method(method)
        return f"{found.label} (plug-in)" if found is not None else method
    return METHOD_LABELS.get(method, method)


def study_tissue_classifier(tissue: TissueSettings, masks_by_slice: Mapping[int, Masks], volume) -> TissueClassifier:
    """`make_tissue_classifier` with the study's references and prototype model computed from the whole segmentation."""
    if is_plugin_method(tissue.method):
        return _plugin_tissue_classifier(tissue, volume.pixel_spacing_mm, masks_by_slice)
    references = tissue_references(masks_by_slice, volume.get_slice_hu) if uses_references(tissue) else None
    model = fit_anchored(masks_by_slice, volume, references) if tissue.method == METHOD_ANCHORED else None
    return make_tissue_classifier(tissue, volume.pixel_spacing_mm, references, model)


# ---- automatic ROIs ----


def eligible_for_auto_rois(masks_by_slice: Mapping[int, Masks]) -> set[int]:
    """The segmented slices with enough brain for automatic ROIs (`slice_eligibility`)."""
    return eligible_slices(masks_by_slice)


def midline_direction(masks_by_slice: Mapping[int, Masks]) -> np.ndarray:
    """The volume-wide midline direction the falx/septum ROI1s are aligned with."""
    return estimate_midline_direction(dict(masks_by_slice))


def auto_roi1(
    masks: Masks, pair: str, pixel_spacing_mm: float, midline: np.ndarray | None, image: np.ndarray | None
) -> RoiRect | None:
    """The pair's ROI1 on its structure on one slice, or None if there is none."""
    candidate = find_roi1_candidate(
        masks, PAIR_STRUCTURES[pair], pixel_spacing_mm, midline_direction=midline, image=image
    )
    return None if candidate is None else candidate_to_rect(candidate)


def auto_roi2(
    roi1: RoiRect, pair: str, masks: Masks, pixel_spacing_mm: float, tissue_masks: Masks | None
) -> tuple[RoiRect, str] | None:
    """(ROI2, region) next to `roi1`; region is `homogeneous_roi.VENTRICLE`,
    `WHITE_MATTER` or `FALLBACK`. None while the falx pair still needs the
    slice's WM/GM masks (`tissue_masks` None)."""
    candidate = find_roi2_candidate(
        RoiCandidate(roi1.y0, roi1.x0, roi1.width, PAIR_STRUCTURES[pair]), masks, pixel_spacing_mm, tissue_masks
    )
    return None if candidate is None else (candidate_to_rect(candidate), candidate.structure)


# ---- measurement ----


def structure_mask(masks: Masks | None, rect: RoiRect) -> np.ndarray | None:
    """The ROI1 structures (septum + dural folds) cropped to `rect` - places the
    MTF seed and checks that the measured edge lies on the structure. None
    without a segmentation of the slice."""
    if masks is None:
        return None
    found = [masks[name] for name in STRUCTURES if name in masks]
    return crop_roi(np.logical_or.reduce(found), rect) if found else None


def measure_pair(
    slice_hu: np.ndarray,
    masks: Masks | None,
    roi1: RoiRect,
    roi2: RoiRect,
    pixel_spacing_mm: float,
    recipe: str,
    *,
    keep_steps: bool = False,
    svm_model=None,
) -> SliceQualityResult:
    """MTF, NPS, contrast and PS/MTF of one ROI pair on one slice. `masks` = the
    slice's segmentation (None for a slice without one): the MTF contour starts
    on the structure's centerline, with fallbacks if a seed finds no edges.
    Raises if the measurement fails."""
    img_roi1, img_roi2 = crop_roi(slice_hu, roi1), crop_roi(slice_hu, roi2)
    structure = structure_mask(masks, roi1)
    return evaluate_slice_quality(
        img_roi1, img_roi2, pixel_spacing_mm,
        svm_model=svm_model if svm_model is not None else model_files.default_chest_svm_model(),
        mtf_seeds=mtf_seed_candidates(img_roi1, structure),
        mtf_structure=structure,
        recipe=recipe,
        keep_steps=keep_steps,
    )


# ---- study level ----


@dataclass(frozen=True)
class ImageHints:
    """Possible artefacts and the basal-ganglia visibility of a series (descriptive, provisional)."""

    artefacts: ArtefactFlags
    basal_ganglia: dict[int, BasalGangliaSlice]


def image_hints(masks_by_slice: Mapping[int, Masks], volume) -> ImageHints:
    masks = dict(masks_by_slice)
    return ImageHints(
        artefacts=evaluate_artefact_flags(masks, volume.get_slice_hu, volume.num_slices, volume.pixel_spacing_mm),
        basal_ganglia=evaluate_basal_ganglia(masks, volume.get_slice_hu),
    )


def study_verdict(
    model: HeadCtModel | None,
    falx_results: Mapping[int, SliceQualityResult],
    septum_results: Mapping[int, SliceQualityResult] | None,
    artefacts: ArtefactFlags | None,
) -> HeadCtVerdict | None:
    """The head-CT verdict (physical recipe) from the falx slices, with the
    metal rule applied. None without a model or with too few slices."""
    return apply_metal_rule(
        evaluate_study_results(model, falx_results, septum_results),
        artefacts.metal_streak_slices if artefacts is not None else (),
    )


# ---- plug-ins (descriptive values only - never part of the verdict) ----


def read_only(array: np.ndarray) -> np.ndarray:
    """A view a plug-in can read but not change (it may be BICTool's cached slice)."""
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


def read_only_masks(masks: Masks | None):
    return None if masks is None else MappingProxyType({name: read_only(mask) for name, mask in masks.items()})


def bictool_values(quality: SliceQualityResult) -> dict[str, float]:
    """BICTool's own values of one measurement, by their `api.Measurement` names."""
    values = dict(zip(CORE_FEATURES, (float(v) for v in quality.feature_vector)))
    values.update(
        nps_f_av=float(quality.nps_f_av), nps_f_peak=float(quality.nps_f_peak),
        contrast_roi1=float(quality.contrast_roi1), contrast_roi2=float(quality.contrast_roi2),
    )
    return values


def pair_context(
    slice_index: int, pair: str, slice_hu: np.ndarray, masks: Masks | None, roi1: RoiRect, roi2: RoiRect,
    pixel_spacing_mm: float, quality: SliceQualityResult,
) -> PairContext:
    """What a plug-in sees of one measured ROI pair."""
    return PairContext(
        slice_index, pair, read_only(slice_hu), pixel_spacing_mm, roi1, roi2, read_only_masks(masks),
        MappingProxyType(bictool_values(quality)),
    )


def plugin_pair_metrics(context: PairContext) -> tuple[PluginValues, ...]:
    """Every active plug-in's values for one measured ROI pair (() without plug-ins)."""
    manager = plugin_manager.current()
    return manager.pair_metrics(context) if manager.has_pair_metrics else ()


class PluginInputs:
    """What the slice and study contexts compute on first use and share: BICTool's
    brain mask of a slice (from the segmentation, else the HU skull-strip - as the
    program's Brain Mask overlay) and its WM/GM/CSF masks from the chosen method
    (with the study's references - as `study_tissue_classifier`). Nothing is
    computed unless a plug-in asks."""

    def __init__(self, volume, masks_by_slice: Mapping[int, Masks], tissue: TissueSettings) -> None:
        self._volume = volume
        self._masks = dict(masks_by_slice)
        self.tissue_settings = tissue
        self._lock = threading.RLock()  # a plug-in may call from its own threads
        self._brains: dict[int, np.ndarray] | None = None
        self._brain: dict[int, np.ndarray] = {}
        self._classify: TissueClassifier | None = None
        self._tissue: dict[int, Mapping[str, np.ndarray]] = {}
        self.tissue_used = False  # a plug-in asked for WM/GM masks (the method is then recorded)

    def labels(self, index: int):
        return read_only_masks(self._masks.get(index))

    def brain_mask(self, index: int) -> np.ndarray:
        with self._lock:
            if index not in self._brain:
                if self._brains is None:
                    self._brains = brain_masks(self._masks, self._volume.num_slices) if self._masks else {}
                mask = self._brains.get(index)
                if mask is None:
                    threshold = self.tissue_settings.threshold
                    image = self._volume.get_slice_hu(index)
                    mask = extract_brain_mask(image, threshold.air_hu_cutoff, threshold.bone_hu_cutoff)
                self._brain[index] = read_only(mask)
            return self._brain[index]

    def tissue(self, index: int) -> Mapping[str, np.ndarray]:
        with self._lock:
            if index not in self._tissue:
                if self._classify is None:
                    self._classify = study_tissue_classifier(self.tissue_settings, self._masks, self._volume)
                result = self._classify(self._volume.get_slice_hu(index), np.asarray(self.brain_mask(index)), index)
                self._tissue[index] = read_only_masks(
                    {name: result.masks[name] for name in (WHITE_MATTER, GRAY_MATTER, CSF) if name in result.masks}
                )
                self.tissue_used = True
            return self._tissue[index]


def slice_context(volume, index: int, pairs: tuple[PairContext, ...], inputs: PluginInputs) -> SliceContext:
    """What a plug-in sees of one measured slice."""
    return SliceContext(
        slice_index=index,
        image=read_only(volume.get_slice_hu(index)),
        pixel_spacing_mm=volume.pixel_spacing_mm,
        labels=inputs.labels(index),
        pairs=tuple(pairs),
        brain_mask=lambda: inputs.brain_mask(index),
        tissue=lambda: inputs.tissue(index),
        tissue_method=inputs.tissue_settings.method,
    )


def study_context(
    volume, masks_by_slice: Mapping[int, Masks], pairs: tuple[PairContext, ...], inputs: PluginInputs | None = None
) -> StudyContext:
    """What a plug-in sees of the whole series."""
    inputs = inputs or PluginInputs(volume, masks_by_slice, TissueSettings())
    return StudyContext(
        num_slices=volume.num_slices,
        pixel_spacing_mm=volume.pixel_spacing_mm,
        slice_thickness_mm=getattr(volume, "slice_thickness_mm", None),
        segmented_slices=tuple(sorted(masks_by_slice)),
        pairs=tuple(pairs),
        image=lambda index: read_only(volume.get_slice_hu(index)),
        labels=inputs.labels,
        brain_mask=inputs.brain_mask,
        tissue=inputs.tissue,
        tissue_method=inputs.tissue_settings.method,
        measured_slices=tuple(sorted({pair.slice_index for pair in pairs})),
    )


def plugin_study_metrics(context: StudyContext) -> tuple[PluginValues, ...]:
    """Every active plug-in's values for the whole series (() without plug-ins)."""
    manager = plugin_manager.current()
    return manager.study_metrics(context) if manager.has_study_metrics else ()


@dataclass(frozen=True)
class PluginStudyOutputs:
    """What the plug-ins reported after the ROI pairs were measured."""

    slice_values: dict[int, tuple[PluginValues, ...]] = field(default_factory=dict)  # slice -> each plug-in's values
    study_values: tuple[PluginValues, ...] = ()
    hints: tuple[PluginHints, ...] = ()
    tissue_method: str | None = None  # the WM/GM method whose masks plug-ins used (None: none asked)


def plugin_study_phase(
    volume, masks_by_slice: Mapping[int, Masks], pairs: tuple[PairContext, ...], tissue: TissueSettings
) -> PluginStudyOutputs:
    """The plug-ins' slice values (each slice with a measured pair, after its pairs),
    series values and artefact hints - in that order. Empty without such plug-ins
    (nothing is computed then). `pairs` = the contexts of the measured pairs."""
    manager = plugin_manager.current()
    if not (manager.has_slice_metrics or manager.has_study_metrics or manager.has_artefact_hints):
        return PluginStudyOutputs()
    inputs = PluginInputs(volume, masks_by_slice, tissue)
    slice_values: dict[int, tuple[PluginValues, ...]] = {}
    if manager.has_slice_metrics:
        by_slice: dict[int, list[PairContext]] = {}
        for pair in pairs:
            by_slice.setdefault(pair.slice_index, []).append(pair)
        for index in sorted(by_slice):
            values = manager.slice_metrics(slice_context(volume, index, tuple(by_slice[index]), inputs))
            if values:
                slice_values[index] = values
    context = study_context(volume, masks_by_slice, pairs, inputs)
    study_values = manager.study_metrics(context) if manager.has_study_metrics else ()
    hints = manager.artefact_hints(context) if manager.has_artefact_hints else ()
    return PluginStudyOutputs(slice_values, study_values, hints, tissue.method if inputs.tissue_used else None)
