"""BrainTissueViewModel: WM/GM/CSF separation of the shown series.

Owns the chosen method (Adaptive Thresholding by default, Fuzzy C-Means,
HU Thresholding, and the "Anchored (prototype)"), each method's settings, the
study's tissue references (from the basal ganglia - used by Adaptive's study
range and by the prototype), and the per-slice results. The
brain mask a method works inside comes from the anatomy segmentation when
the slice has one (more robust on irregular anatomy), else from the
HU-threshold skull-strip. Results are computed on demand (milliseconds per
slice) and cached; any method/settings change drops them and emits
`settings_changed` (the view redraws overlays and re-places white-matter
ROI2s). QtCore only - no widgets.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from bictool.domain.segmentation.brain_tissue import classify_brain_tissue, extract_brain_mask
from bictool.domain.segmentation.brain_tissue_adaptive import classify_brain_tissue_adaptive
from bictool.domain.segmentation.brain_tissue_anchored import (
    AnchoredModel,
    classify_brain_tissue_anchored,
    fit_anchored_model,
)
from bictool.domain.segmentation.brain_tissue_fcm import classify_brain_tissue_fcm
from bictool.domain.segmentation.segmentation_brain_mask import brain_mask_from_segmentation
from bictool.domain.segmentation.tissue_references import (
    SliceSamples,
    TissueReferences,
    combine,
    slice_samples,
)

METHOD_FCM = "Fuzzy C-Means"
METHOD_THRESHOLD = "HU Thresholding"
METHOD_ADAPTIVE = "Adaptive Thresholding"
METHOD_ANCHORED = "Anchored (prototype)"
METHODS = [METHOD_FCM, METHOD_THRESHOLD, METHOD_ADAPTIVE, METHOD_ANCHORED]
DEFAULT_METHOD = METHOD_ADAPTIVE  # no method wins on every scan; this one works on most


class BrainTissueViewModel(QObject):
    settings_changed = Signal()  # method or settings changed - every result was dropped

    def __init__(
        self,
        volume: Callable[[], object],
        segmentation_cache: Callable[[], dict],
        threshold_settings,
        fcm_settings,
        adaptive_settings,
        parent: QObject | None = None,
    ) -> None:
        """`volume()` = the shown DICOM volume; `segmentation_cache()` = the
        anatomy results by slice; the three settings objects are the methods'
        defaults (the settings dialogs' dataclasses)."""
        super().__init__(parent)
        self._volume = volume
        self._segmentation_cache = segmentation_cache
        self.method = DEFAULT_METHOD
        self.default_settings = {
            METHOD_THRESHOLD: threshold_settings, METHOD_FCM: fcm_settings, METHOD_ADAPTIVE: adaptive_settings,
        }
        self.settings = dict(self.default_settings)
        self.cache: dict[int, object] = {}  # slice index -> BrainTissueResult (+ "Brain Mask")
        self._samples: dict[int, SliceSamples] = {}  # per segmented slice, for the tissue references
        self._references_used: TissueReferences | None = None  # what the cached results were computed with
        self._anchored: tuple[frozenset, AnchoredModel | None] | None = None  # (segmented slices, model)

    # ---- commands ----

    def set_method(self, method: str) -> None:
        self.method = method
        self._changed()

    def set_settings(self, method: str, settings) -> None:
        self.settings[method] = settings
        self._changed()

    def invalidate(self, index: int) -> None:
        """Slice `index` got (new) anatomy segmentation - recompute it on demand."""
        self.cache.pop(index, None)
        self._samples.pop(index, None)

    def reset(self) -> None:
        """A new/closed series. The method and settings are app preferences and stay."""
        self.cache = {}
        self._samples = {}
        self._references_used = None
        self._anchored = None

    # ---- the study's tissue references ----

    def references(self) -> TissueReferences | None:
        """Gray/white/CSF reference HU of the shown series from its segmented
        basal ganglia and ventricles (None until they are segmented). Each
        slice's contribution is computed once."""
        volume, cache = self._volume(), self._segmentation_cache()
        if volume is None or not cache:
            return None
        for index, result in cache.items():
            if index not in self._samples:
                self._samples[index] = slice_samples(volume.get_slice_hu(index), result.masks)
        return combine(self._samples[index] for index in cache)

    def _uses_references(self) -> bool:
        return self.method == METHOD_ANCHORED or (
            self.method == METHOD_ADAPTIVE and self.settings[METHOD_ADAPTIVE].use_study_range
        )

    def _current_references(self) -> TissueReferences | None:
        """The references - and, if they changed since the cached results were
        computed (more of the basal ganglia segmented), those results are dropped."""
        references = self.references() if self._uses_references() else None
        if references != self._references_used:
            self.cache = {}
            self._references_used = references
        return references

    def _anchored_model(self, references: TissueReferences | None) -> AnchoredModel | None:
        cache, volume = self._segmentation_cache(), self._volume()
        if references is None or not cache or volume is None:
            return None
        key = frozenset(cache)
        if self._anchored is None or self._anchored[0] != key:
            masks = {i: result.masks for i, result in cache.items()}
            brains = brain_mask_from_segmentation(masks, volume.num_slices)
            model = fit_anchored_model(masks, volume.get_slice_hu, brains, volume.pixel_spacing_mm, references)
            self._anchored = (key, model)
            self.cache = {}
        return self._anchored[1]

    # ---- results ----

    def brain_mask(self, index: int, slice_hu: np.ndarray) -> np.ndarray:
        """The slice's brain mask: from the anatomy segmentation (merged labels,
        largest 3D component) when the slice has one, else the HU-threshold
        skull-strip (`extract_brain_mask`)."""
        cache = self._segmentation_cache()
        if cache:
            masks = {i: result.masks for i, result in cache.items()}
            by_slice = brain_mask_from_segmentation(masks, self._volume().num_slices)
            if index in by_slice:
                return by_slice[index]
        threshold = self.settings[METHOD_THRESHOLD]
        return extract_brain_mask(slice_hu, threshold.air_hu_cutoff, threshold.bone_hu_cutoff)

    def result(self, index: int):
        """The slice's `BrainTissueResult`, computed and cached on first request
        (None without a volume)."""
        volume = self._volume()
        if volume is None:
            return None
        classify = self.classifier()  # may drop stale results (references changed)
        if index not in self.cache:
            slice_hu = volume.get_slice_hu(index)
            self.cache[index] = classify(slice_hu, self.brain_mask(index, slice_hu), index)
        return self.cache[index]

    def classifier(self) -> Callable[..., object]:
        """`(slice_hu, brain_mask, slice_index=None) -> BrainTissueResult` for the
        current method and settings, snapshotted (incl. the study references and
        the prototype's drift model) - safe to run on a worker thread while the
        user keeps changing things. The prototype needs the slice index; without
        it, or before the basal ganglia are segmented, it falls back to Adaptive."""
        method = self.method
        threshold, fcm, adaptive = (self.settings[m] for m in (METHOD_THRESHOLD, METHOD_FCM, METHOD_ADAPTIVE))
        references = self._current_references()
        model = self._anchored_model(references) if method == METHOD_ANCHORED else None
        volume = self._volume()
        spacing = volume.pixel_spacing_mm if volume is not None and adaptive.smoothing_mm > 0 else None
        sigma_px = adaptive.smoothing_mm / spacing if spacing else 0.0
        study_range = references.plausible_range() if references is not None and adaptive.use_study_range else None

        def classify(slice_hu: np.ndarray, brain_mask: np.ndarray, slice_index: int | None = None):
            if method == METHOD_ANCHORED and model is not None and slice_index is not None:
                result = classify_brain_tissue_anchored(slice_hu, brain_mask, model, slice_index)
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

    def _changed(self) -> None:
        self.cache = {}
        self.settings_changed.emit()
