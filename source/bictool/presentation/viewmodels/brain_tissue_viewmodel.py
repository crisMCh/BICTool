"""BrainTissueViewModel: WM/GM/CSF separation of the shown series.

Owns the chosen method (Adaptive Thresholding by default, Fuzzy C-Means,
HU Thresholding), each method's settings, and the per-slice results. The
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
from bictool.domain.segmentation.brain_tissue_fcm import classify_brain_tissue_fcm
from bictool.domain.segmentation.segmentation_brain_mask import brain_mask_from_segmentation

METHOD_FCM = "Fuzzy C-Means"
METHOD_THRESHOLD = "HU Thresholding"
METHOD_ADAPTIVE = "Adaptive Thresholding"
METHODS = [METHOD_FCM, METHOD_THRESHOLD, METHOD_ADAPTIVE]
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

    def reset(self) -> None:
        """A new/closed series. The method and settings are app preferences and stay."""
        self.cache = {}

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
        if index not in self.cache:
            slice_hu = volume.get_slice_hu(index)
            self.cache[index] = self.classifier()(slice_hu, self.brain_mask(index, slice_hu))
        return self.cache[index]

    def classifier(self) -> Callable[[np.ndarray, np.ndarray], object]:
        """`(slice_hu, brain_mask) -> BrainTissueResult` for the current method
        and settings, snapshotted - safe to run on a worker thread while the
        user keeps changing things."""
        method = self.method
        threshold, fcm, adaptive = (self.settings[m] for m in (METHOD_THRESHOLD, METHOD_FCM, METHOD_ADAPTIVE))

        def classify(slice_hu: np.ndarray, brain_mask: np.ndarray):
            if method == METHOD_FCM:
                result = classify_brain_tissue_fcm(
                    slice_hu, brain_mask, fuzziness=fcm.fuzziness, max_iter=fcm.max_iter,
                    spatial_weight=fcm.spatial_weight, min_component_size=fcm.min_component_size,
                )
            elif method == METHOD_ADAPTIVE:
                result = classify_brain_tissue_adaptive(
                    slice_hu, brain_mask, block_size=adaptive.block_size,
                    plausible_range=(adaptive.plausible_low, adaptive.plausible_high),
                    min_component_size=adaptive.min_component_size,
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
