"""PROTOTYPE - WM/GM/CSF separation anchored on the basal ganglia, with a drift model.

Within a scan the tissue baseline drifts by ~5 HU with the distance to the
skull and the slice level (beam hardening, skull partial volume) - nearly the
whole gray/white contrast (~6 HU). This method:

1. takes the study's references (`tissue_references`: gray = caudate/
   lentiform, white = internal capsule, CSF = ventricles);
2. models the drift as a smooth function of the distance to the brain
   surface d and the slice level z (terms 1, exp(-d/3 mm), exp(-d/10 mm), z,
   z^2, z*exp(-d/10 mm)), fitted to (smoothed HU - reference level of the
   current class) on a subsample of brain pixels of the whole series, in 5
   rounds of classify -> fit -> reclassify, and pinned to 0 on the pixels
   where the references were measured;
3. classifies the drift-corrected, lightly smoothed slice with thresholds
   midway between the references.

Status (2026-10-09): visually coherent white matter and deep nuclei; on 425
falx slices its ROI2 was the most homogeneous of three methods (median
structure SD 3.45 vs 3.50 current) and closest to the white-matter reference
(+0.7 vs +1.8 HU). Known weakness: drift and tissue class are confounded -
large areas (e.g. a posterior half, one hemisphere at high slices) can come
out as gray matter; there is no left-right or front-back drift term. Not
validated against a reader. Pure NumPy/SciPy, no I/O; slice indices 0-based.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from bictool.domain.metrics.basal_ganglia import CAUDATE, INTERNAL_CAPSULE, LENTIFORM, measure_slice
from bictool.domain.segmentation._tissue_cleanup import clean_mask
from bictool.domain.segmentation.brain_tissue import BrainTissueResult
from bictool.domain.segmentation.tissue_references import TissueReferences

SMOOTHING_MM = 1.0
ROUNDS = 5
MIN_COMPONENT_SIZE = 20
_SAMPLES_PER_SLICE = 3000
_MIN_BRAIN_PIXELS = 2000
_Z_SCALE = 10.0  # slices -> design-matrix units


def _design(d: np.ndarray, z: np.ndarray) -> np.ndarray:
    near, mid = np.exp(-d / 3.0), np.exp(-d / 10.0)
    return np.column_stack([np.ones_like(d), near, mid, z, z ** 2, z * mid])


@dataclass(frozen=True)
class AnchoredModel:
    references: TissueReferences
    coefficients: np.ndarray   # drift model (see `_design`)
    reference_slice: float     # slice level of the basal ganglia (z = 0)
    pixel_spacing_mm: float

    def drift(self, brain_mask: np.ndarray, slice_index: int) -> np.ndarray:
        d = ndi.distance_transform_edt(brain_mask) * self.pixel_spacing_mm
        z = np.full(d.size, (slice_index - self.reference_slice) / _Z_SCALE)
        return (_design(d.ravel(), z) @ self.coefficients).reshape(d.shape)


def fit_anchored_model(
    masks_by_slice: dict[int, dict[str, np.ndarray]],
    get_slice_hu: Callable[[int], np.ndarray],
    brain_by_slice: dict[int, np.ndarray],
    pixel_spacing_mm: float,
    references: TissueReferences,
) -> AnchoredModel | None:
    """The series' drift model, or None without basal-ganglia slices."""
    basal_slices = [
        index for index, masks in masks_by_slice.items()
        if (m := measure_slice(get_slice_hu(index), masks)) is not None and m.plausible
    ]
    if not basal_slices:
        return None
    z_ref = float(np.median(basal_slices))
    mid_cw = (references.csf + references.white) / 2.0
    mid_gw = (references.gray + references.white) / 2.0
    top = references.plausible_range()[1]
    sigma = SMOOTHING_MM / pixel_spacing_mm
    rng = np.random.default_rng(0)
    values, depths, levels, ref_depths, ref_levels = [], [], [], [], []
    for index, brain in sorted(brain_by_slice.items()):
        if brain.sum() < _MIN_BRAIN_PIXELS:
            continue
        smooth = ndi.gaussian_filter(np.asarray(get_slice_hu(index), dtype=np.float64), sigma)
        d = ndi.distance_transform_edt(brain) * pixel_spacing_mm
        pixels = np.flatnonzero(brain & (smooth > mid_cw) & (smooth < top))
        if pixels.size:
            take = rng.choice(pixels, size=min(pixels.size, _SAMPLES_PER_SLICE), replace=False)
            values.append(smooth.flat[take])
            depths.append(d.flat[take])
            levels.append(np.full(take.size, index - z_ref))
        for name in (LENTIFORM, CAUDATE, INTERNAL_CAPSULE):
            mask = masks_by_slice.get(index, {}).get(name)
            if mask is not None and mask.any():
                core = ndi.binary_erosion(mask)
                ref_depths.append(d[core])
                ref_levels.append(np.full(int(core.sum()), index - z_ref))
    if not values or not ref_depths:
        return None
    s = np.concatenate(values)
    design = _design(np.concatenate(depths), np.concatenate(levels) / _Z_SCALE)
    ref_design = _design(np.concatenate(ref_depths), np.concatenate(ref_levels) / _Z_SCALE)
    coefficients = np.zeros(design.shape[1])
    for _ in range(ROUNDS):
        level = np.where(s - design @ coefficients >= mid_gw, references.gray, references.white)
        coefficients = np.linalg.lstsq(design, np.clip(s - level, -10.0, 10.0), rcond=None)[0]
        coefficients[0] -= float(np.mean(ref_design @ coefficients))  # no drift where the references were measured
    return AnchoredModel(references, coefficients, z_ref, pixel_spacing_mm)


def classify_brain_tissue_anchored(
    slice_hu: np.ndarray, brain_mask: np.ndarray, model: AnchoredModel, slice_index: int
) -> BrainTissueResult:
    refs = model.references
    smooth = ndi.gaussian_filter(np.asarray(slice_hu, dtype=np.float64), SMOOTHING_MM / model.pixel_spacing_mm)
    corrected = smooth - model.drift(brain_mask, slice_index)
    mid_cw, mid_gw, top = (refs.csf + refs.white) / 2.0, (refs.gray + refs.white) / 2.0, refs.plausible_range()[1]
    csf = brain_mask & (corrected < mid_cw)
    white = brain_mask & (corrected >= mid_cw) & (corrected < mid_gw)
    gray = brain_mask & (corrected >= mid_gw) & (corrected < top)
    return BrainTissueResult(masks={
        "CSF": clean_mask(csf, MIN_COMPONENT_SIZE),
        "White Matter": clean_mask(white, MIN_COMPONENT_SIZE),
        "Gray Matter": clean_mask(gray, MIN_COMPONENT_SIZE),
    })
