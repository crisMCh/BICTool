"""Basal-ganglia visibility per slice - descriptive, not part of the study verdict.

WP2 (§3.3) names the basal ganglia as a low-contrast, diagnostically important
region (early stroke shows as a fading border to the white matter). The CNN
labels the lentiform and caudate nuclei, but it also draws them where the
image shows none (it fills them in from shape and position when their
contrast is removed), so the label only says *where to look*; visibility is
measured in the image:

* **border contrast** (per nucleus): mean HU 2 px inside minus 2 px outside
  the nucleus border (soft tissue 0-60 HU only), divided by the pooled SD of
  the nucleus and the internal capsule (both eroded 1 px); the slice value is
  the mean over the lentiform and caudate nuclei that are labelled;
* **anatomical block**: only the contiguous block of slices around the largest
  lentiform label is measured (`basal_ganglia_block`) - the CNN also draws
  nuclei on slices far from them (near the vertex, at the skull base);
* **plausibility** (provisional): on slices a reader rated visible, a nucleus
  was 4.6-8.6 HU denser than the internal capsule (5-95 %, max 10.6). A
  difference outside `PLAUSIBLE_DIFFERENCE_HU` means the label sits on
  something else (metal streaks, blood) - the slice then counts as not visible.

Evidence (2026-10-09, docs/qa_method/basal_ganglia.md): blinded slice ratings
by a non-clinician (16 studies, 157 slices) - the border contrast rose with
the rating (medians 0.07 / 0.43 / 0.75 / 0.88 for not visible / hardly /
visible / clear) and separated "clear" from "hardly" (AUC 0.85) where the
label area did not (0.64). Cut points and the plausibility range are
provisional. Pure NumPy/SciPy, no I/O; slice indices 0-based.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

LENTIFORM, CAUDATE, INTERNAL_CAPSULE = "Lentiform Nucleus", "Caudate Nucleus", "Internal Capsule"
NUCLEI = (LENTIFORM, CAUDATE)
MIN_PIXELS = 30                          # per region after erosion
PLAUSIBLE_DIFFERENCE_HU = (2.0, 15.0)    # nucleus - internal capsule (provisional)
THALAMUS = "Thalamus"
DEEP_GREY = (LENTIFORM, CAUDATE, THALAMUS, INTERNAL_CAPSULE)
# Anatomical block: neighbouring slices count while their deep grey matter is at least this fraction of the
# study's largest (2026-10-20: separates the real block from stray labels near the vertex or skull base in
# all 15 rated studies; docs/qa_method/basal_ganglia.md). Provisional.
BLOCK_FRACTION = 0.08
_SOFT_TISSUE_HU = (0.0, 60.0)
_RING_PX = 2

NOT_VISIBLE, HARDLY, VISIBLE, CLEARLY = "not visible", "hardly visible", "visible", "clearly visible"
NOT_PLAUSIBLE = "label not plausible"
# Border-contrast cut points (provisional), between the medians of the rating levels.
_LEVELS = ((0.85, CLEARLY), (0.6, VISIBLE), (0.25, HARDLY))


@dataclass(frozen=True)
class BasalGangliaSlice:
    border: float                         # mean border contrast of the labelled nuclei (0 if implausible)
    differences: dict[str, float]         # nucleus -> HU difference to the internal capsule
    plausible: bool

    @property
    def level(self) -> str:
        if not self.plausible:
            return NOT_PLAUSIBLE
        for cut, name in _LEVELS:
            if self.border >= cut:
                return name
        return NOT_VISIBLE

    @property
    def visible(self) -> bool:
        return self.level in (VISIBLE, CLEARLY)


def nucleus_contrast(slice_hu: np.ndarray, nucleus: np.ndarray, capsule: np.ndarray) -> tuple[float, float] | None:
    """(HU difference to the internal capsule, border contrast) of one nucleus,
    or None if either region is too small."""
    core, capsule_core = ndi.binary_erosion(nucleus), ndi.binary_erosion(capsule)
    if core.sum() < MIN_PIXELS or capsule_core.sum() < MIN_PIXELS:
        return None
    difference = float(slice_hu[core].mean() - slice_hu[capsule_core].mean())
    sd = float(np.sqrt((slice_hu[core].var() + slice_hu[capsule_core].var()) / 2))
    soft = (slice_hu > _SOFT_TISSUE_HU[0]) & (slice_hu < _SOFT_TISSUE_HU[1])
    inner = nucleus & ~ndi.binary_erosion(nucleus, iterations=_RING_PX) & soft
    outer = ndi.binary_dilation(nucleus, iterations=_RING_PX) & ~nucleus & soft
    if sd <= 0 or inner.sum() <= 20 or outer.sum() <= 20:
        return difference, 0.0
    return difference, float((slice_hu[inner].mean() - slice_hu[outer].mean()) / sd)


def measure_slice(slice_hu: np.ndarray, masks: dict[str, np.ndarray]) -> BasalGangliaSlice | None:
    """Visibility of the basal ganglia on one slice; None without labelled nuclei."""
    capsule = masks.get(INTERNAL_CAPSULE)
    if capsule is None:
        return None
    slice_hu = np.asarray(slice_hu, dtype=np.float64)
    differences, borders = {}, []
    for name in NUCLEI:
        nucleus = masks.get(name)
        if nucleus is None:
            continue
        measured = nucleus_contrast(slice_hu, nucleus, capsule)
        if measured is not None:
            differences[name], border = measured
            borders.append(border)
    if not borders:
        return None
    low, high = PLAUSIBLE_DIFFERENCE_HU
    plausible = all(low <= difference <= high for difference in differences.values())
    return BasalGangliaSlice(border=float(np.mean(borders)) if plausible else 0.0, differences=differences, plausible=plausible)


def _area(masks: dict[str, np.ndarray] | None, name: str) -> int:
    mask = masks.get(name) if masks else None
    return int(mask.sum()) if mask is not None else 0


def basal_ganglia_block(masks_by_slice: dict[int, dict[str, np.ndarray]]) -> set[int]:
    """The slices that anatomically contain the basal ganglia: one contiguous block of slices around the slice
    with the most deep grey matter (CNN lentiform + caudate + thalamus + internal capsule), extended up and down
    while a neighbouring slice keeps at least `BLOCK_FRACTION` of that. The nuclei are one compact structure in
    the centre of the brain, so nucleus labels beyond the block (e.g. near the vertex, where the CNN sometimes
    draws them) are not counted. The thalamus is included because it lies at the same height and keeps the
    block together where the CNN under-labels the nuclei on one slice. A non-segmented slice ends the block."""
    deep = {i: sum(_area(m, name) for name in DEEP_GREY) for i, m in masks_by_slice.items()}
    if not deep or max(deep.values()) == 0 or not any(_area(m, n) for m in masks_by_slice.values() for n in NUCLEI):
        return set()
    start = max(deep, key=lambda i: (deep[i], -i))
    threshold = BLOCK_FRACTION * deep[start]
    block = {start}
    for step in (-1, 1):
        i = start + step
        while i in deep and deep[i] >= threshold:
            block.add(i)
            i += step
    return block


def evaluate_basal_ganglia(
    masks_by_slice: dict[int, dict[str, np.ndarray]], get_slice_hu: Callable[[int], np.ndarray]
) -> dict[int, BasalGangliaSlice]:
    """Every slice of the basal-ganglia block (`basal_ganglia_block`) with labelled nuclei -> its visibility."""
    block = basal_ganglia_block(masks_by_slice)
    out = {}
    for index, masks in sorted(masks_by_slice.items()):
        if index not in block:
            continue
        if not any(masks.get(name) is not None and masks[name].any() for name in NUCLEI):
            continue
        measured = measure_slice(get_slice_hu(index), masks)
        if measured is not None:
            out[index] = measured
    return out
