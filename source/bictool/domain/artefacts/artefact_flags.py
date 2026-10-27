"""Possible-artefact flags of a study: metal, and dark bands across the brainstem.

Descriptive - they point the user to slices worth a look. One exception: metal
streaks that reach the brain within the slice range of the verdict make the study
verdict Not assessable (`domain.results.head_ct_verdict.apply_metal_rule`).

* **Metal**: a slice with at least `MIN_METAL_PIXELS` connected pixels above
  `METAL_HU`. Dense cortical bone reached 2330 HU in our data (photon-counting
  CT); metal in the two studies with metal was 3071 HU (scanner ceiling) and
  4224 HU - so 2800 HU leaves a margin on both sides. Metal is a source of
  streaks, not proof of a bad image.
* **Metal streaks in the brain**: on a metal slice, the share of brain tissue
  (CNN brain without CSF spaces and dural folds, eroded `_STREAK_ERODE_MM`)
  within `METAL_STREAK_REACH_MM` of the metal that differs by more than
  `STREAK_DEVIATION_HU` from the slice's brain median. From
  `METAL_STREAK_SHARE` the streaks reach the brain. Evidence (2026-10-21, the
  two studies with metal, docs/qa_method/artefact_flags.md): large implants
  with streaks rated not acceptable 4-98 % on 7 of 9 metal slices; small
  craniotomy screws without visible streaks in a study rated acceptable at
  most 0.2 %. Provisional.
* **Brainstem dark band**: beam hardening between the petrous bones shows as a
  dark band running left-right across the pons (Barrett & Keat 2004,
  RadioGraphics 24:1679-1691). On posterior-fossa slices the brainstem is the
  CNN's "Brain Parenchyma" piece that touches the cerebellum; per image row
  (anterior to posterior) its mean HU is taken, and the band depth is how much
  darker the darkest row is than its 4 mm-smoothed surroundings. Against
  blinded slice ratings (non-expert, 16 studies) it separated streak slices
  from the rest at the same level with AUC 0.75 (strong streaks 0.80); the same
  measure front-to-back did not (0.5). `BAND_DEPTH_HU` lies between the median
  of mild (3.2 HU) and strong (4.1 HU) slices - provisional, to be confirmed on
  new rated data.

Pure NumPy/SciPy, no I/O. Slice indices are 0-based.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

METAL_HU = 2800.0
MIN_METAL_PIXELS = 3
BAND_DEPTH_HU = 3.5  # provisional
METAL_STREAK_REACH_MM = 30.0
STREAK_DEVIATION_HU = 30.0
METAL_STREAK_SHARE = 0.05  # provisional
_STREAK_ERODE_MM = 2.0      # away from the skull (partial volume with bone)
_MIN_STREAK_AREA_MM2 = 20.0  # less brain near the metal: not judged

# Posterior fossa: the cerebellum is at least this share of the labelled brain.
MIN_CEREBELLUM_FRACTION = 0.15
_TOUCH_MM = 3.0          # brainstem piece within this distance of the cerebellum
_ERODE_MM = 1.0          # away from the brainstem border (partial volume)
_SOFT_TISSUE_HU = (0.0, 70.0)
_MIN_ROW_FRACTION = 0.4  # rows narrower than this share of the widest row are tips (cisterns)
_END_MARGIN_MM = 3.0     # front and back end of the brainstem left out
_MIN_ROWS = 10
_MIN_CORE_WIDTH_MM = 3.0  # thinner pieces are not a brainstem cross-section
_TREND_MM = 4.0

CEREBELLUM = "Cerebellum"
BRAIN_PARENCHYMA = "Brain Parenchyma"
_NOT_TISSUE = ("Ventricular System", "Subarachnoid Space", "Septum Pellucidum", "Dural Folds")


@dataclass(frozen=True)
class ArtefactFlags:
    metal_slices: tuple[int, ...] = ()           # slices with metal
    metal_max_hu: float | None = None            # highest HU among them
    band_depths: dict[int, float] = field(default_factory=dict)  # posterior-fossa slice -> band depth (HU)
    metal_streak_shares: dict[int, float] = field(default_factory=dict)  # metal slice -> disturbed brain share

    @property
    def metal_streak_slices(self) -> tuple[int, ...]:
        """Slices where metal streaks reach the brain."""
        return tuple(sorted(index for index, share in self.metal_streak_shares.items() if share >= METAL_STREAK_SHARE))

    @property
    def band_slices(self) -> tuple[int, ...]:
        """Slices with a possible dark band across the brainstem."""
        return tuple(sorted(index for index, depth in self.band_depths.items() if depth >= BAND_DEPTH_HU))

    def on_slice(self, index: int) -> tuple[str, ...]:
        """Short names of the possible artefacts on one slice ("metal", "possible brainstem band")."""
        metal = "metal streaks in the brain" if index in self.metal_streak_slices else "metal"
        return tuple(
            name for name, slices in ((metal, self.metal_slices), ("possible brainstem band", self.band_slices))
            if index in slices
        )


def metal_pixels(slice_hu: np.ndarray) -> np.ndarray:
    """Pixels of metal pieces (connected, at least `MIN_METAL_PIXELS`) on one slice."""
    candidates = slice_hu > METAL_HU
    if not candidates.any():
        return candidates
    labels, count = ndi.label(candidates)
    sizes = ndi.sum(candidates, labels, range(1, count + 1))
    keep = np.flatnonzero(np.asarray(sizes) >= MIN_METAL_PIXELS) + 1
    return np.isin(labels, keep)


def metal_streak_share(
    slice_hu: np.ndarray, metal: np.ndarray, masks: dict[str, np.ndarray], pixel_spacing_mm: float
) -> float | None:
    """Share of the brain tissue near the metal that differs by more than
    `STREAK_DEVIATION_HU` from the slice's brain median; None without enough
    brain near the metal."""
    tissue = np.zeros(metal.shape, dtype=bool)
    for name, mask in masks.items():
        if mask is not None and name not in _NOT_TISSUE:
            tissue |= mask
    for name in _NOT_TISSUE:
        if masks.get(name) is not None:
            tissue &= ~masks[name]
    tissue = ndi.binary_erosion(tissue, iterations=max(1, round(_STREAK_ERODE_MM / pixel_spacing_mm)))
    near = tissue & (ndi.distance_transform_edt(~metal, sampling=pixel_spacing_mm) <= METAL_STREAK_REACH_MM)
    if near.sum() * pixel_spacing_mm**2 < _MIN_STREAK_AREA_MM2:
        return None
    median = float(np.median(slice_hu[tissue]))
    return float(np.mean(np.abs(slice_hu[near] - median) > STREAK_DEVIATION_HU))


def posterior_fossa_slices(masks_by_slice: dict[int, dict[str, np.ndarray]]) -> list[int]:
    """Segmented slices where the cerebellum is >= 15 % of the labelled brain,
    in the skull-base half of the scan (the CNN puts a few cerebellum pixels on
    some high slices, too)."""
    indices = sorted(masks_by_slice)
    if not indices:
        return []
    areas = {i: _area(masks_by_slice[i].get(CEREBELLUM)) for i in indices}
    half = len(indices) // 2
    first, second = indices[:half], indices[half:]
    base = set(first) if sum(areas[i] for i in first) >= sum(areas[i] for i in second) else set(second)
    out = []
    for index in indices:
        labelled = sum(_area(mask) for mask in masks_by_slice[index].values())
        if index in base and labelled and areas[index] / labelled >= MIN_CEREBELLUM_FRACTION:
            out.append(index)
    return out


def find_brainstem(masks: dict[str, np.ndarray], pixel_spacing_mm: float) -> np.ndarray | None:
    """The largest "Brain Parenchyma" piece touching the cerebellum (posterior-fossa slices)."""
    cerebellum, parenchyma = masks.get(CEREBELLUM), masks.get(BRAIN_PARENCHYMA)
    if cerebellum is None or parenchyma is None or not cerebellum.any() or not parenchyma.any():
        return None
    labels, _count = ndi.label(parenchyma)
    near = ndi.binary_dilation(cerebellum, iterations=max(1, round(_TOUCH_MM / pixel_spacing_mm)))
    touching = np.unique(labels[near & parenchyma])
    touching = touching[touching > 0]
    if touching.size == 0:
        return None
    sizes = ndi.sum(parenchyma, labels, touching)
    stem = labels == touching[int(np.argmax(sizes))]
    if not ndi.binary_erosion(stem, iterations=max(1, round(_MIN_CORE_WIDTH_MM / 2 / pixel_spacing_mm))).any():
        return None  # a thin strip (above the posterior fossa), not a cross-section
    return stem


def band_depth(slice_hu: np.ndarray, brainstem: np.ndarray, pixel_spacing_mm: float) -> float | None:
    """How much darker (HU) the darkest left-right row of the brainstem is than
    its surroundings; None if the brainstem is too small to judge."""
    core = (
        ndi.binary_erosion(brainstem, iterations=max(1, round(_ERODE_MM / pixel_spacing_mm)))
        & (slice_hu > _SOFT_TISSUE_HU[0])
        & (slice_hu < _SOFT_TISSUE_HU[1])
    )
    counts = core.sum(axis=1)
    if counts.max(initial=0) == 0:
        return None
    usable = counts >= max(8, _MIN_ROW_FRACTION * counts.max())
    rows = np.flatnonzero(usable)
    margin = round(_END_MARGIN_MM / pixel_spacing_mm)
    if rows.size:
        usable[: rows[0] + margin] = False
        usable[rows[-1] - margin + 1:] = False
    rows = np.flatnonzero(usable)
    if rows.size < _MIN_ROWS:
        return None
    profile = (np.where(core, slice_hu, 0.0).sum(axis=1)[rows]) / counts[rows]
    trend = ndi.gaussian_filter1d(profile, _TREND_MM / pixel_spacing_mm, mode="nearest")
    return float(np.max(trend - profile))


def evaluate_artefact_flags(
    masks_by_slice: dict[int, dict[str, np.ndarray]],
    get_slice_hu: Callable[[int], np.ndarray],
    num_slices: int,
    pixel_spacing_mm: float,
) -> ArtefactFlags:
    """Metal on every slice (streaks in the brain on the segmented ones); brainstem
    bands on the segmented posterior-fossa slices."""
    fossa = {
        index: find_brainstem(masks_by_slice[index], pixel_spacing_mm)
        for index in posterior_fossa_slices(masks_by_slice)
    }
    metal, metal_max, depths, streaks = [], None, {}, {}
    for index in range(num_slices):
        slice_hu = np.asarray(get_slice_hu(index), dtype=np.float64)
        pieces = metal_pixels(slice_hu)
        if pieces.any():
            metal.append(index)
            top = float(slice_hu[pieces].max())
            metal_max = top if metal_max is None else max(metal_max, top)
            if index in masks_by_slice:
                share = metal_streak_share(slice_hu, pieces, masks_by_slice[index], pixel_spacing_mm)
                if share is not None:
                    streaks[index] = share
        stem = fossa.get(index)
        if stem is not None:
            depth = band_depth(slice_hu, stem, pixel_spacing_mm)
            if depth is not None:
                depths[index] = depth
    return ArtefactFlags(
        metal_slices=tuple(metal), metal_max_hu=metal_max, band_depths=depths, metal_streak_shares=streaks
    )


def _area(mask: np.ndarray | None) -> int:
    return int(mask.sum()) if mask is not None else 0
