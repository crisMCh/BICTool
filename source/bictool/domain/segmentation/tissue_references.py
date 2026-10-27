"""Per-study gray-matter, white-matter and CSF reference HU from the CNN anatomy labels.

The absolute HU of brain tissue differs by up to ~17 HU between scanners and
protocols (caudate 27-44 HU across our 16 studies), so fixed HU ranges for
WM/GM separation fail on some scans. The basal ganglia give per-study
references: deep gray matter has practically the same HU as cortical gray
matter (literature: 1.5-3 HU denser - Metter et al., Resuscitation 2011;
82:1180-1185; our data: caudate/lentiform within ~1 HU of the insular
cortex), the internal capsule is white matter, the ventricles are CSF.

* gray  = caudate + lentiform nucleus, only on slices where they are plausible
          (`metrics.basal_ganglia` - a CNN label on metal streaks or blood is skipped)
* white = internal capsule on those slices
* csf   = ventricles (eroded 2 px, -10..30 HU)
Each value is the median of the per-region means (regions eroded, -10..80 HU).

Samples are collected per slice (`slice_samples`), so a caller can add newly
segmented slices without recomputing the others. Pure NumPy/SciPy, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from bictool.domain.metrics.basal_ganglia import CAUDATE, INTERNAL_CAPSULE, LENTIFORM, measure_slice

VENTRICLES = "Ventricular System"
GRAY_MARGIN_HU = 15.0  # above the gray reference: not brain tissue (blood, calcification)
_MIN_PIXELS = 40


@dataclass(frozen=True)
class TissueReferences:
    gray: float
    white: float
    csf: float

    def plausible_range(self) -> tuple[float, float]:
        """Brain-tissue HU range of this study: midway CSF/white matter .. gray + 15 HU."""
        return (self.csf + self.white) / 2.0, self.gray + GRAY_MARGIN_HU


@dataclass(frozen=True)
class SliceSamples:
    gray: tuple[float, ...] = ()
    white: tuple[float, ...] = ()
    csf: tuple[float, ...] = ()
    basal_ganglia: bool = False  # the slice has plausible basal-ganglia labels


def _region_mean(slice_hu: np.ndarray, mask: np.ndarray | None, low: float, high: float, erode: int) -> float | None:
    if mask is None:
        return None
    core = ndi.binary_erosion(mask, iterations=erode) & (slice_hu > low) & (slice_hu < high)
    return float(slice_hu[core].mean()) if core.sum() >= _MIN_PIXELS else None


def slice_samples(slice_hu: np.ndarray, masks: dict[str, np.ndarray]) -> SliceSamples:
    """One slice's contribution to the references."""
    slice_hu = np.asarray(slice_hu, dtype=np.float64)
    csf = _region_mean(slice_hu, masks.get(VENTRICLES), -10.0, 30.0, 2)
    basal = measure_slice(slice_hu, masks)
    if basal is None or not basal.plausible:
        return SliceSamples(csf=(csf,) if csf is not None else ())
    gray = tuple(v for v in (_region_mean(slice_hu, masks.get(n), -10.0, 80.0, 1) for n in (LENTIFORM, CAUDATE))
                 if v is not None)
    white = _region_mean(slice_hu, masks.get(INTERNAL_CAPSULE), -10.0, 80.0, 1)
    return SliceSamples(gray=gray, white=(white,) if white is not None else (),
                        csf=(csf,) if csf is not None else (), basal_ganglia=True)


def combine(samples: Iterable[SliceSamples]) -> TissueReferences | None:
    """The study's references, or None if any of the three is missing."""
    samples = list(samples)
    gray = [v for s in samples for v in s.gray]
    white = [v for s in samples for v in s.white]
    csf = [v for s in samples for v in s.csf]
    if not gray or not white or not csf:
        return None
    return TissueReferences(gray=float(np.median(gray)), white=float(np.median(white)), csf=float(np.median(csf)))


def estimate_tissue_references(
    masks_by_slice: dict[int, dict[str, np.ndarray]], get_slice_hu: Callable[[int], np.ndarray]
) -> TissueReferences | None:
    return combine(slice_samples(get_slice_hu(index), masks) for index, masks in masks_by_slice.items())
