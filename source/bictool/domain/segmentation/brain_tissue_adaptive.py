"""Adaptive (locally-varying) HU threshold white/gray matter
classification - a lighter-weight alternative to `brain_tissue.classify_
brain_tissue`'s single global HU band, kept in its own module for
side-by-side comparison (2026-09-24 request) against that and the other
two alternative methods (`brain_tissue_fcm.py`, `brain_tissue_gmm_mrf.py`).

Real CT slices can have a slowly-varying intensity baseline across the
brain (beam-hardening, detector inhomogeneity) that a single global cutoff
can't track. This computes a *local* mean HU (via a masked box filter, so
pixels outside `brain_mask` never bias a nearby region's average) over a
`block_size` neighborhood and classifies each brain-mask pixel relative to
*that* local baseline rather than one fixed split point - still restricted
to a permissive global HU band first (`plausible_range`), so a genuinely
non-tissue pixel (CSF, a stray vessel) is never reclassified purely
because of where the local average happens to sit.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from bictool.domain.segmentation._tissue_cleanup import clean_mask
from bictool.domain.segmentation.brain_tissue import BrainTissueResult

DEFAULT_BLOCK_SIZE = 31
DEFAULT_PLAUSIBLE_RANGE = (15.0, 50.0)
DEFAULT_MIN_COMPONENT_SIZE = 20


def _local_mean(slice_hu: np.ndarray, brain_mask: np.ndarray, block_size: int) -> np.ndarray:
    """Masked local mean: `sum(slice_hu * mask) / sum(mask)` over a
    `block_size` window, so pixels outside `brain_mask` never pull a
    nearby in-mask region's average toward skull/air/background HU.
    """
    mask_f = brain_mask.astype(np.float64)
    sum_values = ndimage.uniform_filter(slice_hu * mask_f, size=block_size, mode="constant")
    counts = ndimage.uniform_filter(mask_f, size=block_size, mode="constant")
    counts[counts == 0] = 1e-9
    return sum_values / counts


def classify_brain_tissue_adaptive(
    slice_hu: np.ndarray,
    brain_mask: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    plausible_range: tuple[float, float] = DEFAULT_PLAUSIBLE_RANGE,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> BrainTissueResult:
    low, high = plausible_range
    in_band = brain_mask & (slice_hu >= low) & (slice_hu < high)
    # Below `plausible_range`'s own floor, within the brain mask, is CSF/
    # subarachnoid space rather than "neither" - exposing it as its own
    # class (2026-09-29) fixes a real reported issue where CSF pixels near
    # `low` were occasionally pulled into `in_band` (partial-volume/noise)
    # and read as White Matter with no way to see or correct for it.
    csf_raw = brain_mask & (slice_hu < low)

    local_cutoff = _local_mean(slice_hu, brain_mask, block_size)

    white_matter_raw = in_band & (slice_hu < local_cutoff)
    gray_matter_raw = in_band & (slice_hu >= local_cutoff)

    csf = clean_mask(csf_raw, min_component_size)
    white_matter = clean_mask(white_matter_raw, min_component_size)
    gray_matter = clean_mask(gray_matter_raw, min_component_size)
    return BrainTissueResult(masks={"CSF": csf, "White Matter": white_matter, "Gray Matter": gray_matter})
