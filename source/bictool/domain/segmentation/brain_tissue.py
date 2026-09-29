"""HU-threshold-only brain-parenchyma extraction and white/gray matter
classification - deliberately not ML-based, unlike `head_ct_unet.py`.

`domain.segmentation.__init__`'s own docstring already lists "white/gray
matter" among the structures this sub-module should cover, per WP2 §3 and
`architecture/layers_description.md`'s "The ventricles and white matter can
be used for NPS analysis." This module implements that directly.

A single global HU cutoff over a raw slice is not reliable: the combined
CSF+white-matter+gray-matter histogram of a real head CT slice is close to
unimodal/overlapping (CSF ~10 HU, white matter ~30 HU, gray matter ~40 HU,
per published literature), and is dominated by the much larger HU range of
skull/scalp/background. So this module works in two stages, matching how
real skull-stripping pipelines are structured: `extract_brain_mask` first
isolates brain parenchyma (excluding air and bone/skull, then keeping only
the single largest connected blob - the brain is one contiguous region,
which is a more robust way to exclude the skull than trying to tune a
threshold that perfectly separates bone from soft tissue). Only then does
`classify_brain_tissue` apply the white-matter/gray-matter HU bands, safely
restricted to that mask.

`domain/preprocessing/_pixel_ops.py`'s `bwdist` was considered and is not
reused here - it is a signed-distance transform used by
`active_contour.py`'s Chan-Vese level-set math, not a general morphology
helper; the plain `scipy.ndimage.binary_*` operations used below are the
right primitives for this.

Both functions take a raw 2D HU array (e.g. `DicomVolume.get_slice_hu`
directly, no windowing/normalization) and are pure NumPy/SciPy - no Qt,
file I/O, or persistence, per the Domain Layer's own rules.

`classify_brain_tissue` also follows two more literature-standard steps to
avoid a speckled/salt-and-pepper result (confirmed necessary in practice,
2026-09-24 - a first version without these produced isolated misclassified
spots inside otherwise-uniform tissue, purely from ordinary per-pixel HU
noise crossing the WM/GM band boundary): a median-filter denoise pass on
the HU values *before* thresholding (reduces how often noise alone flips a
pixel's band), and a morphological open+close plus small-connected-
component removal on *each* output mask afterward (consolidates the
survivors into coherent regions instead of scattered dots). Both are
standard, lightweight post-processing steps for threshold-based medical
image segmentation - not a switch to a heavier method like a spatially-
regularized Gaussian Mixture Model/Markov Random Field, which the
literature also flags for this exact problem but which is a materially
bigger methodological jump than this HU-only feature calls for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Excludes air/background (below) and bone/calcification (above) from the
# coarse brain-vs-not-brain mask. Not itself precise enough to separate the
# skull from soft tissue everywhere - the largest-connected-component step
# in `extract_brain_mask` does that part.
DEFAULT_AIR_HU_CUTOFF = -500.0
DEFAULT_BONE_HU_CUTOFF = 100.0

# Structuring elements for the morphological clean-up steps below - small
# and fixed, this is a coarse skull-strip for a QC aid, not a research-grade
# pipeline.
_OPENING_STRUCTURE = np.ones((3, 3), dtype=bool)
_CLOSING_STRUCTURE = np.ones((5, 5), dtype=bool)

# Literature-typical HU bands for CSF, white, and gray matter (CSF ~10 HU,
# white matter ~30 HU, gray matter ~40 HU) - named/tunable rather than hidden
# in the function body, since real HU values drift by scanner/protocol and
# these may need retuning against real data. CSF's own band is exposed
# (2026-09-29) so subarachnoid-space/CSF pixels near the White Matter band's
# lower edge get their own explicit class instead of occasionally being
# misclassified as White Matter - a real reported issue across all three
# WM/GM methods.
DEFAULT_CSF_RANGE = (0.0, 20.0)
DEFAULT_WHITE_MATTER_RANGE = (20.0, 32.0)
DEFAULT_GRAY_MATTER_RANGE = (32.0, 45.0)

# Median-filter kernel size applied to the HU slice before band comparison -
# odd, small, matches the standard "denoise before threshold" practice
# without smearing real tissue boundaries as much as a Gaussian would.
DEFAULT_SMOOTHING_SIZE = 3

# Post-processing: a connected component of a class mask smaller than this
# many pixels is discarded as noise-driven speckle rather than a real region.
DEFAULT_MIN_COMPONENT_SIZE = 20


def extract_brain_mask(
    slice_hu: np.ndarray,
    air_hu_cutoff: float = DEFAULT_AIR_HU_CUTOFF,
    bone_hu_cutoff: float = DEFAULT_BONE_HU_CUTOFF,
) -> np.ndarray:
    """Skull-strips `slice_hu`, returning a 2D boolean mask (True = brain
    parenchyma) of the same shape.

    1. Threshold to `(air_hu_cutoff, bone_hu_cutoff]` - excludes air and
       bone/calcification.
    2. Binary opening - strips thin bridges connecting scalp/table noise to
       the brain.
    3. Keep only the largest connected component - the brain is one
       contiguous blob; this is the actual "skull-strip" step, more robust
       than trying to threshold the skull away perfectly.
    4. Fill holes - closes small calcification/vessel holes into a solid
       parenchyma mask.
    5. Binary closing - smooths the boundary.
    """
    coarse = (slice_hu > air_hu_cutoff) & (slice_hu <= bone_hu_cutoff)
    opened = ndimage.binary_opening(coarse, structure=_OPENING_STRUCTURE)

    labeled, num_labels = ndimage.label(opened)
    if num_labels == 0:
        return np.zeros_like(coarse, dtype=bool)
    component_sizes = ndimage.sum(opened, labeled, index=range(1, num_labels + 1))
    largest_label = int(np.argmax(component_sizes)) + 1
    largest_component = labeled == largest_label

    filled = ndimage.binary_fill_holes(largest_component)
    return ndimage.binary_closing(filled, structure=_CLOSING_STRUCTURE)


@dataclass
class BrainTissueResult:
    """`masks` mirrors `head_ct_unet.SegmentationResult`'s shape (one 2D
    boolean mask per named structure) so it stays compatible with
    `RoiViewport.set_overlay` and any future reuse of
    `decode_masks_for_checkboxes`-style helpers, even though this feature
    keeps its own independent UI/cache rather than joining the CNN's
    `_SEGMENTATION_MAPPING`.
    """

    masks: dict[str, np.ndarray]


def _clean_mask(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    """Standard threshold-segmentation clean-up: open+close to smooth
    boundaries and drop thin noise-driven protrusions, then discard any
    remaining connected component smaller than `min_component_size` - the
    two lightweight steps that turn a speckled raw threshold result into
    coherent regions (see this module's docstring).
    """
    opened = ndimage.binary_opening(mask, structure=_OPENING_STRUCTURE)
    # A larger structuring element than the opening step - standard practice
    # for this pairing: closing with a *bigger* kernel bridges wider gaps
    # between nearby same-class pixels, merging them into one region, rather
    # than only smoothing each region's own boundary. Fixes a real reported
    # issue (2026-09-24): using the same small kernel for both steps left
    # gray matter fragmented into isolated patches instead of united regions.
    closed = ndimage.binary_closing(opened, structure=_CLOSING_STRUCTURE)

    labeled, num_labels = ndimage.label(closed)
    if num_labels == 0:
        return closed
    component_sizes = ndimage.sum(closed, labeled, index=range(1, num_labels + 1))
    keep_labels = {i + 1 for i, size in enumerate(component_sizes) if size >= min_component_size}
    return np.isin(labeled, list(keep_labels))


def classify_brain_tissue(
    slice_hu: np.ndarray,
    brain_mask: np.ndarray,
    csf_range: tuple[float, float] = DEFAULT_CSF_RANGE,
    white_matter_range: tuple[float, float] = DEFAULT_WHITE_MATTER_RANGE,
    gray_matter_range: tuple[float, float] = DEFAULT_GRAY_MATTER_RANGE,
    smoothing_size: int = DEFAULT_SMOOTHING_SIZE,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> BrainTissueResult:
    """Classifies every pixel of `slice_hu` within `brain_mask` as CSF, white
    matter, gray matter, or neither, by HU-band membership on a denoised
    copy of `slice_hu`, then cleans up each resulting mask (see
    `_clean_mask`).

    A pixel outside `brain_mask` is never classified, regardless of its HU
    value - this is what makes the classification safe to apply directly
    (skull, CSF, and background can share HU values with white/gray matter
    in isolation, but not once skull-stripped). `smoothing_size`/
    `min_component_size` are exposed (not hidden constants) since how much
    consolidation is appropriate can reasonably vary with real slice
    resolution/noise.
    """
    smoothed_hu = ndimage.median_filter(slice_hu, size=smoothing_size)

    csf_low, csf_high = csf_range
    white_low, white_high = white_matter_range
    gray_low, gray_high = gray_matter_range

    csf_raw = brain_mask & (smoothed_hu >= csf_low) & (smoothed_hu < csf_high)
    white_matter_raw = brain_mask & (smoothed_hu >= white_low) & (smoothed_hu < white_high)
    gray_matter_raw = brain_mask & (smoothed_hu >= gray_low) & (smoothed_hu < gray_high)

    csf = _clean_mask(csf_raw, min_component_size)
    white_matter = _clean_mask(white_matter_raw, min_component_size)
    gray_matter = _clean_mask(gray_matter_raw, min_component_size)

    return BrainTissueResult(masks={"CSF": csf, "White Matter": white_matter, "Gray Matter": gray_matter})
