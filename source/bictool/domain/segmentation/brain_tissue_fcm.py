"""Spatial Fuzzy C-Means (FCM) white/gray matter classification - an
alternative to `brain_tissue.classify_brain_tissue`'s hard HU-threshold
approach, kept in its own module so the two (and the other two alternative
methods below) can be compared side by side (2026-09-24 request) rather
than one replacing the others.

Standard FCM (Bezdek et al. 1984) assigns each brain-mask pixel a *soft*
membership to each of 3 clusters based on HU value alone, then a spatial
regularization pass (Chuang et al. 2006-style: blend each pixel's own
membership with its local neighborhood's average membership) runs before
the final hard label is taken - this is what makes it more robust to the
same per-pixel noise that produces the speckled look a hard threshold is
prone to, without a separate denoise-then-cleanup step doing all the work.

3 clusters (not 2) because CT brain tissue's own intensity distribution has
three real peaks - CSF ~10 HU, white matter ~30 HU, gray matter ~40 HU, per
published literature (see `brain_tissue.py`'s own docstring for the
citations) - a 2-cluster fit would blur two of them together. Cluster
centers are sorted ascending and assigned CSF/other -> White Matter ->
Gray Matter, matching that same published ordering; this assumes exactly 3
clusters, so `n_clusters` isn't meant to be changed casually.

The CSF/other cluster is exposed as its own "CSF" mask (2026-09-29) rather
than discarded, since it was already being computed to disambiguate White
Matter/Gray Matter - giving it an explicit output class fixes a real
reported issue where subarachnoid-space/CSF pixels near the White Matter
cluster's boundary were occasionally read as White Matter with no way to
see or correct for it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from bictool.domain.segmentation._tissue_cleanup import clean_mask
from bictool.domain.segmentation.brain_tissue import BrainTissueResult

DEFAULT_N_CLUSTERS = 3
DEFAULT_FUZZINESS = 2.0
DEFAULT_MAX_ITER = 50
DEFAULT_SPATIAL_WEIGHT = 0.3
DEFAULT_MIN_COMPONENT_SIZE = 20


def _fit_fcm(values: np.ndarray, n_clusters: int, fuzziness: float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Fits standard FCM on 1D `values`. Returns `(membership, centers)`,
    `membership` shaped `(len(values), n_clusters)`.
    """
    centers = np.quantile(values, np.linspace(0.15, 0.85, n_clusters))
    power = 2.0 / (fuzziness - 1.0)
    membership = None
    for _ in range(max_iter):
        dist = np.abs(values[:, None] - centers[None, :]) + 1e-9
        inv = dist ** (-power)
        membership = inv / inv.sum(axis=1, keepdims=True)
        weights = membership ** fuzziness
        centers = (weights * values[:, None]).sum(axis=0) / weights.sum(axis=0)
    return membership, centers


def classify_brain_tissue_fcm(
    slice_hu: np.ndarray,
    brain_mask: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    fuzziness: float = DEFAULT_FUZZINESS,
    max_iter: int = DEFAULT_MAX_ITER,
    spatial_weight: float = DEFAULT_SPATIAL_WEIGHT,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> BrainTissueResult:
    mask_indices = np.flatnonzero(brain_mask)
    if mask_indices.size < n_clusters:
        empty = np.zeros_like(brain_mask, dtype=bool)
        return BrainTissueResult(masks={"CSF": empty.copy(), "White Matter": empty.copy(), "Gray Matter": empty.copy()})

    values = slice_hu.ravel()[mask_indices].astype(np.float64)
    membership, centers = _fit_fcm(values, n_clusters, fuzziness, max_iter)

    # Spatial regularization: blend each pixel's own membership with its
    # local 3x3 neighborhood's average membership (per class) before taking
    # the final hard label - not from raw HU value/membership alone.
    membership_maps = np.zeros(brain_mask.shape + (n_clusters,), dtype=np.float64)
    for k in range(n_clusters):
        flat = np.zeros(brain_mask.size, dtype=np.float64)
        flat[mask_indices] = membership[:, k]
        membership_maps[..., k] = flat.reshape(brain_mask.shape)

    neighbor_avg = np.stack(
        [ndimage.uniform_filter(membership_maps[..., k], size=3) for k in range(n_clusters)], axis=-1
    )
    blended = (1 - spatial_weight) * membership_maps + spatial_weight * neighbor_avg

    order = np.argsort(centers)  # ascending HU: [CSF/other, White Matter, Gray Matter]
    csf_index, white_index, gray_index = order[0], order[1], order[2]

    labels = np.argmax(blended, axis=-1)
    csf_raw = brain_mask & (labels == csf_index)
    white_matter_raw = brain_mask & (labels == white_index)
    gray_matter_raw = brain_mask & (labels == gray_index)

    csf = clean_mask(csf_raw, min_component_size)
    white_matter = clean_mask(white_matter_raw, min_component_size)
    gray_matter = clean_mask(gray_matter_raw, min_component_size)
    return BrainTissueResult(masks={"CSF": csf, "White Matter": white_matter, "Gray Matter": gray_matter})
