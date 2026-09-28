"""Shared post-processing helper for the alternative WM/GM classification
methods (`brain_tissue_fcm.py`/`brain_tissue_gmm_mrf.py`/
`brain_tissue_adaptive.py`) - morphological open+close plus small-component
removal, the same lightweight consolidation step `brain_tissue.py`'s own
`_clean_mask` uses. Duplicated here rather than imported from
`brain_tissue.py`, deliberately: these are experimental alternatives being
compared side by side (2026-09-24 request), not extensions of
`brain_tissue.py` itself, so each stays self-contained and independently
comparable rather than coupled to that module's own private internals.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_OPENING_STRUCTURE = np.ones((3, 3), dtype=bool)
_CLOSING_STRUCTURE = np.ones((5, 5), dtype=bool)


def clean_mask(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    opened = ndimage.binary_opening(mask, structure=_OPENING_STRUCTURE)
    closed = ndimage.binary_closing(opened, structure=_CLOSING_STRUCTURE)

    labeled, num_labels = ndimage.label(closed)
    if num_labels == 0:
        return closed
    component_sizes = ndimage.sum(closed, labeled, index=range(1, num_labels + 1))
    keep_labels = {i + 1 for i, size in enumerate(component_sizes) if size >= min_component_size}
    return np.isin(labeled, list(keep_labels))
