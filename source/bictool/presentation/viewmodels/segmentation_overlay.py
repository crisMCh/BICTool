"""Maps a segmentation model's own raw class names onto whatever display
keys the sidebar's checkboxes actually use.

Kept separate from `domain.segmentation` (which stays a faithful wrapper
around each third-party model's own output) since this is BICTool-specific
curation - adapting Model data for display, the ViewModel's job per this
project's MVVM split (`architecture/layers_description.md`).
"""

from __future__ import annotations

import numpy as np


def decode_masks_for_checkboxes(result, mapping: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Turns a segmentation result's raw model-class masks into whichever
    display-name keys `main_window.py`'s checkboxes use, OR-ing together
    every raw class listed for a given display name (e.g. WP2's "Basal
    ganglia" -> `Caudate Nucleus` OR `Lentiform Nucleus`).

    `result` is any object with a `.masks: dict[str, np.ndarray]` attribute
    (both `head_ct_unet.SegmentationResult` and a stroke result with a
    single-entry `masks` dict work) - not type-pinned to one dataclass so
    this stays reusable.
    """
    decoded: dict[str, np.ndarray] = {}
    for display_name, raw_names in mapping.items():
        combined = None
        for raw_name in raw_names:
            mask = result.masks[raw_name]
            combined = mask if combined is None else combined | mask
        decoded[display_name] = combined
    return decoded
