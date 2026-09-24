"""Pure clinical windowing (WW/WL) math for the DICOM viewer.

Kept separate from the View so it is testable without Qt, matching
`roi_selection.py`'s split: the View only needs to call `apply_window` and
wrap the result in a `QImage`.
"""

from __future__ import annotations

import numpy as np

# Standard clinical CT head-reading windows. Value is (width, level) in HU;
# each preset uses the midpoint of its documented range, same convention as
# the original Brain/Bone pair. Sources (kept here for later documentation -
# see the user-facing request that added Subdural/Stroke/Soft tissue):
#
# - Brain, Bone: `requirements/BICTool_WP2_FinalDraft.pdf` section 3
#   ("Relevant structures in head CT images") - Brain/soft-tissue also
#   covers CSF (WP2 gives them the same window); Bone also covers vessels.
# - Subdural: Radiopaedia, "CT head (subdural window)"
#   (https://radiopaedia.org/articles/ct-head-subdural-window-1) - widens
#   and raises the window relative to Brain so a thin subdural haematoma
#   against the calvarium (similar density to bone on a Brain window) is no
#   longer masked by the adjacent skull.
# - Stroke: Healthcare Online Tools, "CT Window Presets"
#   (https://healthcareonlinetools.com/en/blog/dicom-tools/ct-window-presets-brain-lung-bone-soft-tissue/),
#   corroborated by Medmastery's brain-CT display guide
#   (https://www.medmastery.com/guides/brain-ct-clinical-guide/how-optimize-display-brain-computed-tomography-ct-images) -
#   a narrower window than Brain, used in suspected acute ischaemic stroke to
#   exaggerate the subtle grey/white attenuation drop of early infarction.
# - Soft tissue: Healthcare Online Tools (same article as Stroke) - for
#   extracranial soft tissue (scalp swelling, orbits, sinuses) visible on a
#   head CT but under-differentiated by the narrow Brain window.
WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    "Brain": (90.0, 35.0),  # WP2: soft tissue/brain/CSF - WW 80-100 HU, WL 30-40 HU
    "Subdural": (200.0, 75.0),  # Radiopaedia: WW 130-300 HU, WL 50-100 HU
    "Bone": (3000.0, 400.0),  # WP2: bone/vessels - WW 2000-4000 HU, WL 300-500 HU
    "Stroke": (35.0, 35.0),  # WW 30-40 HU, WL 30-35 HU
    "Soft tissue": (375.0, 45.0),  # WW 350-400 HU, WL 40-50 HU
}
DEFAULT_WINDOW_PRESET = "Brain"


def apply_window(pixels_hu: np.ndarray, width: float, center: float) -> np.ndarray:
    """Maps HU pixel data to 8-bit grayscale via a DICOM-style window
    (width/center): values are clipped to [center - width/2, center +
    width/2], then linearly scaled to [0, 255]. A non-positive width (or
    zero-span window) returns an all-black image rather than dividing by
    zero.
    """
    low = center - width / 2
    high = center + width / 2
    span = high - low
    if span <= 0:
        return np.zeros_like(pixels_hu, dtype=np.uint8)
    normalized = (np.clip(pixels_hu, low, high) - low) / span * 255.0
    return normalized.astype(np.uint8)
