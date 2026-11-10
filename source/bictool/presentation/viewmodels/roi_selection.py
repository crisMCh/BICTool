"""Pure coordinate/cropping math for manual ROI selection.

Kept separate from the View so it is testable without Qt: the View only
needs to turn a mouse drag into an (x0, y0, x1, y1) image-pixel rectangle
and pass it to `crop_roi`.
"""

from __future__ import annotations

from bictool.application.roi_store import RoiRect, candidate_to_rect, crop_roi  # noqa: F401 (re-exported)


def normalize_rect(xa: int, ya: int, xb: int, yb: int, image_width: int, image_height: int) -> RoiRect:
    """Builds a `RoiRect` from two arbitrary corner points, ordering and clamping them to the image bounds."""
    x0, x1 = sorted((xa, xb))
    y0, y1 = sorted((ya, yb))
    x0 = max(0, min(x0, image_width))
    x1 = max(0, min(x1, image_width))
    y0 = max(0, min(y0, image_height))
    y1 = max(0, min(y1, image_height))
    return RoiRect(x0, y0, x1, y1)


def make_square(rect: RoiRect, image_width: int, image_height: int) -> RoiRect:
    """Adjusts `rect` to a square, centered on the same point and sized to
    its larger dimension, then shifted (never resized further) to fit
    within the image bounds so squareness is preserved even near an edge.

    `domain.metrics.nps.calc_digital_nps` requires an exactly square ROI
    (`"ROI must be symmetric."`) - a manually dragged rectangle essentially
    never lands on a pixel-perfect square, so this runs on every manual
    selection rather than asking the user to somehow drag one by hand.
    """
    size = max(rect.width, rect.height)
    size = min(size, image_width, image_height)
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    x0 = round(cx - size / 2)
    y0 = round(cy - size / 2)
    x0 = max(0, min(x0, image_width - size))
    y0 = max(0, min(y0, image_height - size))
    return RoiRect(x0, y0, x0 + size, y0 + size)
