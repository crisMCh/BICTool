"""Shared pad/crop math for fitting a CT slice into the segmentation
models' fixed 512x512 input, and restoring their output masks back to the
slice's real shape.

Private/shared between `head_ct_unet.py` and `aisd_stroke_unet.py` - the
shape-handling math doesn't depend on which model runs, only on the fact
that both require exactly 512x512 input. Pure NumPy, no Qt/Presentation
dependency, matching the Domain Layer's "free of persistence and network
concerns" rule (`architecture/layers_description.md`) - this is model I/O
shaping, not display logic, so it belongs here rather than in a ViewModel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MODEL_SIZE = 512


@dataclass(frozen=True)
class _AxisFit:
    """How one axis was fit to 512: `offset` is the pad-before amount when
    `mode == "pad"`, or the crop-before amount when `mode == "crop"`.
    """

    mode: str  # "pad" or "crop"
    offset: int


@dataclass(frozen=True)
class ResizeContext:
    """Records how a slice was fitted into the model's fixed 512x512 input,
    so a 512x512 output mask can be restored to the slice's real shape.
    """

    original_shape: tuple[int, int]  # (height, width)
    row_fit: _AxisFit
    col_fit: _AxisFit


def _fit_axis(size: int) -> _AxisFit:
    if size <= _MODEL_SIZE:
        return _AxisFit(mode="pad", offset=(_MODEL_SIZE - size) // 2)
    return _AxisFit(mode="crop", offset=(size - _MODEL_SIZE) // 2)


def resize_to_512(slice_hu: np.ndarray) -> tuple[np.ndarray, ResizeContext]:
    """Fits `slice_hu` (2D, any size) into a 512x512 array for the model.

    Each axis is fit independently by padding (if smaller than 512) or
    center-cropping (if larger than 512) - never by an interpolated resize.
    Real head-CT image matrices are essentially always within a few dozen
    pixels of 512 (e.g. 512x434, or 522x436 on some scanners' larger
    reconstruction FOVs), so a per-axis crop/pad reaches exactly 512x512
    while preserving each pixel's true physical scale.

    This matters because the model's own training data was never resized -
    the base repo's training pipeline hard-rejects any slice that isn't
    already exactly 512x512 - so a full-image interpolated resize (the
    previous approach here for any slice with one dimension over 512) would
    apply *different* scale factors to each axis whenever the slice's aspect
    ratio isn't already 1:1, non-uniformly stretching the anatomy before the
    model ever sees it. That distortion was confirmed (2026-09-16) to
    produce visibly fragmented, anatomically incoherent segmentation output
    on a real 522x436 scan, even though the same weights reproduced a
    512x512 reference volume's known-good prediction pixel-for-pixel - see
    this project's segmentation-integration memory. A center-crop of the
    oversized axis only discards a thin, symmetric border strip (here, 5
    rows off the top and bottom) - almost always table/air at the image's
    extreme edge, not brain anatomy - which is a far smaller and more
    predictable cost than distorting the whole image's proportions.

    Padding uses `slice_hu`'s own minimum value (air, for a real CT slice)
    rather than 0 - 0 HU is inside the soft-tissue range and would bias the
    model into "seeing" tissue in the padding. Both pad and crop are exactly
    invertible with zero interpolation artifacts (see `restore_from_512`).
    """
    height, width = slice_hu.shape
    row_fit = _fit_axis(height)
    col_fit = _fit_axis(width)

    canvas = np.full((_MODEL_SIZE, _MODEL_SIZE), slice_hu.min(), dtype=slice_hu.dtype)

    if row_fit.mode == "crop":
        src_rows = slice(row_fit.offset, row_fit.offset + _MODEL_SIZE)
        dst_rows = slice(0, _MODEL_SIZE)
    else:
        src_rows = slice(0, height)
        dst_rows = slice(row_fit.offset, row_fit.offset + height)

    if col_fit.mode == "crop":
        src_cols = slice(col_fit.offset, col_fit.offset + _MODEL_SIZE)
        dst_cols = slice(0, _MODEL_SIZE)
    else:
        src_cols = slice(0, width)
        dst_cols = slice(col_fit.offset, col_fit.offset + width)

    canvas[dst_rows, dst_cols] = slice_hu[src_rows, src_cols]
    return canvas, ResizeContext(original_shape=(height, width), row_fit=row_fit, col_fit=col_fit)


def restore_from_512(mask_512: np.ndarray, context: ResizeContext) -> np.ndarray:
    """Inverse of `resize_to_512`: restores a 512x512 output mask to
    `context.original_shape`, per axis - cropping back an axis that was
    padded, or padding back (with `False`) an axis that was cropped, since
    a cropped-away border has no model output to restore.
    """
    height, width = context.original_shape
    result = np.zeros((height, width), dtype=mask_512.dtype)

    row_fit, col_fit = context.row_fit, context.col_fit
    if row_fit.mode == "crop":
        dst_rows = slice(row_fit.offset, row_fit.offset + _MODEL_SIZE)
        src_rows = slice(0, _MODEL_SIZE)
    else:
        dst_rows = slice(0, height)
        src_rows = slice(row_fit.offset, row_fit.offset + height)

    if col_fit.mode == "crop":
        dst_cols = slice(col_fit.offset, col_fit.offset + _MODEL_SIZE)
        src_cols = slice(0, _MODEL_SIZE)
    else:
        dst_cols = slice(0, width)
        src_cols = slice(col_fit.offset, col_fit.offset + width)

    result[dst_rows, dst_cols] = mask_512[src_rows, src_cols]
    return result
