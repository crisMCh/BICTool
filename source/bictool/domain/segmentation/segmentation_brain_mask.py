"""Derives a skull-strip "Brain Mask" from `head_ct_unet`'s CNN anatomy
segmentation, as a more robust alternative to `brain_tissue.extract_brain_mask`'s
HU-threshold approach on irregular anatomy.

Background (2026-09-28/29, real postop DICOM data under `E:\\Datasets`, a
craniotomy/hardware case): `extract_brain_mask`'s HU-threshold pipeline was
found unfixable for this class of scan by three different attempts (a
size-fraction union of components, a spatially-local gap-closing step, and a
requirement that the mask be mostly bone-enclosed) - real postop anatomy can
have a substantial air/fluid gap between brain and skull (confirmed on this
scan: 10-20+ pixel gap, pure air HU) even on otherwise-correct slices, so no
purely HU/connectivity-based rule reliably distinguishes true brain from the
connected scalp/orbit/sinus soft-tissue envelope. The CNN segmentation model
was trained on real anatomy and does not rely on either assumption.

This module instead builds the brain mask from the *union* of every
structure `head_ct_unet.segment_slice` predicts (all 16 named classes,
i.e. everything that isn't background) - "the merged labels," per the
model's own understanding of anatomy rather than an HU/connectivity rule.

On an irregular scan, the model can still occasionally mislabel a small,
spatially-implausible patch elsewhere in the image (e.g. a face/scalp
region) as one of its 16 classes. `largest_connected_volume` guards against
this the same way `extract_brain_mask` always has - keep only the single
largest connected region - but does so in 3D across the whole segmented
volume (not per 2D slice), since a real brain is one contiguous 3D object
even where an individual axial slice's own cross-section looks fragmented
(e.g. a craniotomy/hardware artifact that happens to bisect one slice is
almost always still connected through the slices above/below it in 3D).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# 3D connectivity for the volume-wide largest-component check: only
# face-connected (not edge/corner) neighbors count as touching - the
# conservative default, consistent with `ndimage.label`'s own 2D default
# used throughout the rest of this package.
_VOLUME_CONNECTIVITY = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ],
    dtype=bool,
)


def merge_segmentation_labels(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Unions every structure mask in one slice's `SegmentationResult.masks`
    into a single boolean "this pixel is some brain structure" mask.
    Returns an all-`False` array of the caller-inferred shape if `masks` is
    empty (nothing to union) - callers must supply the shape separately in
    that case, so this only handles the non-empty case.
    """
    merged: np.ndarray | None = None
    for mask in masks.values():
        merged = mask.copy() if merged is None else (merged | mask)
    assert merged is not None, "merge_segmentation_labels requires at least one mask"
    return merged


def largest_connected_volume(volume_mask: np.ndarray) -> np.ndarray:
    """Keeps only the single largest 3D-connected component of a
    `(num_slices, height, width)` boolean volume, discarding any other
    component regardless of which 2D slice(s) it appears on - a spurious
    same-slice-sized 2D fragment that isn't actually part of the same real
    3D object gets dropped, while a real object that looks fragmented on any
    single 2D slice survives as long as it's connected through adjacent
    slices.
    """
    labeled, num_labels = ndimage.label(volume_mask, structure=_VOLUME_CONNECTIVITY)
    if num_labels == 0:
        return np.zeros_like(volume_mask, dtype=bool)
    component_sizes = ndimage.sum(volume_mask, labeled, index=range(1, num_labels + 1))
    largest_label = int(np.argmax(component_sizes)) + 1
    return labeled == largest_label


def brain_mask_from_segmentation(
    masks_by_slice: dict[int, dict[str, np.ndarray]],
    total_slices: int,
) -> dict[int, np.ndarray]:
    """Builds a per-slice brain mask from whatever CNN segmentation results
    are currently available, via `merge_segmentation_labels` +
    `largest_connected_volume`.

    `masks_by_slice` need not cover every slice (segmentation may still be
    in progress, or only a few slices may ever get segmented) - a slice with
    no entry is treated as an all-`False` layer in the 3D volume, at its own
    real slice index (`total_slices` gives the true volume depth), so 3D
    adjacency is computed against genuine geometric neighbors, never against
    a different real slice compacted into the wrong Z position. Only slices
    present in `masks_by_slice` are included in the returned dict - this
    function never fabricates a result for a slice with no segmentation
    data at all.
    """
    if not masks_by_slice:
        return {}

    shape = next(iter(next(iter(masks_by_slice.values())).values())).shape
    volume = np.zeros((total_slices, *shape), dtype=bool)
    for slice_index, masks in masks_by_slice.items():
        volume[slice_index] = merge_segmentation_labels(masks)

    filtered = largest_connected_volume(volume)
    return {slice_index: filtered[slice_index] for slice_index in masks_by_slice}
