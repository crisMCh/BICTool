"""The anatomy CNN's confidence on one slice, from its softmax output (TensorFlow-free, so the cache can use it
without importing the model). Recorded with every segmentation since 2026-10-20; not used by a check yet - it is
scanner-dependent and the CNN is often confidently wrong (docs/qa_method/segmentation_plausibility.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pixels whose most likely class has a probability below this count as "unsure" (`SegmentationConfidence`).
LOW_CONFIDENCE_PROBABILITY = 0.5


@dataclass(frozen=True)
class SegmentationConfidence:
    """How sure the CNN was on one slice, from its softmax output over the pixels it labelled as some
    structure (not background): `mean_entropy` (0 = certain, 1 = all 17 classes equally likely) and
    `low_fraction` (share of those pixels whose most likely class has p < `LOW_CONFIDENCE_PROBABILITY`).
    Not a quality measure of the image."""

    mean_entropy: float
    low_fraction: float
    labelled_pixels: int


def confidence_from_probabilities(probabilities: np.ndarray) -> SegmentationConfidence:
    """`probabilities`: (..., n_classes) softmax output, class 0 = background."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1, probabilities.shape[-1])
    labelled = np.argmax(p, axis=1) != 0
    if not labelled.any():
        return SegmentationConfidence(0.0, 0.0, 0)
    q = p[labelled]
    entropy = -(q * np.log(np.clip(q, 1e-12, 1.0))).sum(axis=1) / np.log(p.shape[1])
    return SegmentationConfidence(
        mean_entropy=float(entropy.mean()),
        low_fraction=float((q.max(axis=1) < LOW_CONFIDENCE_PROBABILITY).mean()),
        labelled_pixels=int(labelled.sum()),
    )
