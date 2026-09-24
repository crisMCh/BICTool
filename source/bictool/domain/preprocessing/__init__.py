"""Preprocessing sub-module of the Domain Layer.

Image-level preparation prior to segmentation/metric computation
(e.g. denoising, normalization). `_pixel_ops` holds internal,
Octave-Forge-equivalent pixel-level numeric primitives used by the legacy
pipeline translation (see `bictool.domain.metrics.mtf`) - not a public API.
"""
