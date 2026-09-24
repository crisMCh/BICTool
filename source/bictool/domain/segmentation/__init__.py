"""Segmentation sub-module of the Domain Layer.

Detects/segments the anatomical structures relevant to head-CT image
quality (ventricles, septum pellucidum, falx cerebri, tentorium
cerebelli, basal ganglia, white/gray matter, brainstem - per WP2 §3).

`head_ct_unet` wires in a pretrained 17-class CNN (`jasonccai/
HeadCTSegmentation`) as the AI-based segmentation method WP2 anticipated -
see its module docstring for the WP2-structure-to-model-class mapping and
the Phase 2 curation step (`presentation.viewmodels.segmentation_overlay
.decode_masks_for_checkboxes`) that renames/consolidates its raw output
into WP2's own structure names. `aisd_stroke_unet` reuses the same
pretrained encoder for a different, non-QC purpose (stroke-lesion
detection) - see its own module docstring for why that's kept separate.
`_unet_architecture`/`_resize` are private helpers shared by both.

`active_contour` is unrelated to either: a faithful translation of the
legacy Octave Chan-Vese segmentation used by the MTF pipeline (see
`bictool.domain.metrics.mtf`) for single-edge curve-fitting, not
whole-image anatomical segmentation.
"""
