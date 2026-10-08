"""Binary (background-vs-lesion) acute ischemic stroke segmentation.

**This is a diagnostic-classification output, not a QC/image-quality
measurement** - the one place in this integration where that distinction
needs to be stated in code, not just UI copy. Every other module under
`domain/` measures or characterizes image quality (MTF/NPS/PS, contrast,
anatomy for ROI placement); this one estimates a clinical finding. It
exists here because it's built on the same pretrained encoder BICTool
already uses for anatomy segmentation, not because stroke detection is a
stated requirement of this tool (see `main_window.py`'s "Stroke Detection"
box, which is deliberately separated from the anatomical Segmentation
controls and labeled investigational).

Base: `jasonccai/HeadCTSegmentation`'s pretrained encoder (see
`head_ct_unet.py`), transfer-learned to this task by
`Ashly1991/AISD-HeadCT-Transfer-Learning` via 3-stage progressive
unfreezing (Stage 1: decoder only; Stage 2: + deepest encoder block,
reported best - checkpoint `best_stage2.weights.h5`; Stage 3: + two
encoder blocks). Architecture is `_unet_architecture.build_unet` (shared
with `head_ct_unet.py`), `nb_classes=2`.

Preprocessing: identical to `head_ct_unet.py` - confirmed raw HU, no
windowing/normalization (verified directly against the transfer-learning
repo's own `data_preparation/convert_trainval_dicom_to_hu_nifti.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bictool.domain.segmentation._resize import restore_from_512, resize_to_512
from bictool.domain.segmentation._unet_architecture import build_unet

NB_CLASSES = 2  # background, lesion
_LESION_CLASS_INDEX = 1


@dataclass
class StrokeSegmentationResult:
    """A single 2D boolean lesion mask, cropped back to the original
    slice's shape. Named distinctly from `head_ct_unet.SegmentationResult`
    (a single mask, not a per-structure dict) to keep this module's output
    honest about being a different kind of thing - one binary finding, not
    an anatomical atlas.
    """

    masks: dict[str, np.ndarray]  # {"Lesion": mask} - dict-shaped (not a bare array) so it
    # can be passed through the same `decode_masks_for_checkboxes` as the anatomy result.


def build_model():
    """Builds the (untrained) 2-class stroke `keras.Model` - load its weights
    (`integration.model_files.load_stroke_weights`) before `segment_slice_for_stroke`."""
    return build_unet(NB_CLASSES)


def segment_slice_for_stroke(model, slice_hu: np.ndarray) -> StrokeSegmentationResult:
    """Segments one 2D HU slice (any size - see `resize_to_512`) for a
    suspected stroke lesion.
    """
    padded, context = resize_to_512(slice_hu)
    batch = padded[np.newaxis, :, :, np.newaxis].astype(np.float32)
    probabilities = model.predict(batch, verbose=0)[0]
    labels_512 = np.argmax(probabilities, axis=-1).reshape(512, 512)

    lesion_mask_512 = labels_512 == _LESION_CLASS_INDEX
    lesion_mask = restore_from_512(lesion_mask_512, context)
    return StrokeSegmentationResult(masks={"Lesion": lesion_mask})
