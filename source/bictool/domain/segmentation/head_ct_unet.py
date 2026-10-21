"""17-class head-CT anatomy segmentation (FR-02), wrapping a pretrained
model rather than implementing a new one.

Base model: `jasonccai/HeadCTSegmentation` (Mayo Clinic Radiology
Informatics Lab; Cai et al., "Fully Automated Segmentation of Head CT
Neuroanatomy Using Deep Learning," Radiology: Artificial Intelligence,
2020, DOI 10.1148/ryai.2020190183). Architecture is `_unet_architecture
.build_unet` (shared with `aisd_stroke_unet.py`); this module supplies the
class count/names and the model-specific pre/post-processing.

Preprocessing - confirmed, not assumed: the student's own DICOM-to-HU
conversion script (`Ashly1991/AISD-HeadCT-Transfer-Learning`'s
`data_preparation/convert_trainval_dicom_to_hu_nifti.py`, which itself
feeds this same base model's pretrained encoder) applies only the standard
`pixel_array * RescaleSlope + RescaleIntercept` HU conversion - no
windowing, clipping, or normalization. `segment_slice` therefore takes
`DicomVolume.get_slice_hu()`'s output directly, with no rescaling beyond
the 512x512 shape handling in `domain.segmentation._resize`.

The 16-structure label map (index 0 is background, not a named structure)
combines a user-supplied initial guess with the base repo's own training
code - `jasonccai/HeadCTSegmentation`'s `z_controlboard.py` installs its
per-channel Dice metrics as `install_channel_dicemetric(1,"Brain")`,
`(2,"CSF")`, `(3,"DuraNSinus")`, `(4,"SeptumPellucidum")`,
`(5,"Cerebellum")`, `(6,"Caudate")`, `(7,"Lentiform")`, `(8,"Insular")`,
`(9,"InternalCapsule")`, `(10,"Ventricle")`, `(11,"CentralSulcus")` - this
is the actual channel order the model was trained against for indices
1-11 (the paper's Table 3 only reports Dice scores for these same 11; the
full 16 is a later, undocumented expansion of the same pipeline, so
indices 12-16 have no equivalent source-code confirmation). Cross-checked
against `reference/segmentation/Prediction_2CP_Bad2_2_head_40_axial.nii.gz`
(and its matching `Image_...` HU volume) 2026-09-16: per-label centroid,
bilaterality (caudate/lentiform/insular/internal-capsule/temporal-lobe all
show as two near-equal, mirrored connected components; septum
pellucidum/dural-folds/thalamus sit near midline as expected) and relative
position (frontal > parietal > occipital front-to-back; central sulcus at
the vertex, well above the basal-ganglia level; cerebellum and temporal
lobe both inferior; ventricle the single lowest-HU class) all matched
`_CLASS_NAMES` below - **except indices 8 and 11, which were swapped in an
earlier version of this map** (Central Sulcus and Insular Cortex have very
different anatomical positions - Insular is lateral, at the basal-ganglia
level; Central Sulcus is at the vertex - so this was a real bug, not a
naming nuance). Fixed to match `z_controlboard.py`'s order. Indices 12-16
(the four lobes + thalamus) are unconfirmed by source code but matched the
reference volume's anatomy cleanly (frontal=anterior extreme, occipital=
posterior extreme, parietal=between them and superior, temporal=lateral+
inferior+bilateral, thalamus=near-midline at the basal-ganglia z-level with
nucleus-like HU) - worth re-checking if a future reference case disagrees.
Worth a one-time sanity check once real weights are loaded: a wrong
`nb_classes` here would fail `load_weights` immediately (shape mismatch),
not silently - see this module's own test for the shape-only check that
doesn't need real weights at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bictool.domain.segmentation._resize import restore_from_512, resize_to_512
from bictool.domain.segmentation._unet_architecture import build_unet
from bictool.domain.segmentation.segmentation_confidence import (  # noqa: F401 - re-exported
    SegmentationConfidence,
    confidence_from_probabilities,
)

NB_CLASSES = 17  # background (0) + 16 named structures

_CLASS_NAMES = {
    1: "Brain Parenchyma",
    2: "Subarachnoid Space",
    3: "Dural Folds",
    4: "Septum Pellucidum",
    5: "Cerebellum",
    6: "Caudate Nucleus",
    7: "Lentiform Nucleus",
    8: "Insular Cortex",
    9: "Internal Capsule",
    10: "Ventricular System",
    11: "Central Sulcus",
    12: "Frontal Lobe",
    13: "Parietal Lobe",
    14: "Occipital Lobe",
    15: "Temporal Lobe",
    16: "Thalamus",
}


@dataclass
class SegmentationResult:
    """One 2D boolean mask per anatomical structure, already cropped back
    to the original slice's shape - keyed by the model's own class names
    (WP2 renaming/consolidation happens in `viewmodels.segmentation_overlay
    .decode_masks_for_checkboxes`, not here, so this stays a faithful
    wrapper around the third-party model's own output).
    """

    masks: dict[str, np.ndarray]
    confidence: SegmentationConfidence | None = None   # None for results from before 2026-10-20 (e.g. old caches)


def build_model():
    """Builds the (untrained) 17-class anatomy `keras.Model` - load its
    weights (`integration.model_files.load_anatomy_weights`) before `segment_slice`."""
    return build_unet(NB_CLASSES)


def segment_slice(model, slice_hu: np.ndarray) -> SegmentationResult:
    """Segments one 2D HU slice (any size - see `resize_to_512`) into one
    boolean mask per anatomical structure, each shaped like `slice_hu`.
    """
    padded, context = resize_to_512(slice_hu)
    batch = padded[np.newaxis, :, :, np.newaxis].astype(np.float32)
    probabilities = model.predict(batch, verbose=0)[0]
    labels_512 = np.argmax(probabilities, axis=-1).reshape(512, 512)
    confidence = confidence_from_probabilities(probabilities)

    masks: dict[str, np.ndarray] = {}
    for class_index, name in _CLASS_NAMES.items():
        mask_512 = labels_512 == class_index
        masks[name] = restore_from_512(mask_512, context)
    return SegmentationResult(masks=masks, confidence=confidence)
