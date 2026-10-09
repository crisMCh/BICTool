"""Automatic ROI selection + per-slice QA for one already-segmented volume -
the same steps the GUI runs (Auto ROIs, then Run All), without the GUI, so a
whole dataset can be processed in one go (`application.batch_qa`).

Per ROI1 structure (septum pellucidum, dural folds), for every segmented
slice with enough brain (`slice_eligibility`): ROI1 on the structure (`structured_roi`), ROI2 next to it
(`homogeneous_roi`, white matter from `tissue_classify`), then
`evaluate_slice_quality` with MTF seeds on the structure (`mtf_seed`). A
slice whose QA raises is recorded as an error - it doesn't stop the volume.

Pure NumPy/SciPy - no Qt, no file I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from bictool.domain.metrics.mtf_seed import mtf_seed_candidates
from bictool.domain.results.study_quality import DEFAULT_RECIPE, SliceQualityResult, evaluate_slice_quality
from bictool.domain.roi_selection.homogeneous_roi import find_roi2_candidate
from bictool.domain.roi_selection.slice_eligibility import eligible_slices
from bictool.domain.roi_selection.structured_roi import STRUCTURES, RoiCandidate, find_roi1_candidates
from bictool.domain.segmentation.segmentation_brain_mask import brain_mask_from_segmentation

logger = logging.getLogger(__name__)

# (slice_hu, brain_mask) -> object with `.masks` ("White Matter"/"Gray Matter"/...)
TissueClassifier = Callable[[np.ndarray, np.ndarray], object]


@dataclass
class StructureQa:
    """One ROI1 structure's results over a volume, keyed by 0-based slice index."""

    structure: str
    rois: dict[int, tuple[RoiCandidate, RoiCandidate]] = field(default_factory=dict)
    results: dict[int, SliceQualityResult] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    # Slices where the structure was found but that have too little brain
    # for automatic ROIs (`slice_eligibility`) - not measured.
    skipped: list[int] = field(default_factory=list)


@dataclass
class VolumeQa:
    """Everything one volume contributes to a dataset summary. `group` is
    the dataset it belongs to (e.g. the scanner), `name` the study.
    `error` is set when the volume couldn't be processed at all.
    """

    group: str
    name: str
    series_label: str = ""
    slice_count: int = 0
    segmented_slices: int = 0
    pixel_spacing_mm: float = float("nan")
    acquisition_date: str | None = None
    structures: dict[str, StructureQa] = field(default_factory=dict)
    # Study-level head-CT verdict (physical recipe; `head_ct_verdict`), from
    # the dural-folds (falx) results - None for the legacy recipe or too few slices.
    head_ct_verdict: object | None = None
    # Possible-artefact hints (`domain.artefacts.artefact_flags.ArtefactFlags`).
    artefact_flags: object | None = None
    error: str | None = None


def evaluate_volume_qa(
    masks_by_slice: Mapping[int, dict[str, np.ndarray]],
    get_slice_hu: Callable[[int], np.ndarray],
    num_slices: int,
    pixel_spacing_mm: float,
    tissue_classify: TissueClassifier,
    structures: tuple[str, ...] = STRUCTURES,
    recipe: str = DEFAULT_RECIPE,
    *,
    svm_model,
) -> dict[str, StructureQa]:
    """Auto ROIs + QA for every segmented slice, separately per structure.

    `masks_by_slice` - the segmentation's masks per slice (DICOM pixel
    space), only for segmented slices. `svm_model` - the legacy per-slice
    classifier (`classification`), passed to `evaluate_slice_quality`. `tissue_classify` is only called for
    slices that get an ROI1, at most once per slice.
    """
    images = {index: get_slice_hu(index) for index in masks_by_slice}
    brain_masks = brain_mask_from_segmentation(dict(masks_by_slice), num_slices)
    tissue_cache: dict[int, dict[str, np.ndarray]] = {}

    def tissue_masks(index: int) -> dict[str, np.ndarray]:
        if index not in tissue_cache:
            brain = brain_masks.get(index)
            if brain is None:
                brain = np.zeros(images[index].shape, dtype=bool)
            tissue_cache[index] = tissue_classify(images[index], brain).masks
        return tissue_cache[index]

    eligible = eligible_slices(masks_by_slice)
    out: dict[str, StructureQa] = {}
    for structure in structures:
        qa = StructureQa(structure)
        # Searched on all slices (the midline direction is volume-wide), kept
        # only where there is enough brain.
        roi1s = find_roi1_candidates(dict(masks_by_slice), structure, pixel_spacing_mm, images_by_slice=images)
        qa.skipped = sorted(index for index in roi1s if index not in eligible)
        roi1s = {index: roi1 for index, roi1 in roi1s.items() if index in eligible}
        for index in sorted(roi1s):
            roi1 = roi1s[index]
            masks = masks_by_slice[index]
            roi2 = find_roi2_candidate(roi1, masks, pixel_spacing_mm, tissue_masks(index))
            if roi2 is None:
                qa.errors[index] = "No ROI2 found"
                continue
            qa.rois[index] = (roi1, roi2)
            image = images[index]
            img_roi1 = image[roi1.row0 : roi1.row1, roi1.col0 : roi1.col1]
            img_roi2 = image[roi2.row0 : roi2.row1, roi2.col0 : roi2.col1]
            hint = np.logical_or.reduce([masks[name] for name in STRUCTURES if name in masks])
            seeds = mtf_seed_candidates(img_roi1, hint[roi1.row0 : roi1.row1, roi1.col0 : roi1.col1])
            try:
                qa.results[index] = evaluate_slice_quality(
                    img_roi1, img_roi2, pixel_spacing_mm, svm_model=svm_model, mtf_seeds=seeds, recipe=recipe
                )
            except Exception as exc:  # one bad slice must not stop the volume
                logger.info("QA failed on slice %d (%s): %s", index + 1, structure, exc)
                qa.errors[index] = str(exc)
        out[structure] = qa
    return out
