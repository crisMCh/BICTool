"""What one volume contributes to a dataset summary (`application.batch_qa`,
the Excel/CSV exports): the automatic ROIs and per-slice QA per ROI1
structure, the study verdict and hints. Filled from an `api.StudyResult`
(`StudyResult.to_volume_qa`). Plain dataclasses - no Qt, no file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bictool.domain.results.study_quality import SliceQualityResult
from bictool.domain.roi_selection.structured_roi import RoiCandidate


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
    # Basal-ganglia visibility per slice (`domain.metrics.basal_ganglia`, provisional).
    basal_ganglia: dict | None = None
    # CNN confidence per slice (`segmentation_confidence.SegmentationConfidence`); {} for old caches.
    segmentation_confidence: dict = field(default_factory=dict)
    # Study-level segmentation check (`segmentation_plausibility`, provisional).
    segmentation_plausibility: object | None = None
    # Plug-in values as export rows (`results_export.plugin_rows`); empty without plug-ins.
    plugin_rows: list = field(default_factory=list)
    error: str | None = None
