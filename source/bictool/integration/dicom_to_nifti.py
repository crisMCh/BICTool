"""Converts a loaded DICOM series to a NIfTI volume for anatomy
segmentation (FR-02), via the third-party `dicom2nifti` package (the
library `E:\\Segmentation\\CTseg\\dicom2niftii.py` also uses, for an
unrelated segmentation method - only its DICOM/NIfTI plumbing is reused
here, not any of its own model/analysis code).

Purely an Integration-Layer adapter: reading DICOM files, deduplicating
slices, and invoking `dicom2nifti` - no image-analysis reasoning lives
here. The actual orientation-transform discovery and per-slice matching
between the DICOM and NIfTI representations is pure array logic with no
file I/O, so it lives in `domain.segmentation._nifti_alignment` instead
(see that module's docstring for the full rationale/evidence) - this
module's only job is to get real arrays loaded and hand them over there.

Two things are handled differently from the `dicom2niftii.py` reference
script, deliberately:

1. `dicom2niftii.py` calls `dicom2nifti.convert_directory`, which re-scans
   and re-groups an entire folder from scratch - duplicating work this
   project's own `discover_series` already does more carefully (splitting
   mismatched geometry, skipping burn-to-CD clutter by filename first, a
   study-count safety check). This module instead converts an
   already-discovered `DicomVolume`'s own sorted `slice_paths` directly, via
   the same underlying per-series conversion `convert_directory` itself
   calls internally.
2. `dicom2nifti` silently drops slices it doesn't consider part of the
   coherent stack, and does not expose which original slice index each
   NIfTI slice came from. This module detects duplicates itself first
   (comparing `ImagePositionPatient`), rather than depending on that
   undocumented internal behavior, so the original-DICOM-index -> NIfTI-
   index correspondence (built by `_nifti_alignment.align_dicom_to_nifti`)
   is always explicit and known, not inferred after the fact.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

from bictool.domain.segmentation._nifti_alignment import DicomNiftiVolume, align_dicom_to_nifti
from bictool.integration.dicom_file_loader import _parse_decimal

logger = logging.getLogger(__name__)

# Which DICOM slice's own array is used to discover the orientation
# transform (see `_nifti_alignment`'s docstring) - any slice with real
# anatomy works equally well; index 0 is simplest and always present.
_REFERENCE_SLICE_INDEX = 0


def _read_position(path: Path) -> tuple[float, float, float] | None:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    position = getattr(ds, "ImagePositionPatient", None)
    # `_parse_decimal` (not plain `float()`) - tolerates a comma decimal
    # separator, a real non-conformant-DICOM issue confirmed on a Siemens
    # NAEOTOM Alpha export (see its own docstring in dicom_file_loader.py).
    return tuple(_parse_decimal(v) for v in position) if position is not None else None


def _deduplicate(slice_paths: list[Path]) -> tuple[list[Path], dict[int, int]]:
    """Groups slices sharing the same `ImagePositionPatient` (exact
    duplicates - a real, observed burn-to-CD export artifact) and returns
    the deduplicated file list (one path per group, in original order) plus
    a map from every original index to its group's position in that
    deduplicated list.
    """
    unique_paths: list[Path] = []
    position_to_unique_index: dict[tuple[float, float, float] | None, int] = {}
    dicom_index_to_unique_index: dict[int, int] = {}

    for index, path in enumerate(slice_paths):
        position = _read_position(path)
        if position not in position_to_unique_index:
            position_to_unique_index[position] = len(unique_paths)
            unique_paths.append(path)
        dicom_index_to_unique_index[index] = position_to_unique_index[position]

    removed = len(slice_paths) - len(unique_paths)
    if removed:
        logger.info("dicom_to_nifti: skipped %d duplicate slice(s) (same ImagePositionPatient)", removed)
    return unique_paths, dicom_index_to_unique_index


def _convert_via_dicom2nifti(dicom_paths: list[Path]) -> np.ndarray:
    """Converts the given (already deduplicated, already sorted) DICOM
    files to a NIfTI volume via `dicom2nifti`, returning the HU array.

    Copies the files into an isolated temp directory rather than pointing
    `dicom2nifti` at their real (possibly mixed-content, burn-to-CD)
    parent folder - `dicom2nifti.convert_directory` re-scans and re-groups
    whatever directory it's given, and this project's own `discover_series`
    has already done that grouping more carefully (see module docstring);
    an isolated copy of just this series' files guarantees dicom2nifti
    converts exactly this series, nothing else found alongside it.
    """
    import dicom2nifti
    from dicom2nifti import settings as d2n_settings

    # Real head-CT acquisitions are routinely gantry-tilted (confirmed on
    # the real scan this module was validated against); dicom2nifti's
    # default validation rejects that outright. Disabling it does not
    # disable HU conversion or reorientation - it only stops dicom2nifti
    # from refusing to convert an oblique/irregularly-spaced series,
    # matching `E:\Segmentation\CTseg\dicom2niftii.py`'s own reasoning.
    d2n_settings.disable_validate_orthogonal()
    d2n_settings.disable_validate_slice_increment()

    with tempfile.TemporaryDirectory(prefix="bictool_dicom2nifti_") as tmpdir:
        tmp_dicom_dir = Path(tmpdir) / "dicom"
        tmp_dicom_dir.mkdir()
        for path in dicom_paths:
            shutil.copy2(path, tmp_dicom_dir / path.name)

        out_path = Path(tmpdir) / "converted.nii.gz"
        dicom2nifti.dicom_series_to_nifti(str(tmp_dicom_dir), str(out_path), reorient_nifti=True)
        return nib.load(str(out_path)).get_fdata().astype(np.float32)


def build_dicom_nifti_volume(slice_paths: list[Path], get_slice_hu) -> DicomNiftiVolume:
    """Converts the DICOM series given by `slice_paths` (e.g.
    `DicomVolume.slice_paths`) to a NIfTI volume, ready for segmentation.

    `get_slice_hu` is a callable `(dicom_index: int) -> np.ndarray` (e.g.
    `DicomVolume.get_slice_hu`) - reused rather than decoding pixel data a
    second time independently.

    This function's own job stops at "read the files, run dicom2nifti, hand
    real arrays to the Domain Layer" - `_nifti_alignment.align_dicom_to_
    nifti` does the actual orientation-transform discovery and per-slice
    matching (see that module's docstring); this just expands its result
    (computed over deduplicated slices) back out to cover every original
    DICOM index, including duplicates.
    """
    unique_paths, dicom_index_to_unique_index = _deduplicate(slice_paths)
    nifti_hu = _convert_via_dicom2nifti(unique_paths)

    # First DICOM index seen for each unique (deduplicated) slice - used
    # both to fetch that slice's own HU array and, after alignment, to
    # re-key the result back onto every original index via `rekeyed`.
    unique_index_to_dicom_index: dict[int, int] = {}
    for dicom_index, unique_index in dicom_index_to_unique_index.items():
        unique_index_to_dicom_index.setdefault(unique_index, dicom_index)

    dicom_slices = {
        unique_index: get_slice_hu(dicom_index) for unique_index, dicom_index in unique_index_to_dicom_index.items()
    }
    reference_unique_index = dicom_index_to_unique_index[_REFERENCE_SLICE_INDEX]

    alignment = align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=reference_unique_index)
    return alignment.rekeyed(dicom_index_to_unique_index)
