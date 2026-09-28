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
        # `reorient_nifti=False` - deliberately NOT letting dicom2nifti
        # canonicalize the array axes to world-axis order. Confirmed as a
        # real bug on a real gantry-tilted scan (2026-09-25,
        # 20260131_164257_435458 "good 19yrs", 15 deg tilt): dicom2nifti's
        # own axis-canonicalization heuristic (assign each array axis to
        # whichever world axis its affine column is most aligned with) can
        # get confused by a tilted slice axis and swap the slice axis with
        # an in-plane axis entirely - producing a NIfTI shaped e.g.
        # (512, 37, 559) instead of the correct (512, 559, 37) for a
        # 37-slice, 559x512 series, which then fails
        # `_nifti_alignment.align_dicom_to_nifti` outright ("no transpose/
        # flip combination matched any NIfTI slice's shape"). This module
        # never depended on dicom2nifti's own canonical orientation being
        # correct anyway - `_nifti_alignment.py` discovers the real
        # transpose/flip empirically by content-matching against the DICOM
        # slices themselves, so disabling this step only removes a source
        # of risk, not a capability actually being relied on.
        try:
            dicom2nifti.dicom_series_to_nifti(str(tmp_dicom_dir), str(out_path), reorient_nifti=False)
        except (IndexError, dicom2nifti.exceptions.ConversionValidationError, dicom2nifti.exceptions.ConversionError) as exc:
            # `dicom2nifti` refuses a series that can't become a usable
            # multi-slice volume, via two different code paths depending on
            # *why* - both confirmed as real cases, 2026-09-25/28, on
            # E:\Datasets\iViolin\data\PSR_58's own candidates:
            # 1. A whole series that's pure localizer/scout content (every
            #    file's `ImageType` containing "LOCALIZER") gets every file
            #    silently stripped by dicom2nifti's own internal `remove_
            #    non_imaging_slices`/`remove_duplicate_slices`/`remove_
            #    localizers_by_imagetype`, and its own "anything left?"
            #    check comes *after* the next step that needs at least one
            #    file - so it crashes with a bare, opaque `IndexError`
            #    (`dicom_input[0]` on an empty list) rather than its own
            #    `ConversionValidationError`.
            # 2. A series with too few real slices to begin with (e.g. a
            #    genuine single-slice series) is validated properly and
            #    raises dicom2nifti's own `ConversionValidationError
            #    ('TOO_FEW_SLICES/LOCALIZER')` - a real, named exception,
            #    but still a third-party one, not one of BICTool's own.
            # Both are translated into the same clear, actionable error
            # here rather than letting either leak through as-is - matches
            # `dicom_file_loader.py`'s own plain-`ValueError` convention for
            # "can't proceed with this input" cases. `ConversionError` is
            # dicom2nifti's other, more generic exception class (e.g. "no
            # DICOM files found") - caught alongside for the same reason,
            # even though no real case has hit it here yet.
            raise ValueError(
                "This series has no usable image content to segment - it looks like a scout/"
                "localizer series or has too few real slices. Pick a different series."
            ) from exc
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

    # No fixed reference index (e.g. "slice 0") - `align_dicom_to_nifti`
    # auto-selects a slice with real content to discover the orientation
    # transform from (see its own/`_nifti_alignment`'s docstring for why a
    # fixed choice is unsafe: a blank/uniform slice can silently lock in an
    # untested, wrong transform).
    alignment = align_dicom_to_nifti(dicom_slices, nifti_hu)
    return alignment.rekeyed(dicom_index_to_unique_index)
