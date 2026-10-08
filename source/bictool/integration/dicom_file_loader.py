"""Loads DICOM data from local disk (FR-01, local-import path).

Distinct from `dicom_adapter.DicomAdapter` (PACS network retrieval, not yet
implemented) - this is the "Open DICOM..." local-file path used for manual
testing and analysis.

Real DICOM exports are almost never a flat folder of just the slices you
want: they're commonly nested under patient/study/series subfolders and
mixed in with scouts/localizers, dose report objects, and a `DICOMDIR` index
file - and a single *nominal* series (one `SeriesInstanceUID`) occasionally
turns out to contain more than one genuinely different image stack (a stray
localizer or secondary-capture sharing the main series' UID but a different
image matrix, for instance). `discover_series` handles all of this:

- A directory is scanned *recursively* - the user should be able to point at
  a top-level study folder without knowing which subfolder holds the images.
- Every readable DICOM image file found is grouped by `StudyInstanceUID`
  first; if the files found span more than one study, discovery is refused
  with a clear error rather than silently guessing (that's a sign the folder
  is an archive of multiple patients, not one export).
- Within a study, files are grouped by `SeriesInstanceUID`, and *each series
  is further split by image geometry* (`Rows`/`Columns`) before being
  offered as a candidate. This is not a workaround: it's the same approach
  GDCM/SimpleITK's `ImageSeriesReader.SetUseSeriesDetails` documents for
  exactly this situation ("multiple volumes are being grouped as a single
  series"), and what dcm2niix does when it separates a mismatched localizer
  from the main volume. The result is that every `SeriesCandidate` returned
  is internally consistent by construction - loading one can never fail with
  a "mismatched image dimensions" error, because an inconsistent nominal
  series is transparently split into separate, individually-loadable
  candidates instead.
- `discover_series` returns *every* candidate found (best-first: CT
  modality, then most slices), so the caller (see `main_window`'s series
  selector) can show the user what was found and let them pick, rather than
  the tool silently guessing among genuinely different acquisitions.

Four performance/safety choices matter for real exports (a study can easily
be hundreds of multi-hundred-KB files, and a folder scanned recursively may
contain non-DICOM files of any size or number - a "burn to CD" export in
particular commonly *doubles* the file count with a full JPEG rendering of
every image alongside the real DICOM series, plus a DICOMDIR index and
viewer software):

- Files are first filtered by filename/extension alone (`DICOMDIR`, `.jpg`,
  `.xml`, ...) with no file open at all - see `_SKIP_FILENAMES`/
  `_SKIP_EXTENSIONS`. On a real hospital export where the JPEG-preview
  folder alone can match the DICOM folder's file count 1:1, this cuts the
  number of files even considered in half before any of the steps below.
- The recursive scan/grouping/sorting phase reads every remaining candidate
  file with `stop_before_pixels=True` - only the (small) header is parsed,
  not the actual pixel bytes, since grouping and ordering only need header
  tags (`StudyInstanceUID`, `SeriesInstanceUID`, `Modality`, `InstanceNumber`,
  `ImagePositionPatient`, `Rows`/`Columns`). Reading full pixel data for
  every file just to decide which ones to keep would be the dominant cost
  for a large study.
- Before even asking pydicom to parse a file, a cheap 132-byte read checks
  for the standard DICOM preamble+magic. Only files that lack it (real, if
  less common, for exports that strip the preamble) fall back to a forced
  parse, and only if the file isn't implausibly large for a single CT slice
  - so a big non-DICOM file (a video, a zip, ...) sitting in the scanned
  tree gets skipped in one stat() call instead of being parsed byte-by-byte
  until it fails.
- `DicomVolume` does not eagerly decode pixel data for the whole chosen
  series either - it stores the sorted list of file paths and decodes (and
  caches) one slice's pixel data on first access via `get_slice_hu(index)`.
  The GUI only ever needs the currently displayed slice, so this avoids
  paying for slices the user never looks at.

All of the above still runs on a background thread (see
`application.dicom_load_worker.DicomLoadWorker`) - even a fast per-file scan
adds up to real, non-instant disk I/O for a large study, and must not block
the GUI thread regardless of how fast it's made.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pydicom

logger = logging.getLogger(__name__)

_DICOM_PREAMBLE_LEN = 128
_DICOM_MAGIC = b"DICM"
# A single CT slice (even high-res, multi-frame) is realistically well under
# this - anything bigger without a DICOM preamble is assumed not worth a
# forced parse attempt.
_MAX_FORCE_PARSE_BYTES = 20 * 1024 * 1024

# A "burn to CD" export (the IHE PDI profile most hospital PACS produce) is
# typically the real DICOM series PLUS a full duplicate JPEG rendering of
# every image, an XML manifest, viewer software, and a DICOMDIR index - on
# a real study this can double (or more) the number of files under the
# folder, none of which are DICOM images. These are skipped by filename/
# extension alone, with no file open at all, before the (already cheap)
# magic-byte check - on a >1000-file export this is the difference between
# a scan taking seconds vs. minutes, especially with per-file antivirus
# scanning overhead on Windows.
_SKIP_FILENAMES = {"dicomdir"}
_SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
    ".xml", ".html", ".htm", ".txt", ".pdf", ".doc", ".docx",
    ".exe", ".dll", ".msi", ".zip", ".gz", ".7z", ".rar",
    ".nii", ".json", ".csv", ".ini", ".log", ".db", ".mp4", ".avi",
}


@dataclass
class DicomSlice:
    """A single loaded DICOM slice: pixel data in Hounsfield units plus header metadata.

    Field set mirrors the standard corner-overlay content real DICOM viewers
    (RadiAnt/OsiriX/MicroDicom) show on the image itself - see
    `presentation.views.main_window._update_viewer_overlay`'s docstring for
    the research this is based on. Every field is best-effort (`None` when
    the tag is absent) since real-world exports routinely omit optional tags.
    """

    pixels_hu: np.ndarray
    pixel_spacing_mm: float
    slice_location_mm: float | None
    patient_id: str | None
    patient_name: str | None
    patient_sex: str | None
    study_id: str | None
    study_date: str | None
    study_description: str | None
    series_description: str | None
    series_number: str | None
    institution_name: str | None
    manufacturer_model: str | None
    acquisition_date: str | None
    acquisition_time: str | None
    kvp: float | None
    tube_current_ma: float | None
    transfer_syntax_name: str | None
    modality: str | None
    ctdi_vol_mgy: float | None
    dlp_mgy_cm: float | None
    reconstruction: str | None


@dataclass
class DicomVolume:
    """A lazy view over one DICOM series: `slice_paths` is the full series in
    display order, but pixel data for a given slice is only decoded (and
    then cached) the first time `get_slice_hu` is called for it - loading a
    volume does not mean decoding every slice up front. Metadata is taken
    from the first slice in series order - CTDIvol/DLP etc. are normally
    study/series-level values that don't vary slice-to-slice, so this is not
    a meaningful loss of information. `slice_locations_mm` is the one
    per-slice exception (it genuinely varies slice-to-slice) - indexed the
    same as `slice_paths`.
    """

    slice_paths: list[Path]
    pixel_spacing_mm: float
    slice_thickness_mm: float | None
    slice_locations_mm: list[float | None]
    patient_id: str | None
    patient_name: str | None
    patient_sex: str | None
    study_id: str | None
    study_date: str | None
    study_description: str | None
    series_description: str | None
    series_number: str | None
    institution_name: str | None
    manufacturer_model: str | None
    acquisition_date: str | None
    acquisition_time: str | None
    kvp: float | None
    tube_current_ma: float | None
    transfer_syntax_name: str | None
    modality: str | None
    ctdi_vol_mgy: float | None
    dlp_mgy_cm: float | None
    reconstruction: str | None
    _slice_cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    @property
    def num_slices(self) -> int:
        return len(self.slice_paths)

    def get_slice_hu(self, index: int) -> np.ndarray:
        """Decodes slice `index`'s pixel data to Hounsfield units, caching it
        for subsequent calls. Cache is unbounded (kept for the volume's
        lifetime) - fine for the slice counts a single head CT study has;
        would need an eviction policy if this ever loaded much larger series.
        """
        if index not in self._slice_cache:
            ds = pydicom.dcmread(str(self.slice_paths[index]))
            self._slice_cache[index] = _rescale_to_hu(ds)
        return self._slice_cache[index]


@dataclass
class SeriesCandidate:
    """One coherent, loadable stack of DICOM slices - same series *and* the
    same image geometry (Rows/Columns). See the module docstring for why a
    single nominal series can produce more than one of these.

    `label` is meant for a GUI series-selector (e.g. a dropdown): it names
    what actually distinguishes candidates for a human - modality, slice
    count, image matrix size, and reconstruction kernel where available.

    `slice_locations_mm` carries each slice's through-plane position
    (`_slice_location_mm`, same ordering as `sorted_paths`) forward from the
    header scan already done by `discover_series` - it's per-slice (unlike
    the rest of this project's metadata), so it can't be read once from
    `_reference_header` alone; capturing it here means `load_series` never
    has to re-read every slice's header just to show which one is on screen.
    """

    label: str
    sorted_paths: list[Path]
    slice_locations_mm: list[float | None]
    _reference_header: pydicom.Dataset = field(repr=False, compare=False)
    # Whether the slices form an axial stack (see `_stack_is_axial`); None =
    # not determined, then the reference header's orientation decides.
    axial: bool | None = None

    @property
    def num_slices(self) -> int:
        return len(self.sorted_paths)


def load_dicom_slice(path: str | Path) -> DicomSlice:
    """Reads one DICOM file and returns its pixel data (HU) and header metadata."""
    ds = pydicom.dcmread(str(path))
    pixels_hu = _rescale_to_hu(ds)
    return DicomSlice(
        pixels_hu=pixels_hu,
        pixel_spacing_mm=_pixel_spacing_mm(ds),
        slice_location_mm=_slice_location_mm(ds),
        **_extract_metadata(ds),
    )


def discover_series(
    path: str | Path, on_progress: Callable[[int], None] | None = None
) -> list[SeriesCandidate]:
    """Recursively scans directory `path` and returns every coherent
    series/stack found, best-first (CT modality preferred, then most
    slices) - see module docstring for the full pipeline. Raises
    `ValueError` if no DICOM image files are found, or if they span more
    than one study.

    `on_progress`, if given, is called with the number of files scanned so
    far every 25 files - the scan is real, non-instant disk I/O for a large
    study, and the caller (see `DicomLoadWorker`) uses this to keep the
    GUI's status visible while it runs (NFR-07).
    """
    p = Path(path)
    logger.info("discover_series: path=%s", p)

    all_files = [f for f in sorted(p.rglob("*")) if f.is_file()]
    candidate_files = [f for f in all_files if not _looks_like_non_dicom(f)]
    logger.debug(
        "Recursive scan: %d file(s) found, %d skipped by filename/extension, %d candidate(s) remain",
        len(all_files),
        len(all_files) - len(candidate_files),
        len(candidate_files),
    )
    datasets = []
    for i, f in enumerate(candidate_files, start=1):
        ds = _try_read_header(f)
        if ds is not None:
            datasets.append(ds)
        if on_progress is not None and i % 25 == 0:
            on_progress(i)
    logger.debug(
        "Header scan complete: %d readable DICOM image file(s) of %d candidate(s)",
        len(datasets), len(candidate_files),
    )
    if not datasets:
        raise ValueError(f"No DICOM image files found under {p} (searched recursively)")

    study_groups = _group_by(datasets, "StudyInstanceUID")
    logger.debug("Grouped into %d distinct stud(y/ies)", len(study_groups))
    if len(study_groups) > 1:
        patients = sorted({str(getattr(g[0], "PatientID", "")) or "?" for g in study_groups.values()})
        logger.error("Refusing to load: %d studies found under %s (patients: %s)", len(study_groups), p, patients)
        raise ValueError(
            f"Found {len(study_groups)} different studies ({len(datasets)} DICOM files total) under {p} - "
            f"patients: {', '.join(patients)}. Point at a single study's folder instead of one "
            "containing multiple patients/studies."
        )

    candidate_groups = _split_into_candidate_groups(datasets)
    candidates = [_build_candidate(group) for group in candidate_groups]
    candidates.sort(key=_candidate_sort_key)
    logger.info(
        "Discovered %d series candidate(s): %s",
        len(candidates), [c.label for c in candidates],
    )
    return candidates


def load_series(candidate: SeriesCandidate) -> DicomVolume:
    """Builds a `DicomVolume` from an already-discovered `SeriesCandidate` -
    no re-scanning, since `discover_series` already found and sorted the
    files. Cheap: only the reference header (already in memory) is used for
    metadata: no pixel data is decoded here either.
    """
    ref = candidate._reference_header
    return DicomVolume(
        slice_paths=candidate.sorted_paths,
        pixel_spacing_mm=_pixel_spacing_mm(ref),
        slice_thickness_mm=_as_float_or_none(getattr(ref, "SliceThickness", None)),
        slice_locations_mm=candidate.slice_locations_mm,
        **_extract_metadata(ref),
    )


def discover(path: str | Path, on_progress: Callable[[int], None] | None = None) -> list[SeriesCandidate]:
    """Unified discovery entry point: a directory is scanned via
    `discover_series`; a single file is wrapped as one single-slice
    candidate. The caller (see `DicomLoadWorker`/`main_window`'s series
    selector) always ends up with "a list of candidates" either way,
    regardless of what the user picked in the file dialog.
    """
    p = Path(path)
    if p.is_dir():
        return discover_series(p, on_progress=on_progress)

    logger.info("Single-file discovery: %s", p)
    reference_ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    modality = getattr(reference_ds, "Modality", None) or "?"
    rows = getattr(reference_ds, "Rows", "?")
    columns = getattr(reference_ds, "Columns", "?")
    label = f"{modality} · 1 slice(s) · {rows}×{columns}"
    return [
        SeriesCandidate(
            label=label,
            sorted_paths=[p],
            slice_locations_mm=[_slice_location_mm(reference_ds)],
            _reference_header=reference_ds,
        )
    ]


def load_dicom_volume(path: str | Path, on_progress: Callable[[int], None] | None = None) -> DicomVolume:
    """Convenience wrapper for the simple "just give me a volume" case
    (tests, callers that don't need to offer a choice): auto-selects
    `discover`'s first (best) candidate.

    Prefer `discover` + `load_series` directly when the caller wants to let
    the user choose among multiple series instead of auto-selecting (see
    `main_window`'s series selector).
    """
    candidates = discover(path, on_progress=on_progress)
    return load_series(candidates[0])


def _looks_like_non_dicom(path: Path) -> bool:
    """Cheap filename/extension check to skip obvious clutter (JPEG previews,
    `DICOMDIR`, manifests, viewer software...) without opening the file at
    all - see the `_SKIP_*` comment above for why this matters at scale.
    """
    return path.name.lower() in _SKIP_FILENAMES or path.suffix.lower() in _SKIP_EXTENSIONS


def _has_dicom_preamble(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(_DICOM_PREAMBLE_LEN + len(_DICOM_MAGIC))
    except OSError:
        return False
    return header[_DICOM_PREAMBLE_LEN:] == _DICOM_MAGIC


def _try_read_header(path: Path) -> pydicom.Dataset | None:
    """Reads just `path`'s header (no pixel data - see module docstring),
    returning None (not raising) if it isn't DICOM at all, or is DICOM but
    not an image (e.g. `DICOMDIR`, a Structured Report/dose object) - so a
    recursive folder scan can quietly skip clutter instead of failing on the
    first non-image file it meets.

    Files with the standard preamble are read directly; those without one
    (real-world exports sometimes strip it) only get a forced parse attempt
    if they're not implausibly large for a single CT slice - see the
    `_MAX_FORCE_PARSE_BYTES` comment.
    """
    try:
        if _has_dicom_preamble(path):
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        else:
            if path.stat().st_size > _MAX_FORCE_PARSE_BYTES:
                return None
            ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
    except Exception:
        return None
    if "Rows" not in ds or "Columns" not in ds:
        return None
    return ds


def _group_by(datasets: list[pydicom.Dataset], uid_attribute: str) -> dict[str, list[pydicom.Dataset]]:
    groups: dict[str, list[pydicom.Dataset]] = defaultdict(list)
    for ds in datasets:
        uid = getattr(ds, uid_attribute, "") or ""
        groups[uid].append(ds)
    return groups


def _split_into_candidate_groups(datasets: list[pydicom.Dataset]) -> list[list[pydicom.Dataset]]:
    """Groups by `SeriesInstanceUID`, then further splits each series by
    image geometry (`Rows`/`Columns`) - see module docstring for why this
    two-level grouping is standard practice (GDCM/SimpleITK's
    `SetUseSeriesDetails`), not a workaround. Guarantees every returned
    group is internally shape-consistent, so loading one can never fail
    with a dimension-mismatch error.
    """
    series_groups = _group_by(datasets, "SeriesInstanceUID")
    candidate_groups: list[list[pydicom.Dataset]] = []
    for series_uid, group in series_groups.items():
        by_geometry: dict[tuple, list[pydicom.Dataset]] = defaultdict(list)
        for ds in group:
            geometry = (getattr(ds, "Rows", None), getattr(ds, "Columns", None))
            by_geometry[geometry].append(ds)
        if len(by_geometry) > 1:
            logger.warning(
                "Series %s contains %d distinct image geometries %s - treating as separate candidates "
                "(a stray localizer/secondary-capture sharing a series UID is a common real-world cause)",
                series_uid, len(by_geometry), sorted(by_geometry.keys(), key=str),
            )
        candidate_groups.extend(by_geometry.values())
    return candidate_groups


def _build_candidate(group: list[pydicom.Dataset]) -> SeriesCandidate:
    sorted_group = sorted(group, key=_slice_sort_key)
    reference = sorted_group[0]
    modality = getattr(reference, "Modality", None) or "?"
    rows = getattr(reference, "Rows", "?")
    columns = getattr(reference, "Columns", "?")
    label_parts = [str(modality), f"{len(sorted_group)} slice(s)", f"{rows}×{columns}"]
    kernel = getattr(reference, "ConvolutionKernel", None)
    if kernel:
        label_parts.append(str(kernel))
    thickness = _as_float_or_none(getattr(reference, "SliceThickness", None))
    if thickness is not None:
        label_parts.append(f"{thickness:g}mm")
    axial = _stack_is_axial(sorted_group)
    if axial:
        label_parts.append("axial")
    return SeriesCandidate(
        label=" · ".join(label_parts),
        sorted_paths=[Path(ds.filename) for ds in sorted_group],
        slice_locations_mm=[_slice_location_mm(ds) for ds in sorted_group],
        _reference_header=reference,
        axial=axial,
    )


# The routine head-CT protocol this tool targets (WP2) reconstructs axial
# slices at 4mm thickness for the standard reading series - thinner/thicker
# or reformatted (sagittal/coronal) series in the same study are usually
# secondary reconstructions, not the one to default to.
_PREFERRED_SLICE_THICKNESS_MM = 4.0


# A slice stack counts as axial when its normal is within this angle of the
# patient's Z axis - tilted/angulated axial head CT included (a real Siemens
# series with 15 deg gantry tilt plus angulation sat ~47 deg off Z), true
# coronal/sagittal stacks (90 deg) not.
_AXIAL_MAX_ANGLE_DEG = 60.0


def _is_axial(ds: pydicom.Dataset) -> bool:
    """True if `ImageOrientationPatient`'s row/column direction cosines
    describe an axial slice - the slice normal (their cross product) within
    `_AXIAL_MAX_ANGLE_DEG` of the patient's Z axis. Unknown (tag absent) is
    treated as *not* axial - it's a guess either way, and a missing tag
    shouldn't be scored as if it positively confirmed axial.

    Only a fallback - see `_stack_is_axial`: some scanners write the source
    volume's orientation into coronal/sagittal reformats.
    """
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if orientation is None or len(orientation) != 6:
        return False
    # Built element-by-element via `_parse_decimal` (not `np.array(...,
    # dtype=float)` directly) - numpy's own string->float conversion has
    # the same comma-decimal-separator problem plain `float()` does, see
    # `_parse_decimal`'s docstring.
    row = np.array([_parse_decimal(v) for v in orientation[0:3]])
    col = np.array([_parse_decimal(v) for v in orientation[3:6]])
    return _normal_is_axial(np.cross(row, col))


def _normal_is_axial(normal: np.ndarray) -> bool:
    norm = np.linalg.norm(normal)
    if norm == 0:
        return False
    return bool(abs(normal[2]) / norm >= np.cos(np.radians(_AXIAL_MAX_ANGLE_DEG)))


def _stack_is_axial(group: list[pydicom.Dataset]) -> bool:
    """Whether a series' slices are stacked axially - judged from where the
    slices actually are (`ImagePositionPatient`), not from the orientation
    tag: real Siemens photon-counting exports write the source volume's
    axial `ImageOrientationPatient` (and ImageType "AXIAL") into their
    coronal ("kor") and sagittal ("sag") MPR reformats, whose slices then
    step along Y/X. The stacking direction is the main axis of all slice
    positions (order-independent, robust to a stray slice). Series with
    fewer than 3 positioned slices fall back to `_is_axial`.
    """
    positions = []
    for ds in group:
        position = getattr(ds, "ImagePositionPatient", None)
        if position is not None and len(position) == 3:
            positions.append([_parse_decimal(v) for v in position])
    points = np.array(positions, dtype=np.float64)
    if len(points) < 3 or np.allclose(points, points[0]):
        return _is_axial(group[0])
    _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    return _normal_is_axial(vt[0])


def _candidate_sort_key(candidate: SeriesCandidate) -> tuple:
    """Best-first ordering: prefer CT modality (a study folder often also has
    scouts/localizers or other modalities mixed in), then axial orientation
    over sagittal/coronal/oblique reformats, then the slice thickness
    closest to the standard 4mm head-CT reading series, then the most
    slices, breaking ties deterministically by the first slice's path.
    """
    ref = candidate._reference_header
    is_ct = getattr(ref, "Modality", None) == "CT"
    is_axial = candidate.axial if candidate.axial is not None else _is_axial(ref)
    thickness = _as_float_or_none(getattr(ref, "SliceThickness", None))
    thickness_distance = abs(thickness - _PREFERRED_SLICE_THICKNESS_MM) if thickness is not None else float("inf")
    return (
        0 if is_ct else 1,
        0 if is_axial else 1,
        thickness_distance,
        -candidate.num_slices,
        str(candidate.sorted_paths[0]),
    )


def _slice_sort_key(ds: pydicom.Dataset) -> tuple:
    """Orders slices spatially by `ImagePositionPatient`'s through-plane
    coordinate where available (the robust way, per the DICOM standard -
    handles gaps/re-numbering correctly for axial series), falling back to
    `InstanceNumber`, then filename.
    """
    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None and len(position) == 3:
        return (0, _parse_decimal(position[2]))
    instance_number = getattr(ds, "InstanceNumber", None)
    if instance_number is not None:
        return (1, int(instance_number))
    return (2, str(getattr(ds, "filename", "")))


def _rescale_to_hu(ds: pydicom.Dataset) -> np.ndarray:
    """Converts pixel data to Hounsfield units via RescaleSlope/RescaleIntercept
    (defaulting to slope 1, intercept 0 if absent, per the DICOM standard).
    """
    pixels = ds.pixel_array.astype(np.float64)
    slope = _parse_decimal(getattr(ds, "RescaleSlope", 1))
    intercept = _parse_decimal(getattr(ds, "RescaleIntercept", 0))
    return pixels * slope + intercept


def _pixel_spacing_mm(ds: pydicom.Dataset) -> float:
    """Row spacing from the DICOM `PixelSpacing` tag (row, column) - assumes
    square pixels, matching this project's metric functions, which take one
    scalar `pix_spc` (see `domain.core_api`).
    """
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    return _parse_decimal(pixel_spacing[0]) if pixel_spacing else 1.0


def _extract_metadata(ds: pydicom.Dataset) -> dict:
    manufacturer = str(getattr(ds, "Manufacturer", "")).strip()
    model = str(getattr(ds, "ManufacturerModelName", "")).strip()
    manufacturer_model = " ".join(part for part in (manufacturer, model) if part) or None
    return dict(
        patient_id=str(getattr(ds, "PatientID", "")) or None,
        patient_name=_format_person_name(getattr(ds, "PatientName", None)),
        patient_sex=str(getattr(ds, "PatientSex", "")).strip() or None,
        study_id=str(getattr(ds, "StudyID", "")) or None,
        study_date=str(getattr(ds, "StudyDate", "")) or None,
        study_description=str(getattr(ds, "StudyDescription", "")).strip() or None,
        series_description=str(getattr(ds, "SeriesDescription", "")).strip() or None,
        series_number=str(getattr(ds, "SeriesNumber", "")) or None,
        institution_name=str(getattr(ds, "InstitutionName", "")).strip() or None,
        manufacturer_model=manufacturer_model,
        acquisition_date=str(getattr(ds, "AcquisitionDate", "")) or None,
        acquisition_time=_format_time(getattr(ds, "AcquisitionTime", None)),
        kvp=_as_float_or_none(getattr(ds, "KVP", None)),
        tube_current_ma=_as_float_or_none(getattr(ds, "XRayTubeCurrent", None)),
        transfer_syntax_name=_transfer_syntax_name(ds),
        modality=str(getattr(ds, "Modality", "")) or None,
        # CTDIvol/DLP are frequently absent from the CT Image IOD itself (often
        # only present in a separate Radiation Dose SR object) - best-effort only.
        ctdi_vol_mgy=_as_float_or_none(getattr(ds, "CTDIvol", None)),
        dlp_mgy_cm=_as_float_or_none(getattr(ds, "DLP", None)),
        reconstruction=str(getattr(ds, "ConvolutionKernel", "")) or None,
    )


def _parse_decimal(value: object) -> float:
    """Parses a DICOM DS (decimal string) tag value as a float, tolerating
    a comma decimal separator.

    The DICOM standard (PS3.5) requires a period regardless of locale, but
    a real Siemens NAEOTOM Alpha export (confirmed 2026-09-24 -
    `ImagePositionPatient`'s through-plane coordinate written as the string
    "-748,312", crashing plain `float()`) violates this - evidently a
    European-locale bug in the scanner's own export tool, not something
    pydicom itself corrects (it stores the tag's raw text when it can't
    coerce it to its own DS type, deferring the failure to here). Every
    float extracted from a raw DICOM tag value in this module goes through
    this rather than a bare `float(value)` call, since the same exported
    study can plausibly have the same locale bug in any other DS-valued tag
    (`RescaleSlope`, `PixelSpacing`, `KVP`, ...), not just this one.
    """
    text = str(value)
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    return float(text)


def _as_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return _parse_decimal(value)
    except (TypeError, ValueError):
        return None


def _format_person_name(value: object) -> str | None:
    """`PatientName` comes back from pydicom as a `PersonName` in DICOM's
    "Last^First^Middle" component order - reformatted to a plain
    space-separated display string, matching how every real DICOM viewer's
    overlay renders it rather than the raw DICOM encoding.
    """
    if not value:
        return None
    text = str(value).replace("^", " ").strip()
    return text or None


def _format_time(value: object) -> str | None:
    """DICOM `TM` values are `HHMMSS.FFFFFF` with no separators - reformatted
    to `HH:MM:SS` for display. Falls back to the raw string for a value that
    doesn't match the expected shape, rather than raising on odd real-world data.
    """
    text = str(value or "").strip()
    if len(text) >= 6 and text[:6].isdigit():
        return f"{text[0:2]}:{text[2:4]}:{text[4:6]}"
    return text or None


def _transfer_syntax_name(ds: pydicom.Dataset) -> str | None:
    """The file meta's transfer syntax as a human-readable name (e.g.
    "Implicit VR Little Endian") rather than its raw UID - shown by real
    DICOM viewers (see MicroDicom's overlay) as a quick compression/encoding
    indicator. Header-only reads (`stop_before_pixels=True`) still parse
    file_meta, so this is available even before pixel data is decoded.
    """
    file_meta = getattr(ds, "file_meta", None)
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None) if file_meta is not None else None
    if transfer_syntax is None:
        return None
    return getattr(transfer_syntax, "name", None) or str(transfer_syntax) or None


def _slice_location_mm(ds: pydicom.Dataset) -> float | None:
    """Prefers `ImagePositionPatient`'s through-plane coordinate (matches
    `_slice_sort_key`'s ordering logic, and is the standard's recommended
    source per PS3.3) - falls back to the dedicated `SliceLocation` tag,
    which some scanners populate even when position data is absent.
    """
    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None and len(position) == 3:
        return _parse_decimal(position[2])
    return _as_float_or_none(getattr(ds, "SliceLocation", None))
