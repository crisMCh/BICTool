"""Tests for `integration.dicom_to_nifti`.

`_deduplicate` (file-based, reads `ImagePositionPatient`) runs always with
minimal hand-built DICOM files - never real patient data. `build_dicom_
nifti_volume`'s full pipeline (real `dicom2nifti` conversion, delegating
the actual orientation-transform discovery to `domain.segmentation.
_nifti_alignment` - see `tests/domain/segmentation/test_nifti_alignment.py`
for that logic's own pure-array tests) is exercised as a regression test
against the real scan this module was designed and verified against
(2026-09-17); skipped when that external, non-repo dataset isn't present
on the machine running the tests.
"""

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from bictool.domain.segmentation._nifti_alignment import AxisTransform, SliceNotSegmentableError
from bictool.integration.dicom_to_nifti import _convert_via_dicom2nifti, _deduplicate

_REAL_SCAN_DIR = Path(
    r"E:\Datasets\Siemens PhontonCounting\Siemens PhontonCounting\20250911_144228_435458 - bad\IMAGE\DCM\15704294"
)


def _make_position_only_dicom(directory: Path, filename: str, image_position_patient) -> Path:
    """A minimal DICOM file carrying only `ImagePositionPatient` - enough
    for `_deduplicate`, which reads no other tag."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    path = directory / filename
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.ImagePositionPatient = list(image_position_patient)
    ds.save_as(str(path), enforce_file_format=True)
    return path


class TestDeduplicate:
    def test_all_unique_positions_kept_in_order(self, tmp_path):
        paths = [_make_position_only_dicom(tmp_path, f"s{i}.dcm", (0.0, 0.0, float(i))) for i in range(3)]

        unique_paths, dicom_index_to_unique_index = _deduplicate(paths)

        assert unique_paths == paths
        assert dicom_index_to_unique_index == {0: 0, 1: 1, 2: 2}

    def test_duplicate_position_maps_to_the_first_occurrence(self, tmp_path):
        p0 = _make_position_only_dicom(tmp_path, "s0.dcm", (0.0, 0.0, 0.0))
        p1 = _make_position_only_dicom(tmp_path, "s1.dcm", (0.0, 0.0, 4.0))
        p2_dup = _make_position_only_dicom(tmp_path, "s2.dcm", (0.0, 0.0, 4.0))  # same as p1
        p3 = _make_position_only_dicom(tmp_path, "s3.dcm", (0.0, 0.0, 8.0))

        unique_paths, dicom_index_to_unique_index = _deduplicate([p0, p1, p2_dup, p3])

        assert unique_paths == [p0, p1, p3]
        assert dicom_index_to_unique_index == {0: 0, 1: 1, 2: 1, 3: 2}


def _make_minimal_ct_dicom(directory: Path, filename: str, *, localizer: bool) -> Path:
    """A minimal synthetic CT slice - real enough for `dicom2nifti` to
    process. `localizer=True` tags it `ImageType=LOCALIZER` so dicom2nifti's
    own internal `remove_localizers_by_imagetype` filtering strips it out;
    `localizer=False` leaves it as an ordinary (real) slice.
    """
    rows, cols = 8, 8
    pixels = np.full((rows, cols), 100, dtype=np.uint16)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    path = directory / filename
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PixelSpacing = ["0.5", "0.5"]
    ds.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"] if localizer else ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.Modality = "CT"
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.save_as(str(path), enforce_file_format=True)
    return path


class TestConvertViaDicom2NiftiUnusableSeries:
    """Both cases are real, confirmed dicom2nifti failures (2026-09-25/28,
    E:\\Datasets\\iViolin\\data\\PSR_58's own three candidates) for a series
    that can't become a usable multi-slice volume - reached via two
    different code paths inside dicom2nifti itself (see `_convert_via_
    dicom2nifti`'s own comment), both translated into the same clear
    BICTool-level `ValueError`.
    """

    def test_all_localizer_series_raises_a_clear_error_not_an_opaque_indexerror(self, tmp_path):
        # dicom2nifti's own internal filtering (remove_non_imaging_slices/
        # remove_duplicate_slices/remove_localizers_by_imagetype) can strip
        # out every single file in a series before ever checking anything
        # is left, then crashes with a bare `IndexError` deep in its own
        # code (`dicom_input[0]` on an empty list).
        path = _make_minimal_ct_dicom(tmp_path, "localizer.dcm", localizer=True)

        with pytest.raises(ValueError, match="scout/localizer"):
            _convert_via_dicom2nifti([path])

    def test_single_slice_series_raises_a_clear_error_not_a_raw_dicom2nifti_exception(self, tmp_path):
        # A genuine single-slice series (not a localizer) is properly
        # validated by dicom2nifti itself, which raises its own
        # `ConversionValidationError('TOO_FEW_SLICES/LOCALIZER')` - a real,
        # named exception, but still a third-party one that shouldn't leak
        # through as-is.
        path = _make_minimal_ct_dicom(tmp_path, "single_slice.dcm", localizer=False)

        with pytest.raises(ValueError, match="too few real slices"):
            _convert_via_dicom2nifti([path])


@pytest.mark.skipif(
    not _REAL_SCAN_DIR.is_dir(),
    reason="requires the real Siemens NAEOTOM Alpha scan this module was validated against (not part of the repo)",
)
class TestBuildDicomNiftiVolumeAgainstRealScan:
    """Regression test codifying the exact findings from this module's
    2026-09-17 validation against a real scan: one duplicate-position pair
    (slices 22/23), one geometry-sharing outlier with no NIfTI counterpart
    (slice 40, a scout/localizer), and a lossless transpose+flip transform -
    exercising the real `dicom2nifti` conversion + file dedup end-to-end,
    not just the pure alignment algorithm (see `test_nifti_alignment.py`).
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def dicom_nifti_volume():
        from bictool.integration.dicom_file_loader import discover_series, load_series
        from bictool.integration.dicom_to_nifti import build_dicom_nifti_volume

        candidates = discover_series(str(_REAL_SCAN_DIR))
        volume = load_series(candidates[0])
        return volume, build_dicom_nifti_volume(volume.slice_paths, volume.get_slice_hu)

    def test_discovers_the_known_transform(self, dicom_nifti_volume):
        # Was flip1=True until 2026-09-25's `reorient_nifti=False` fix (see
        # `_convert_via_dicom2nifti`'s docstring) - disabling dicom2nifti's
        # own axis canonicalization changes which specific flip its output
        # array happens to need, but not whether the discovery/mapping
        # itself is correct (see `test_forward_transform_matches_the_real_
        # nifti_slice_exactly`, unaffected by that change).
        _, dn = dicom_nifti_volume
        assert dn._transform == AxisTransform(transpose=True, flip0=False, flip1=False)

    def test_excludes_the_scout_localizer_outlier(self, dicom_nifti_volume):
        _, dn = dicom_nifti_volume
        assert dn.is_segmentable(40) is False
        with pytest.raises(SliceNotSegmentableError):
            dn.get_nifti_slice_hu(40)

    def test_duplicate_slices_share_the_same_nifti_slice(self, dicom_nifti_volume):
        _, dn = dicom_nifti_volume
        assert dn.is_segmentable(22) and dn.is_segmentable(23)
        np.testing.assert_array_equal(dn.get_nifti_slice_hu(22), dn.get_nifti_slice_hu(23))

    def test_forward_transform_matches_the_real_nifti_slice_exactly(self, dicom_nifti_volume):
        volume, dn = dicom_nifti_volume
        for dicom_index in (0, 10, 19, 25, 30):
            dicom_slice = volume.get_slice_hu(dicom_index)
            nifti_slice = dn.get_nifti_slice_hu(dicom_index)
            np.testing.assert_array_equal(dn._transform.apply(dicom_slice), nifti_slice)

    def test_mask_round_trips_back_to_the_dicom_slices_own_shape(self, dicom_nifti_volume):
        volume, dn = dicom_nifti_volume
        dicom_slice = volume.get_slice_hu(19)
        nifti_slice = dn.get_nifti_slice_hu(19)
        fake_mask = np.zeros(nifti_slice.shape, dtype=bool)
        fake_mask[5:10, 5:10] = True

        mapped_back = dn.map_mask_to_dicom_space(fake_mask)

        assert mapped_back.shape == dicom_slice.shape
