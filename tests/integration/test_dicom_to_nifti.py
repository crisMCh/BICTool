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
from bictool.integration.dicom_to_nifti import _deduplicate

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
        _, dn = dicom_nifti_volume
        assert dn._transform == AxisTransform(transpose=True, flip0=False, flip1=True)

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
