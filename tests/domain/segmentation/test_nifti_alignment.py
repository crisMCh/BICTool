"""Tests for `domain.segmentation._nifti_alignment` - pure array logic, no
file I/O, no real DICOM/`dicom2nifti` dependency, so everything here runs
with synthetic data. The real-data regression test (verifying the actual
transform discovered on a real scan) lives in
`tests/integration/test_dicom_to_nifti.py`, since building the real NIfTI
volume needs the real `dicom2nifti` conversion pipeline.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bictool.domain.segmentation._nifti_alignment import (
    AxisTransform,
    OrientationError,
    SliceNotSegmentableError,
    _match_slice_to_nifti,
    align_dicom_to_nifti,
)


class TestAxisTransform:
    def test_identity_round_trips(self):
        arr = np.arange(20).reshape(4, 5)
        t = AxisTransform(transpose=False, flip0=False, flip1=False)
        np.testing.assert_array_equal(t.invert(t.apply(arr)), arr)

    @pytest.mark.parametrize("transpose", [False, True])
    @pytest.mark.parametrize("flip0", [False, True])
    @pytest.mark.parametrize("flip1", [False, True])
    def test_every_combination_round_trips(self, transpose, flip0, flip1):
        arr = np.arange(20).reshape(4, 5).astype(np.float64)
        t = AxisTransform(transpose=transpose, flip0=flip0, flip1=flip1)
        np.testing.assert_array_equal(t.invert(t.apply(arr)), arr)

    def test_transpose_changes_shape(self):
        arr = np.zeros((4, 5))
        t = AxisTransform(transpose=True, flip0=False, flip1=False)
        assert t.apply(arr).shape == (5, 4)

    def test_flip_only_does_not_change_shape(self):
        arr = np.zeros((4, 5))
        t = AxisTransform(transpose=False, flip0=True, flip1=True)
        assert t.apply(arr).shape == (4, 5)
        np.testing.assert_array_equal(t.apply(arr), arr[::-1, ::-1])


class TestMatchSliceToNifti:
    def test_finds_the_exact_match(self):
        nifti_hu = np.stack([np.full((3, 3), v, dtype=np.float64) for v in (10, 20, 30)], axis=-1)
        target = np.full((3, 3), 20.0)

        match = _match_slice_to_nifti(target, nifti_hu, claimed=set())

        assert match is not None
        nifti_index, mad = match
        assert nifti_index == 1
        assert mad == 0.0

    def test_ignores_already_claimed_slices(self):
        nifti_hu = np.stack([np.full((3, 3), v, dtype=np.float64) for v in (10, 20, 30)], axis=-1)
        target = np.full((3, 3), 20.0)

        match = _match_slice_to_nifti(target, nifti_hu, claimed={1})

        assert match is not None
        nifti_index, _mad = match
        assert nifti_index != 1  # claimed, so the (worse) next-best is returned instead

    def test_returns_none_when_no_slice_has_a_matching_shape(self):
        nifti_hu = np.zeros((3, 3, 2))
        target = np.zeros((5, 5))

        assert _match_slice_to_nifti(target, nifti_hu, claimed=set()) is None


def _make_synthetic_scan(transform: AxisTransform, n_slices: int = 4, shape=(6, 5)):
    """Builds a synthetic (dicom_slices, nifti_hu) pair related by exactly
    `transform` - lets `align_dicom_to_nifti` be tested without any real
    DICOM/NIfTI data, since it never touches files itself."""
    rng = np.random.default_rng(0)
    dicom_slices = {i: rng.uniform(-100, 100, size=shape) for i in range(n_slices)}
    nifti_hu = np.stack([transform.apply(dicom_slices[i]) for i in range(n_slices)], axis=-1)
    return dicom_slices, nifti_hu


class TestAlignDicomToNifti:
    def test_discovers_the_correct_transform(self):
        real_transform = AxisTransform(transpose=True, flip0=False, flip1=True)
        dicom_slices, nifti_hu = _make_synthetic_scan(real_transform)

        result = align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=0)

        assert result._transform == real_transform

    def test_matches_every_slice_to_its_own_nifti_counterpart(self):
        real_transform = AxisTransform(transpose=False, flip0=True, flip1=False)
        dicom_slices, nifti_hu = _make_synthetic_scan(real_transform, n_slices=5)

        result = align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=2)

        for i in range(5):
            assert result.is_segmentable(i)
            np.testing.assert_array_equal(result.get_nifti_slice_hu(i), nifti_hu[:, :, i])

    def test_excludes_a_slice_with_no_matching_nifti_slice(self):
        real_transform = AxisTransform(transpose=False, flip0=False, flip1=False)
        dicom_slices, nifti_hu = _make_synthetic_scan(real_transform, n_slices=4)
        dicom_slices[99] = np.full((6, 5), 12345.0)  # no counterpart in nifti_hu at all

        result = align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=0)

        assert result.is_segmentable(99) is False
        with pytest.raises(SliceNotSegmentableError):
            result.get_nifti_slice_hu(99)

    def test_raises_orientation_error_when_no_combination_matches_well(self):
        dicom_slices = {0: np.random.default_rng(1).uniform(-100, 100, size=(6, 5))}
        nifti_hu = np.random.default_rng(2).uniform(-100, 100, size=(6, 5, 1))  # unrelated data

        with pytest.raises(OrientationError):
            align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=0)

    def test_map_mask_to_dicom_space_round_trips(self):
        real_transform = AxisTransform(transpose=True, flip0=True, flip1=False)
        dicom_slices, nifti_hu = _make_synthetic_scan(real_transform)
        result = align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=0)
        nifti_mask = np.zeros(nifti_hu.shape[:2], dtype=bool)
        nifti_mask[1:3, 1:3] = True

        mapped_back = result.map_mask_to_dicom_space(nifti_mask)

        assert mapped_back.shape == dicom_slices[0].shape
        np.testing.assert_array_equal(result._transform.apply(mapped_back), nifti_mask)

    def test_remap_result_to_dicom_space_preserves_result_type(self):
        real_transform = AxisTransform(transpose=False, flip0=False, flip1=True)
        dicom_slices, nifti_hu = _make_synthetic_scan(real_transform)
        result = align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=0)
        nifti_mask = np.zeros(nifti_hu.shape[:2], dtype=bool)
        nifti_mask[0, 0] = True
        fake_segmentation_result = SimpleNamespace(masks={"Structure": nifti_mask})

        remapped = result.remap_result_to_dicom_space(fake_segmentation_result)

        assert isinstance(remapped, SimpleNamespace)
        assert remapped.masks["Structure"].shape == dicom_slices[0].shape

    def test_rekeyed_expands_duplicate_indices(self):
        real_transform = AxisTransform(transpose=False, flip0=False, flip1=False)
        dicom_slices, nifti_hu = _make_synthetic_scan(real_transform, n_slices=2)
        result = align_dicom_to_nifti(dicom_slices, nifti_hu, reference_key=0)

        expanded = result.rekeyed({10: 0, 11: 0, 12: 1})

        assert expanded.is_segmentable(10) and expanded.is_segmentable(11) and expanded.is_segmentable(12)
        np.testing.assert_array_equal(expanded.get_nifti_slice_hu(10), expanded.get_nifti_slice_hu(11))
