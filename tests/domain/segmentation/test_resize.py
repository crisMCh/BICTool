import numpy as np

from bictool.domain.segmentation._resize import resize_to_512, restore_from_512


def test_pad_mode_used_for_a_smaller_than_512_slice():
    slice_hu = np.zeros((512, 434), dtype=np.float64)

    padded, context = resize_to_512(slice_hu)

    assert padded.shape == (512, 512)
    assert context.row_fit.mode == "pad"
    assert context.col_fit.mode == "pad"
    assert context.original_shape == (512, 434)


def test_pad_fills_with_the_slices_own_minimum_not_zero():
    slice_hu = np.full((512, 434), -1000.0, dtype=np.float64)  # all air
    slice_hu[0, 0] = 40.0  # one non-air pixel, so min() isn't trivially the fill itself

    padded, context = resize_to_512(slice_hu)

    # Every padded pixel (outside the original content region) equals the
    # slice's own minimum (-1000, air) - not 0, which sits inside the
    # soft-tissue HU range and would bias the model.
    top, left = context.row_fit.offset, context.col_fit.offset
    padded_only = padded.copy()
    padded_only[top : top + 512, left : left + 434] = -1000.0
    assert np.all(padded_only == -1000.0)


def test_round_trip_restores_the_exact_original_content_for_pad_mode():
    rng = np.random.default_rng(0)
    slice_hu = rng.uniform(-1000, 1000, size=(512, 434))

    padded, context = resize_to_512(slice_hu)
    # Simulate a "mask" that is True everywhere the real slice was placed.
    top, left = context.row_fit.offset, context.col_fit.offset
    mask_512 = np.zeros((512, 512), dtype=bool)
    mask_512[top : top + 512, left : left + 434] = True

    restored = restore_from_512(mask_512, context)

    assert restored.shape == slice_hu.shape
    assert np.all(restored)


def test_exact_512x512_slice_is_a_no_op_pad():
    slice_hu = np.zeros((512, 512), dtype=np.float64)

    padded, context = resize_to_512(slice_hu)

    assert context.row_fit.offset == 0 and context.col_fit.offset == 0
    np.testing.assert_array_equal(padded, slice_hu)


def test_larger_than_512_slice_is_center_cropped_not_resized():
    slice_hu = np.zeros((600, 600), dtype=np.float64)

    cropped, context = resize_to_512(slice_hu)

    assert cropped.shape == (512, 512)
    assert context.row_fit.mode == "crop"
    assert context.col_fit.mode == "crop"
    assert context.original_shape == (600, 600)


def test_crop_mode_round_trip_preserves_shape_and_content_location():
    slice_hu = np.zeros((600, 550), dtype=np.float64)
    _, context = resize_to_512(slice_hu)
    mask_512 = np.ones((512, 512), dtype=bool)

    restored = restore_from_512(mask_512, context)

    assert restored.shape == (600, 550)
    assert restored.dtype == bool
    # The cropped-away border (outside the centered 512x512 region) has no
    # model output, so it must come back False, not True.
    assert not restored[0, 0]
    assert not restored[-1, -1]
    row_off, col_off = context.row_fit.offset, context.col_fit.offset
    assert np.all(restored[row_off : row_off + 512, col_off : col_off + 512])


def test_mixed_axes_one_padded_one_cropped():
    # A real scanner matrix can exceed 512 in one axis while staying under
    # it in the other (e.g. 522x436, confirmed on a real Siemens NAEOTOM
    # Alpha export) - each axis must be fit independently.
    slice_hu = np.zeros((522, 436), dtype=np.float64)

    fitted, context = resize_to_512(slice_hu)

    assert fitted.shape == (512, 512)
    assert context.row_fit.mode == "crop"
    assert context.col_fit.mode == "pad"
