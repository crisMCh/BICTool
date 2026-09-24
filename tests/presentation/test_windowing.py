import numpy as np

from bictool.presentation.viewmodels.windowing import WINDOW_PRESETS, apply_window


def test_apply_window_maps_center_to_mid_gray():
    pixels = np.array([[35.0]])
    result = apply_window(pixels, width=90.0, center=35.0)
    assert result[0, 0] == 127  # (35 - (-10)) / 90 * 255 = 127.5 -> truncated to 127


def test_apply_window_clips_below_and_above_range():
    pixels = np.array([[-1000.0, 5000.0]])
    result = apply_window(pixels, width=90.0, center=35.0)
    np.testing.assert_array_equal(result, [[0, 255]])


def test_apply_window_linearly_scales_within_range():
    low, high = -10.0, 80.0  # center=35, width=90
    pixels = np.array([[low, (low + high) / 2, high]])
    result = apply_window(pixels, width=90.0, center=35.0)
    np.testing.assert_array_equal(result, [[0, 127, 255]])


def test_apply_window_returns_black_for_non_positive_width():
    pixels = np.array([[10.0, 20.0]])
    result = apply_window(pixels, width=0.0, center=35.0)
    np.testing.assert_array_equal(result, [[0, 0]])


def test_apply_window_output_dtype_is_uint8():
    pixels = np.array([[35.0]])
    result = apply_window(pixels, width=90.0, center=35.0)
    assert result.dtype == np.uint8


def test_window_presets_match_wp2_documented_ranges():
    # WP2 section 3: soft tissue/brain/CSF window WW 80-100 HU, WL 30-40 HU;
    # bone/vessels window WW 2000-4000 HU, WL 300-500 HU. Presets use the
    # midpoint of each documented range.
    brain_width, brain_center = WINDOW_PRESETS["Brain"]
    assert 80.0 <= brain_width <= 100.0
    assert 30.0 <= brain_center <= 40.0

    bone_width, bone_center = WINDOW_PRESETS["Bone"]
    assert 2000.0 <= bone_width <= 4000.0
    assert 300.0 <= bone_center <= 500.0
