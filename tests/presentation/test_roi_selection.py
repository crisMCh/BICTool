import numpy as np

from bictool.domain.roi_selection.structured_roi import RoiCandidate
from bictool.presentation.viewmodels.roi_selection import (
    RoiRect,
    candidate_to_rect,
    crop_roi,
    make_square,
    normalize_rect,
)


def test_candidate_to_rect_swaps_row_col_into_x_y():
    candidate = RoiCandidate(row0=30, col0=10, size=20, structure="Dural Folds")

    assert candidate_to_rect(candidate) == RoiRect(10, 30, 30, 50)


def test_normalize_rect_orders_reversed_corners():
    rect = normalize_rect(xa=50, ya=40, xb=10, yb=5, image_width=100, image_height=100)
    assert rect == RoiRect(10, 5, 50, 40)


def test_normalize_rect_clamps_to_image_bounds():
    rect = normalize_rect(xa=-20, ya=-5, xb=200, yb=150, image_width=100, image_height=90)
    assert rect == RoiRect(0, 0, 100, 90)


def test_crop_roi_extracts_expected_block():
    pixels = np.arange(25).reshape(5, 5)
    rect = RoiRect(x0=1, y0=1, x1=3, y1=4)

    cropped = crop_roi(pixels, rect)

    np.testing.assert_array_equal(cropped, pixels[1:4, 1:3])
    assert cropped.shape == (3, 2)


def test_make_square_leaves_an_already_square_rect_unchanged():
    rect = RoiRect(10, 10, 30, 30)
    result = make_square(rect, image_width=100, image_height=100)
    assert result == rect


def test_make_square_expands_shorter_dimension_to_match_the_longer():
    # 40 wide x 30 tall -> squared to 40x40, centered on the same point.
    rect = RoiRect(x0=10, y0=10, x1=50, y1=40)
    result = make_square(rect, image_width=100, image_height=100)
    assert result.width == result.height == 40
    # center preserved: original center is (30, 25)
    assert (result.x0 + result.x1) / 2 == 30
    assert (result.y0 + result.y1) / 2 == 25


def test_make_square_result_is_always_square():
    for x1, y1 in [(43, 42), (42, 43), (11, 50), (99, 5)]:
        rect = RoiRect(0, 0, x1, y1)
        result = make_square(rect, image_width=200, image_height=200)
        assert result.width == result.height


def test_make_square_shifts_instead_of_shrinking_near_image_edge():
    # A square centered near the top-left corner would spill outside the
    # image bounds - it must be shifted inward, not shrunk (still square).
    rect = RoiRect(x0=0, y0=0, x1=40, y1=30)
    result = make_square(rect, image_width=100, image_height=100)
    assert result.width == result.height == 40
    assert result.x0 >= 0 and result.y0 >= 0
    assert result.x1 <= 100 and result.y1 <= 100


def test_make_square_caps_size_to_image_bounds():
    rect = RoiRect(x0=0, y0=0, x1=150, y1=20)
    result = make_square(rect, image_width=100, image_height=50)
    assert result.width == result.height == 50
    assert result.x0 >= 0 and result.x1 <= 100
    assert result.y0 >= 0 and result.y1 <= 50


def test_make_square_zero_size_rect_stays_zero_size():
    rect = RoiRect(5, 5, 5, 5)
    result = make_square(rect, image_width=100, image_height=100)
    assert result.width == 0
    assert result.height == 0
