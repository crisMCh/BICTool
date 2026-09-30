"""Tests `mtf_seed.py` on synthetic ROIs - no patient data."""

import numpy as np
import pytest

import bictool.domain.metrics.mtf_seed as mtf_seed
from bictool.domain.metrics.mtf_seed import calc_mtf_first_viable_seed, mtf_seed_candidates


def _vertical_line_roi(size=26, col=14, width=3, background=30.0, line=80.0):
    roi = np.full((size, size), background)
    roi[:, col - width // 2 : col + width // 2 + 1] = line
    return roi


def test_first_seed_is_on_the_bright_lines_centerline():
    x, y = mtf_seed_candidates(_vertical_line_roi(col=14))[0]

    assert x == 14
    assert abs(y - 12.5) <= 26 / 4  # near the middle of the line's length


def test_later_seeds_are_separated_and_the_legacy_quarter_seed_comes_last():
    seeds = mtf_seed_candidates(_vertical_line_roi(size=40, col=20))

    for i, (x, y) in enumerate(seeds):
        for x2, y2 in seeds[i + 1 :]:
            assert np.hypot(x - x2, y - y2) >= 3
    assert seeds[-1] == (10, 10)  # the Octave-era default for a 40 px ROI


def test_structure_mask_restricts_seeds_to_the_structure():
    roi = _vertical_line_roi(col=8)
    roi[:, 18] = 200.0  # brighter distractor (e.g. a calcification streak)
    mask = np.zeros(roi.shape, dtype=bool)
    mask[:, 7:10] = True

    seeds = mtf_seed_candidates(roi, mask)

    assert all(7 <= x <= 9 for x, _y in seeds[:-1])  # all but the legacy fallback


def test_a_mask_without_pixels_in_the_center_falls_back_to_the_image():
    roi = _vertical_line_roi(col=13)
    mask = np.zeros(roi.shape, dtype=bool)
    mask[0:2, 0:2] = True  # only at the border

    assert mtf_seed_candidates(roi, mask)[0][0] == 13


def test_seeds_stay_inside_calc_mtfs_valid_range():
    roi = np.zeros((18, 18))
    roi[:, 17] = 100.0  # bright only at the border

    for x, y in mtf_seed_candidates(roi):
        assert 2 <= x <= 16 and 2 <= y <= 16


def test_first_viable_seed_retries_and_reports_the_seed_used(monkeypatch):
    def fake_calc_mtf(img_roi, x, y, pix_spc):
        if x < 5:
            raise ValueError("No edge columns survived bump removal")
        return "xaxis", "mtf", "area"

    monkeypatch.setattr(mtf_seed, "calc_mtf", fake_calc_mtf)

    assert calc_mtf_first_viable_seed(None, [(2, 2), (6, 6)], 0.5) == ("xaxis", "mtf", "area", (6, 6))


def test_first_viable_seed_raises_the_last_error_when_every_seed_fails(monkeypatch):
    def fake_calc_mtf(img_roi, x, y, pix_spc):
        raise ValueError(f"failed at {x}")

    monkeypatch.setattr(mtf_seed, "calc_mtf", fake_calc_mtf)

    with pytest.raises(ValueError, match="failed at 6"):
        calc_mtf_first_viable_seed(None, [(2, 2), (6, 6)], 0.5)


def test_real_calc_mtf_finds_edges_from_a_centerline_seed():
    rng = np.random.default_rng(0)
    roi = _vertical_line_roi(size=26, col=13, width=3) + rng.normal(0, 2, (26, 26))

    _xaxis, mtf, _area, seed = calc_mtf_first_viable_seed(roi, mtf_seed_candidates(roi), 0.44)

    assert mtf.shape[1] > 0
    assert seed[0] == 13
