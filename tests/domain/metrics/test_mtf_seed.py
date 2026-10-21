"""Tests `mtf_seed.py` on synthetic ROIs - no patient data."""

import numpy as np
import pytest

import bictool.domain.metrics.mtf_seed as mtf_seed
from types import SimpleNamespace

from bictool.domain.metrics.mtf_seed import (
    calc_mtf_first_viable_seed,
    calc_mtf_steps_first_viable_seed,
    mtf_seed_candidates,
    region_on_structure,
)


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
    def fake_calc_mtf(img_roi, x, y, pix_spc, **_kwargs):
        if x < 5:
            raise ValueError("No edge columns survived bump removal")
        return "xaxis", "mtf", "area"

    monkeypatch.setattr(mtf_seed, "calc_mtf", fake_calc_mtf)

    assert calc_mtf_first_viable_seed(None, [(2, 2), (6, 6)], 0.5) == ("xaxis", "mtf", "area", (6, 6))


def test_first_viable_seed_raises_the_last_error_when_every_seed_fails(monkeypatch):
    def fake_calc_mtf(img_roi, x, y, pix_spc, **_kwargs):
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


def _upsampled(mask):
    """A region on calc_mtf's 4x grid (4(n-1)+1 points) covering the original pixels of `mask`."""
    n = mask.shape[0]
    return np.kron(mask, np.ones((4, 4), bool))[2:2 + 4 * (n - 1) + 1, 2:2 + 4 * (n - 1) + 1]


def test_region_on_structure_compares_on_the_upsampled_grid():
    line = np.zeros((20, 20), bool)
    line[:, 9:11] = True
    beside = np.zeros((20, 20), bool)
    beside[:, 2:6] = True

    assert region_on_structure(_upsampled(line), line) == pytest.approx(1.0)
    assert region_on_structure(_upsampled(beside), line) == 0.0
    # a thin region along the line (one upsampled point wide) is still counted
    thin = np.zeros((77, 77), bool)
    thin[:, 39] = True
    assert region_on_structure(thin, line) == pytest.approx(1.0)


def test_a_seed_whose_region_is_beside_the_structure_is_rejected(monkeypatch):
    line = np.zeros((20, 20), bool)
    line[:, 9:11] = True
    beside = np.zeros((20, 20), bool)
    beside[:, 2:6] = True

    def fake_steps(img_roi, x, y, pix_spc, **_kwargs):  # the seed beside the line grows a region beside it
        region = beside if x < 8 else line
        return "xaxis", "mtf", "area", SimpleNamespace(segmentation=_upsampled(region))

    monkeypatch.setattr(mtf_seed, "calc_mtf_with_steps", fake_steps)

    result = calc_mtf_first_viable_seed(None, [(4, 4), (10, 10)], 0.5, structure_mask=line)
    assert result == ("xaxis", "mtf", "area", (10, 10))
    with pytest.raises(ValueError, match="not on the structure"):
        calc_mtf_first_viable_seed(None, [(4, 4)], 0.5, structure_mask=line)


def test_without_a_known_structure_any_viable_seed_counts(monkeypatch):
    monkeypatch.setattr(mtf_seed, "calc_mtf", lambda img, x, y, px, **_k: ("xaxis", "mtf", "area"))

    assert calc_mtf_first_viable_seed(None, [(4, 4)], 0.5, structure_mask=np.zeros((20, 20), bool))[3] == (4, 4)
    assert calc_mtf_steps_first_viable_seed(None, [(4, 4)], 0.5, None, keep_steps=False)[4] is None
