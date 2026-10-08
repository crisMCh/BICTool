"""Tests `segmentation_cache.py` on synthetic masks - no patient data."""

from pathlib import Path

import numpy as np

from bictool.integration.segmentation_cache import cache_path, load_segmentation, save_segmentation


def test_segmentation_cache_round_trips_exclusive_masks(tmp_path):
    a = np.zeros((4, 5), dtype=bool)
    a[0, :] = True
    b = np.zeros((4, 5), dtype=bool)
    b[2, 1:3] = True
    masks = {0: {"A": a, "B": b}, 3: {"A": b, "B": np.zeros((4, 5), dtype=bool)}}

    save_segmentation(tmp_path / "sub" / "c.npz", masks)
    loaded = load_segmentation(tmp_path / "sub" / "c.npz")

    assert sorted(loaded) == [0, 3]
    np.testing.assert_array_equal(loaded[0]["A"], a)
    np.testing.assert_array_equal(loaded[0]["B"], b)
    np.testing.assert_array_equal(loaded[3]["A"], b)


def test_cache_path_differs_per_series():
    a = cache_path(Path("c"), "Scanner", "study", [Path("s/kor_1.dcm")])
    b = cache_path(Path("c"), "Scanner", "study", [Path("s/ax_1.dcm")])

    assert a != b
    assert a.parent == Path("c") and a.suffix == ".npz"
