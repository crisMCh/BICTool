"""Tests `segmentation_cache.py` on synthetic masks - no patient data."""

from pathlib import Path

import numpy as np

from bictool.domain.segmentation.segmentation_confidence import SegmentationConfidence
from bictool.integration.segmentation_cache import cache_path, load_confidence, load_segmentation, save_segmentation


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


def test_the_cnn_confidence_is_stored_and_old_caches_have_none(tmp_path):
    masks = {0: {"A": np.ones((4, 5), dtype=bool)}, 1: {"A": np.zeros((4, 5), dtype=bool)}}
    confidence = {0: SegmentationConfidence(0.12, 0.05, 20), 1: SegmentationConfidence(0.4, 0.3, 0)}

    save_segmentation(tmp_path / "new.npz", masks, confidence)
    save_segmentation(tmp_path / "old.npz", masks)

    assert load_confidence(tmp_path / "new.npz") == confidence
    assert load_confidence(tmp_path / "old.npz") == {}
    assert sorted(load_segmentation(tmp_path / "new.npz")) == [0, 1]
