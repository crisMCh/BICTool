"""CNN confidence from synthetic softmax outputs."""

import numpy as np
import pytest

from bictool.domain.segmentation.segmentation_confidence import confidence_from_probabilities


def _probs(rows):
    return np.array(rows, dtype=float)[np.newaxis]  # (1, n_pixels, n_classes)


def test_certain_pixels_have_zero_entropy_and_background_is_ignored():
    p = _probs([[0, 1, 0], [0, 1, 0], [1, 0, 0]])  # two certain structure pixels, one background

    c = confidence_from_probabilities(p)

    assert c.labelled_pixels == 2 and c.mean_entropy == pytest.approx(0.0, abs=1e-9) and c.low_fraction == 0.0


def test_unsure_pixels_raise_entropy_and_the_low_fraction():
    p = _probs([[0.1, 0.45, 0.45], [0.0, 0.9, 0.1]])

    c = confidence_from_probabilities(p)

    assert c.labelled_pixels == 2 and c.low_fraction == 0.5 and 0.2 < c.mean_entropy < 0.8


def test_a_slice_without_structures():
    assert confidence_from_probabilities(_probs([[1, 0, 0]])).labelled_pixels == 0
