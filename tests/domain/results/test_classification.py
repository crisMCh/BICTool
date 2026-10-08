"""Regression tests for `classify_quality`, verified against real Octave's
`svmpredict` on real rows from the shipped training set (`OctaveFeatSet.mat`,
via `_svm_model.npz`) and a hand-crafted feature vector.

Note: Octave's `svmpredict` (Octave-Forge `statistics` v1.6.6, this port's
reference environment) is broken for single-row queries - it always returns
`decision=0`. Ground truth here was obtained by padding each query to two
identical rows (which recovers the correct value; verified against the
package's own bundled multi-row self-test, which passes) - see
`classification.py`'s module docstring.
"""

import numpy as np
import pytest

from bictool.domain.results.classification import classify_quality, scale_features
from bictool.integration.model_files import default_chest_svm_model

MODEL = default_chest_svm_model()  # the shipped chest-CT SVM

# Raw (unscaled) feature vectors: rows 1 and 12 of the training set
# (`OctaveFeatSet.mat`), and a hand-crafted vector resembling a real
# computed slice.
CASES = [
    ([0.377, 1097.0, 3154.0, 4738.071991381964, 0.9026663908542882, 15.079793477694379, 0.6792452830188679],
     1.22795041701806, 1),
    ([0.415, 4354.0, 6888.0, 4586.405617112955, 1.082202367755502, 68.23230492826055, 0.49056603773584906],
     1.01759265252104, 1),
    ([0.342283233, 100, 68.14, 51.07, 15.4947, -1.4646, 0.108333],
     -0.262839652171095, 0),
]


@pytest.mark.parametrize("features, expected_decision, expected_label", CASES)
def test_classify_quality_matches_octave(features, expected_decision, expected_label):
    label, decision = classify_quality(features, MODEL)
    assert decision == pytest.approx(expected_decision, rel=1e-9)
    assert label == expected_label


def test_scale_features_uses_shipped_training_range():
    # Scaling the training set's own min should give exactly 0, and its own
    # max should give exactly 1, for every feature.
    np.testing.assert_allclose(scale_features(MODEL["feature_min"], MODEL), np.zeros(7), atol=1e-12)
    np.testing.assert_allclose(scale_features(MODEL["feature_max"], MODEL), np.ones(7), atol=1e-12)
