"""SVM-based image-quality classification (predicted label / decision value
feeding into FR-04's quality score). Faithful translation of the
classification tail of `reference/octave/funcImageQualityOctave.m`:
min-max scaling of the 7-element feature vector against the training set's
per-feature range, followed by a binary RBF-kernel SVM decision.

The RBF decision-function formula (`decision = sum(sv_coef * exp(-gamma *
||x - SV||^2)) - rho`, predicted label = `Label[0]` if `decision >= 0` else
`Label[1]`) is libsvm's public, portable model format - the same formula
libsvm's C core computes in every language binding, not an Octave-specific
detail - so it is implemented directly here rather than shimmed.

The model is passed in (a dict of arrays, read by
`integration/model_files.py` - the domain does no file I/O). The shipped
chest-CT model (`_svm_model.npz`, next to this module) was extracted from
`reference/octave/svmModelOctave.mat` (support vectors, dual coefficients,
gamma, rho, class labels) and `reference/octave/OctaveFeatSet.mat` (the
training set's per-feature min/max, used for scaling). `model.SVs` was
verified to equal the min-max-scaled `featSet` rows at `model.sv_indices`
exactly (max abs diff 0), confirming the scaling convention.

Known issue found while validating this port: on this machine's Octave +
Octave-Forge `statistics` v1.6.6 build, `svmpredict` is broken for
single-row queries specifically (always returns `decision=0,
predicted_label=0`), even though its own bundled multi-row self-test
passes. `funcImageQualityOctave.m`'s only call site queries exactly one row
(one image's features) - so the *original* Octave tool's classification
step may itself be non-functional on that combination, silently predicting
"bad image" for everything. This was reported upstream, not fixed here.
Ground truth for this port was obtained by padding the query to two
identical rows, which recovers the correct value.
"""

from collections.abc import Mapping

import numpy as np


def scale_features(features, model: Mapping[str, np.ndarray]):
    """Min-max scale a raw 7-element feature vector against the training
    set's per-feature range, matching `funcImageQualityOctave.m`'s
    `(featSetTest - sv_train_min) / (sv_train_max - sv_train_min)`.
    """
    features = np.asarray(features, dtype=np.float64)
    return (features - model["feature_min"]) / (model["feature_max"] - model["feature_min"])


def classify_quality(features, model: Mapping[str, np.ndarray]):
    """Port of the `svmpredict(1, featSetTest, model)` classification step
    (libsvm RBF decision rule) for any model in this format (see
    `application.train_classifier.export_model`).

    Parameters
    ----------
    features : raw (unscaled) 7-element feature vector, in the order
        `[mtfArea, npsHomo, npsStruct, varStruct, ratioLfHf, slopeAvg,
        locMax]` used by `funcImageQualityOctave.m`.
    model : support vectors, dual coefficients, gamma, rho, labels and the
        training feature min/max, as arrays.

    Returns
    -------
    predicted_label : 1 ("good image") or 0 ("bad image")
    decision_value : signed distance from the separating hyperplane
    """
    x = scale_features(features, model)
    k = np.exp(-model["gamma"] * np.sum((model["SVs"] - x) ** 2, axis=1))
    decision = float(np.dot(model["sv_coef"], k) - model["rho"])
    label = int(model["Label"][0] if decision >= 0 else model["Label"][1])
    return label, decision
