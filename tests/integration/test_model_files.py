"""Tests `model_files.py` (head-CT model I/O) - no patient data."""

from types import SimpleNamespace

import numpy as np
import pytest

from bictool.domain.results.head_ct_verdict import CORE_FEATURES, HeadCtModel
from bictool.integration.model_files import (
    default_chest_svm_model,
    default_head_ct_model,
    load_anatomy_weights,
    load_head_ct_model,
    load_stroke_weights,
    save_head_ct_model,
)


def test_shipped_model_loads_and_is_consistent():
    model = default_head_ct_model()

    assert model is not None and model.version.startswith("head-ct-")
    assert model.feature_names == CORE_FEATURES
    assert model.boot_weights.shape == (len(model.boot_intercepts), len(CORE_FEATURES))
    assert np.all(model.good_low <= model.good_high)
    assert model.weights[model.feature_names.index("var_struct")] < 0  # noisier -> less likely acceptable


def test_a_model_survives_save_and_load(tmp_path):
    model = default_head_ct_model()

    save_head_ct_model(model, tmp_path / "sub" / "m.npz")
    again = load_head_ct_model(tmp_path / "sub" / "m.npz")

    assert isinstance(again, HeadCtModel) and again.version == model.version
    np.testing.assert_array_equal(again.boot_weights, model.boot_weights)


def test_shipped_chest_svm_has_the_classifier_format():
    model = default_chest_svm_model()

    assert {"SVs", "sv_coef", "gamma", "rho", "Label", "feature_min", "feature_max"} <= set(model)
    assert model["SVs"].shape[1] == 7 and len(model["sv_coef"]) == len(model["SVs"])
    assert default_chest_svm_model() is model  # read once


@pytest.mark.parametrize("load", [load_anatomy_weights, load_stroke_weights])
def test_missing_weights_file_raises_a_clear_error(tmp_path, load):
    loaded = []
    model = SimpleNamespace(load_weights=loaded.append)

    with pytest.raises(FileNotFoundError, match="not found .* not bundled with BICTool"):
        load(model, tmp_path / "missing.h5")
    assert loaded == []

    (tmp_path / "w.h5").write_bytes(b"")
    load(model, tmp_path / "w.h5")
    assert loaded == [str(tmp_path / "w.h5")]
