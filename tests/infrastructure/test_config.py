import json

from bictool.infrastructure.config import load_config


def test_missing_config_file_returns_empty_dict(tmp_path):
    config = load_config(str(tmp_path / "does_not_exist.json"))
    assert config == {}


def test_loads_a_real_config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"segmentation": {"anatomy_weights_path": "/models/weights.hdf5"}}))

    config = load_config(str(path))

    assert config["segmentation"]["anatomy_weights_path"] == "/models/weights.hdf5"


def test_env_var_overrides_the_anatomy_weights_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BICTOOL_HEADCT_WEIGHTS", "/env/weights.hdf5")

    config = load_config(str(tmp_path / "does_not_exist.json"))

    assert config["segmentation"]["anatomy_weights_path"] == "/env/weights.hdf5"


def test_env_var_overrides_the_stroke_weights_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BICTOOL_STROKE_WEIGHTS", "/env/best_stage2.weights.h5")

    config = load_config(str(tmp_path / "does_not_exist.json"))

    assert config["segmentation"]["stroke_weights_path"] == "/env/best_stage2.weights.h5"


def test_env_var_overrides_a_value_already_in_the_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"segmentation": {"anatomy_weights_path": "/models/weights.hdf5"}}))
    monkeypatch.setenv("BICTOOL_HEADCT_WEIGHTS", "/env/weights.hdf5")

    config = load_config(str(path))

    assert config["segmentation"]["anatomy_weights_path"] == "/env/weights.hdf5"


def test_no_env_vars_set_leaves_config_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("BICTOOL_HEADCT_WEIGHTS", raising=False)
    monkeypatch.delenv("BICTOOL_STROKE_WEIGHTS", raising=False)

    config = load_config(str(tmp_path / "does_not_exist.json"))

    assert config == {}
