"""Reads (and writes) every model/weights file BICTool uses, so the domain
layer stays free of file I/O - the domain receives the loaded model:

* the head-CT study model (`domain.results.head_ct_verdict.HeadCtModel`),
  plain-array `.npz`, written by `application/train_classifier.py fit-study`;
* the legacy chest-CT SVM (`domain.results.classification`), plain-array `.npz`;
* the U-Net weights for anatomy and stroke segmentation (user-supplied,
  never bundled), loaded into a model built by the domain module.

The shipped `.npz` models lie next to the domain modules that evaluate them
(`domain/results/`, package data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import bictool.domain.results as results_package
from bictool.domain.results.head_ct_verdict import HeadCtModel

DEFAULT_HEAD_CT_MODEL_PATH = Path(results_package.__file__).with_name("_head_ct_model.npz")
DEFAULT_CHEST_SVM_MODEL_PATH = Path(results_package.__file__).with_name("_svm_model.npz")
_cache: dict[Path, object] = {}


def save_head_ct_model(model: HeadCtModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model.to_arrays())


def load_head_ct_model(path: str | Path) -> HeadCtModel:
    with np.load(path) as data:
        return HeadCtModel.from_arrays({k: data[k] for k in data.files})


def default_head_ct_model() -> HeadCtModel | None:
    """The shipped model, read once (None if its file is missing or unreadable)."""
    path = DEFAULT_HEAD_CT_MODEL_PATH
    if path not in _cache:
        try:
            _cache[path] = load_head_ct_model(path)
        except (OSError, KeyError, ValueError):
            _cache[path] = None
    return _cache[path]


def load_svm_model(path: str | Path) -> dict[str, np.ndarray]:
    """An SVM in `domain.results.classification`'s format as a dict of arrays."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def default_chest_svm_model() -> dict[str, np.ndarray]:
    """The shipped legacy chest-CT SVM (Passand 2022), read once."""
    path = DEFAULT_CHEST_SVM_MODEL_PATH
    if path not in _cache:
        _cache[path] = load_svm_model(path)
    return _cache[path]


def _load_keras_weights(model, weights_path: str | Path, missing_message: str) -> None:
    """Raises `FileNotFoundError` with an actionable message rather than a
    bare Keras/h5py error - the weights are never bundled with BICTool (see
    `infrastructure.config`), so a missing path is an expected, user-
    actionable situation, not a bug."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(missing_message.format(path=path))
    model.load_weights(str(path))


def load_anatomy_weights(model, weights_path: str | Path) -> None:
    """Pretrained weights into a model from `head_ct_unet.build_model()`."""
    _load_keras_weights(
        model,
        weights_path,
        "Anatomy segmentation weights not found at {path}. This file is not bundled with "
        "BICTool - download it from https://github.com/jasonccai/HeadCTSegmentation and "
        "configure its path (see infrastructure.config).",
    )


def load_stroke_weights(model, weights_path: str | Path) -> None:
    """Pretrained weights into a model from `aisd_stroke_unet.build_model()`."""
    _load_keras_weights(
        model,
        weights_path,
        "Stroke segmentation weights not found at {path}. This file is not bundled with "
        "BICTool - obtain best_stage2.weights.h5 from the AISD-HeadCT-Transfer-Learning "
        "project and configure its path (see infrastructure.config).",
    )
