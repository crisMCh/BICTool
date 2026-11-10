"""Fixtures for the API tests (the helpers are in `api_synthetic`)."""

from types import SimpleNamespace

import pytest
from api_synthetic import fake_quality, septum_slice

import bictool.application.pipeline as pipeline


@pytest.fixture
def fake_measurement(monkeypatch):
    """`evaluate_slice_quality` replaced by a fixed result; returns the calls."""
    calls = []
    monkeypatch.setattr(pipeline, "evaluate_slice_quality", lambda *a, **k: calls.append((a, k)) or fake_quality())
    return calls


@pytest.fixture
def fake_cnn(monkeypatch):
    """The CNN replaced: every slice gets the synthetic septum labels."""
    segmented = []
    monkeypatch.setattr(pipeline, "load_anatomy_model", lambda weights_path=None: "model")
    monkeypatch.setattr(pipeline, "prepare_segmentation", lambda volume: "nifti")

    def segment(model, nifti, index):
        segmented.append(index)
        return SimpleNamespace(masks=septum_slice(), confidence=None)

    monkeypatch.setattr(pipeline, "segment_slice", segment)
    return segmented
