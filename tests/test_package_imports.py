"""Smoke test verifying the package and its layer subpackages import cleanly."""

import importlib

LAYER_MODULES = [
    "bictool",
    "bictool.presentation",
    "bictool.presentation.views",
    "bictool.presentation.viewmodels",
    "bictool.application",
    "bictool.domain",
    "bictool.domain.preprocessing",
    "bictool.domain.segmentation",
    "bictool.domain.metrics",
    "bictool.domain.results",
    "bictool.integration",
    "bictool.infrastructure",
]


def test_layers_import():
    for module_name in LAYER_MODULES:
        importlib.import_module(module_name)
