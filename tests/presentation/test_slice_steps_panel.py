"""Tests the Advanced panel (`views.slice_steps_panel`) with a real computation."""

import numpy as np
import pytest

from bictool.domain.metrics.mtf_seed import mtf_seed_candidates
from bictool.domain.results.study_quality import evaluate_slice_quality
from bictool.integration.model_files import default_chest_svm_model
from bictool.presentation.viewmodels.slice_steps import slice_steps_view
from bictool.presentation.views.slice_steps_panel import SliceStepsPanel, ramp_color


@pytest.fixture(scope="module")
def steps_view():
    rng = np.random.default_rng(3)
    roi1 = np.full((26, 26), 30.0)
    roi1[:, 12:15] = 80.0
    roi1 += rng.normal(0, 2, roi1.shape)
    roi2 = 30.0 + rng.normal(0, 3, (26, 26))
    result = evaluate_slice_quality(
        roi1, roi2, 0.44, svm_model=default_chest_svm_model(), mtf_seeds=mtf_seed_candidates(roi1), keep_steps=True
    )
    return slice_steps_view(result, 0.44)


def test_the_panel_shows_every_tab_and_draws_without_errors(steps_view):
    panel = SliceStepsPanel()
    panel.resize(900, 600)
    panel.show_steps(steps_view)

    titles = [panel.tabText(i) for i in range(panel.count()) if panel.isTabVisible(i)]
    assert titles == ["MTF steps", "MTF spread", "ROIs && 2D spectra", "PS/MTF && features"]  # "&&" shows "&"
    assert len(panel.mtf.chart.series()) >= len(steps_view.mtf.faint) + 1
    assert "finite MTF" in panel.edge_counts.text() and "LF/HF" in panel.ratio_features.text()
    for index in range(4):  # paint every tab once
        panel.setCurrentIndex(index)
        panel.grab()


def test_without_steps_the_panel_says_why():
    panel = SliceStepsPanel()
    panel.show_steps(None, "No intermediate steps for this slice.")
    visible = [i for i in range(panel.count()) if panel.isTabVisible(i)]
    assert len(visible) == 1 and panel.empty.text() == "No intermediate steps for this slice."


def test_the_colour_ramp_runs_dark_to_bright():
    assert ramp_color(0.0).lightness() < ramp_color(0.5).lightness() < ramp_color(1.0).lightness()
