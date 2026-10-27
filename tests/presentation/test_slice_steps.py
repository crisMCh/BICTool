"""Tests `viewmodels.slice_steps` on a real computation (synthetic falx-like ROI)."""

from types import SimpleNamespace

import numpy as np
import pytest

from bictool.domain.metrics.mtf_seed import mtf_seed_candidates
from bictool.domain.results.study_quality import evaluate_slice_quality
from bictool.integration.model_files import default_chest_svm_model
from bictool.presentation.viewmodels.slice_steps import (
    MAX_CURVES,
    ROLE_DROPPED,
    ROLE_EDGE,
    log_display,
    sample_columns,
    slice_steps_view,
    window,
)

SPACING = 0.44


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(2)
    roi1 = np.full((26, 26), 30.0)
    roi1[:, 12:15] = 80.0
    roi1 += rng.normal(0, 2, roi1.shape)
    roi2 = 30.0 + rng.normal(0, 3, (26, 26))
    return evaluate_slice_quality(
        roi1, roi2, SPACING, svm_model=default_chest_svm_model(), mtf_seeds=mtf_seed_candidates(roi1), keep_steps=True
    )


def test_without_kept_steps_there_is_nothing_to_show():
    assert slice_steps_view(SimpleNamespace(steps=None), SPACING) is None


def test_mtf_panels_line_up_with_the_result(result):
    view = slice_steps_view(result, SPACING)
    n_edges = result.mtf.shape[1]
    assert len(view.mtf.faint) == min(n_edges, MAX_CURVES)
    np.testing.assert_allclose(view.mtf.mean, np.nanmean(result.mtf, axis=1))
    assert view.esf.x.shape == view.esf.mean.shape and view.lsf.x.shape == view.lsf.mean.shape
    assert view.esf.x[0] < 0 < view.esf.x[-1]  # centred on the edge
    assert view.esf.mean[0] < view.esf.mean[-1]  # profiles run dark -> bright
    roles = dict((role, points) for points, role in view.roi1_edges.points)
    assert len(roles[ROLE_EDGE]) == n_edges and len(roles[ROLE_DROPPED]) >= n_edges
    assert view.roi1_edges.contour.shape[1] == 2 and len(view.roi1_edges.contour) > 0
    assert view.mtf.bands == ((0.0, 0.8, "band"),)


def test_spread_and_counts(result):
    view = slice_steps_view(result, SPACING)
    assert len(view.edge_map.coloured_points) == len(view.edge_map.point_values) == result.mtf.shape[1]
    assert view.area_histogram.counts.sum() == np.isfinite(result.mtf_area_per_edge).sum()
    assert view.area_histogram.marks == (pytest.approx(result.mtf_area),)
    assert view.edge_counts[-2].endswith(str(result.mtf.shape[1]))


def test_rois_spectra_and_ratio(result):
    view = slice_steps_view(result, SPACING)
    assert view.roi1.image.shape == (26, 26) and 0 <= view.roi1.image.min() and view.roi1.image.max() <= 1
    assert view.ps_2d.image.shape == result.nps_roi1.nps_measured.shape
    assert view.nps_2d.extent[0] < 0 < view.nps_2d.extent[1]
    (lo0, lo1, low), (hi0, hi1, high) = view.ratio.bands
    assert (lo0, hi1, low, high) == (pytest.approx(0.1), pytest.approx(0.8), "low", "high") and lo1 == hi0
    assert view.ratio.marks == (pytest.approx(result.ratio.loc_max),)
    assert view.ratio_features[0].startswith("LF/HF ratio")


def test_display_helpers():
    assert window(np.array([0.0, 40.0, 80.0, 200.0])).tolist() == [0.0, 0.5, 1.0, 1.0]
    assert log_display(np.ones((4, 4))).max() == 0.0  # flat spectrum: no contrast to show
    assert list(sample_columns(5, 10)) == [0, 1, 2, 3, 4] and len(sample_columns(100, 10)) == 10


def test_the_legacy_recipe_gets_the_same_panels_without_bands():
    rng = np.random.default_rng(4)
    roi1 = np.full((26, 26), 30.0)
    roi1[:, 12:15] = 80.0
    roi1 += rng.normal(0, 2, roi1.shape)
    roi2 = 30.0 + rng.normal(0, 3, (26, 26))
    legacy = evaluate_slice_quality(
        roi1, roi2, SPACING, svm_model=default_chest_svm_model(), mtf_seeds=mtf_seed_candidates(roi1),
        recipe="octave", keep_steps=True,
    )
    view = slice_steps_view(legacy, SPACING)
    assert view.mtf.bands == () and view.ratio.bands == () and view.ratio.marks == ()
    assert len(view.mtf.x) == len(legacy.mtf_xaxis) // 2  # mirrored second half left out
