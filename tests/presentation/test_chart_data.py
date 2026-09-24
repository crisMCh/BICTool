import numpy as np

from bictool.presentation.viewmodels.chart_data import mtf_display_curve


def test_mtf_display_curve_averages_across_edges():
    xaxis = np.array([0.0, 0.5, 1.0])
    mtf = np.array([[1.0, 1.0], [0.6, 0.5], [0.2, 0.1]])  # monotonically decreasing mean

    x, y = mtf_display_curve(xaxis, mtf)

    np.testing.assert_array_equal(x, xaxis)
    np.testing.assert_allclose(y, [1.0, 0.55, 0.15])


def test_mtf_display_curve_truncates_at_global_minimum():
    # Mirrored/V-shaped curve: falls to a minimum at index 2, then rises
    # again - only the first half (through the minimum) should be kept.
    xaxis = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    mtf = np.array([[1.0], [0.6], [0.1], [0.6], [1.0]])

    x, y = mtf_display_curve(xaxis, mtf)

    np.testing.assert_array_equal(x, xaxis[:3])
    np.testing.assert_allclose(y, [1.0, 0.6, 0.1])


def test_mtf_display_curve_ignores_nan_edges_when_averaging():
    xaxis = np.array([0.0, 0.5, 1.0])
    mtf = np.array([[1.0, np.nan], [0.5, 0.5], [0.1, np.nan]])

    x, y = mtf_display_curve(xaxis, mtf)

    np.testing.assert_allclose(y, [1.0, 0.5, 0.1])


def test_mtf_display_curve_handles_no_edges():
    xaxis = np.array([0.0, 0.5, 1.0])
    mtf = np.zeros((3, 0))

    x, y = mtf_display_curve(xaxis, mtf)

    assert x.size == 0
    assert y.size == 0


def test_mtf_display_curve_handles_all_nan():
    xaxis = np.array([0.0, 0.5, 1.0])
    mtf = np.full((3, 2), np.nan)

    x, y = mtf_display_curve(xaxis, mtf)

    assert x.size == 0
    assert y.size == 0
