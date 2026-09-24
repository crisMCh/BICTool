"""Pure data-shaping helpers for the Results page's charts.

Kept separate from the View so it is testable without Qt, matching
`roi_selection.py`/`windowing.py`'s split.
"""

from __future__ import annotations

import warnings

import numpy as np


def mtf_display_curve(mtf_xaxis: np.ndarray, mtf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Averages `calc_mtf`'s per-edge MTF matrix across edges, then
    truncates the result to its first half.

    `calc_mtf`'s frequency axis spans the full FFT output, so a real-valued
    line profile's magnitude spectrum mirrors around the Nyquist frequency:
    the curve falls to a minimum and then rises again as a reflection of
    the first half, which isn't a second, meaningful measurement - it's the
    same information plotted twice. Truncating at the curve's global
    minimum keeps only the physically meaningful first half.

    Returns `(x, y)`, both empty if `mtf` has no edge columns or is entirely
    NaN.
    """
    if mtf.shape[1] == 0:
        return np.array([]), np.array([])

    # A frequency row where every edge is NaN is an expected, handled case
    # (returns an empty curve below), not a real warning-worthy problem -
    # suppress numpy's "Mean of empty slice" for it specifically.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_curve = np.nanmean(mtf, axis=1)
    if np.all(np.isnan(mean_curve)):
        return np.array([]), np.array([])

    min_index = int(np.nanargmin(mean_curve))
    return mtf_xaxis[: min_index + 1], mean_curve[: min_index + 1]
