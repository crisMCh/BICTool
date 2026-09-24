"""Regression tests for the Octave/Octave-Forge numeric-compatibility shims.

Expected values here are taken either directly from Octave-Forge's own
`image` package test suite (im2int16.m), or from small synthetic inputs run
once through real Octave/Octave-Forge (no patient data involved) to freeze
known-correct outputs for two bugs found while validating this port against
the legacy `reference/octave/funcCalcMTFNew.m` pipeline:

- `double_to_int16` combined the `x*65535` scale and the `-32768` offset
  into one rounding step, but Octave-Forge's `imcast(x, 'int16')` rounds
  and saturates `x*65535` to the `uint16` range *first*, then subtracts
  32768 exactly. The two orderings disagree on values that land on a
  half-integer tie after scaling (e.g. `1/6`).
- `edge_kirsch` normalized the Kirsch kernel *after* correlating instead of
  before, matching Octave's `edge.m` (`h1 ./= sum(abs(h1(:)))` precedes
  `imfilter`).
"""

import numpy as np
import pytest

from bictool.domain.preprocessing._pixel_ops import (
    double_to_int16,
    edge_kirsch,
    interp1_linear,
    octave_colon,
)


def test_double_to_int16_matches_octave_forge_doctests():
    # bictool: im2int16.m %!assert vectors (Octave-Forge image-2.14.0 source).
    assert np.array_equal(double_to_int16(np.array([0.0, 0.5, 1.0])), np.array([-32768, 0, 32767], dtype=np.int16))
    assert np.array_equal(double_to_int16(np.array([-1.0, 0.0, 1.0, 2.0])), np.array([-32768, -32768, 32767, 32767], dtype=np.int16))
    assert np.array_equal(
        double_to_int16(np.array([0.0, 1.4 / 65535, 1.5 / 65535, 2 / 65535, 1.0])),
        np.array([-32768, -32767, -32766, -32766, 32767], dtype=np.int16),
    )


def test_double_to_int16_scale_then_offset_tie_break():
    # Regression for the exact tie Octave resolves differently from a naive
    # round(x*65535 - 32768): 1/6 * 65535 = 10922.5 rounds (half-away-from-
    # zero, within uint16) to 10923 *before* subtracting 32768, giving
    # -21845 - not round(-21845.5) = -21846.
    assert double_to_int16(np.array([1.0 / 6.0]))[0] == -21845


def test_edge_kirsch_matches_octave_on_synthetic_disc():
    # A small filled disc on a 12x12 grid (no patient data), run once
    # through real Octave's `edge(mask, "Kirsch")`.
    n = 12
    yy, xx = np.mgrid[1 : n + 1, 1 : n + 1].astype(np.float64)
    cx, cy, r = 6.3, 6.7, 3.6
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2).astype(np.float64)

    expected_bw = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    bw = edge_kirsch(mask)

    assert np.array_equal(bw, expected_bw)
    assert bw.sum() == 8


def test_octave_colon_matches_real_octave():
    # Octave uses floor((stop-start)/step), NOT round: `0:1:8.5` stops at 8,
    # it does not round up to include a 9th step reaching past 8.5.
    r = octave_colon(0, 1, 8.5)
    np.testing.assert_array_equal(r, np.arange(9))

    # A near-exact ratio (41.6666...) still floors correctly despite
    # floating-point noise in the division.
    r2 = octave_colon(1.0 / 3.0, 0.01, 0.75)
    assert len(r2) == 42
    assert r2[-1] == pytest.approx(0.743333333333333)


def test_interp1_linear_matches_octave_defaults():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])

    # Linear interpolation within the domain.
    np.testing.assert_allclose(interp1_linear(x, y, [0.5, 1.5]), [5.0, 15.0])

    # Octave's `interp1` default returns NaN outside the domain of `x`
    # (no extrapolation).
    result = interp1_linear(x, y, [-0.5, 2.5])
    assert np.all(np.isnan(result))
