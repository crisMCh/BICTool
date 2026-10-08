"""Tests the physical QA recipe (`physical_spectra.py`) against analytical
answers - synthetic data only."""

import numpy as np
import pytest

from bictool.domain.metrics.physical_spectra import (
    FEATURE_BAND_MM,
    mtf_area_in_band,
    mtf_sample_spacing_mm,
    physical_mtf,
    power_spectrum,
    ratio_ps_mtf_in_band,
)


@pytest.mark.parametrize("pix_spc", [0.316, 0.33, 0.44, 0.58])
def test_edge_profiles_are_about_5_mm_long_whatever_the_pixel_spacing(pix_spc):
    # 15 samples over the real ~5 mm profile (Passand 2022, Sec. 4.4.2) -
    # not the 10 mm / 15 = 0.667 mm the Octave frequency axis assumes.
    assert mtf_sample_spacing_mm(pix_spc) * 14 == pytest.approx(5.0, abs=0.12)


def _octave_style_mtf(lsf: np.ndarray) -> np.ndarray:
    """What calc_mtf returns for one LSF: |FFT| normalised by its first harmonic."""
    magnitude = np.abs(np.fft.fft(lsf))
    return (magnitude / magnitude[1])[:, None]


@pytest.mark.parametrize("pix_spc", [0.33, 0.44, 0.58])
def test_physical_mtf_reproduces_a_known_blur(pix_spc):
    # Gaussian LSF with sigma 0.6 mm, sampled like calc_mtf's padded profiles
    # -> true MTF exp(-2 pi^2 sigma^2 f^2), MTF50 = 0.442 1/mm.
    sigma = 0.6
    dx = mtf_sample_spacing_mm(pix_spc)
    x = (np.arange(45) - 22) * dx
    f, mtf = physical_mtf(_octave_style_mtf(np.exp(-(x**2) / (2 * sigma**2))), pix_spc)

    expected = np.exp(-2 * np.pi**2 * sigma**2 * f**2)
    assert mtf[0, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(mtf[:, 0], expected, atol=0.01)
    assert f[-1] <= 1 / (2 * dx) + 1e-9  # mirrored half dropped


def test_mtf_area_is_integrated_up_to_the_band_limit():
    f = np.linspace(0, 1.4, 141)
    mtf = np.ones((141, 2))
    np.testing.assert_allclose(mtf_area_in_band(f, mtf), FEATURE_BAND_MM[1])
    assert np.isnan(mtf_area_in_band(f[:50], mtf[:50])).all()  # axis ends before 0.8


@pytest.mark.parametrize("px", [0.33, 0.44, 0.58])
def test_power_spectrum_of_white_noise_integrates_to_its_variance(px):
    rng = np.random.default_rng(0)
    variances = [power_spectrum(rng.normal(0, 10, (32, 32)), px).var_nps for _ in range(200)]

    assert np.mean(variances) == pytest.approx(100.0, rel=0.05)


def test_power_spectrum_frequency_axis_is_the_dft_grid():
    result = power_spectrum(np.random.default_rng(1).normal(0, 1, (26, 26)), 0.5)

    np.testing.assert_allclose(np.diff(result.f_avg), 1 / (26 * 0.5))
    assert result.f_avg[0] == 0.0


def _features(px, ps_curve, mtf_curve):
    """Ratio features for analytical spectra sampled the way BICTool would at pixel spacing `px`."""
    n = int(round(11.0 / px))  # ROI1 is ~11 mm whatever the spacing
    f_ps = np.arange(n // 2 + 1) / (n * px)
    f_mtf = np.arange(23) / (45 * mtf_sample_spacing_mm(px))
    return ratio_ps_mtf_in_band(f_ps, ps_curve(f_ps), f_mtf, mtf_curve(f_mtf)[:, None])


def test_ratio_features_do_not_depend_on_the_pixel_spacing():
    ps_curve = lambda f: 50 * np.exp(-((f - 0.3) ** 2) / 0.05) + 5  # noqa: E731
    mtf_curve = lambda f: np.exp(-2 * np.pi**2 * 0.4**2 * f**2)  # noqa: E731

    results = [_features(px, ps_curve, mtf_curve) for px in (0.32, 0.44, 0.58)]

    for result in results:
        assert result.band == FEATURE_BAND_MM
        assert result.band_truncated is False
    for name in ("ratio_lf_hf", "slope_avg", "loc_max"):
        values = [getattr(result, name) for result in results]
        assert max(values) - min(values) <= 0.05 * abs(np.mean(values)) + 0.02, name


def test_identical_ps_and_mtf_shapes_give_a_flat_ratio():
    f = np.linspace(0, 1.2, 121)
    curve = np.exp(-f)
    result = ratio_ps_mtf_in_band(f, 7 * curve, f, curve[:, None])

    np.testing.assert_allclose(result.ratio_nps_mtf, 1.0)
    assert result.slope_avg == pytest.approx(0.0, abs=1e-9)
    assert result.ratio_lf_hf == pytest.approx(1.0, rel=0.01)


def test_band_is_cut_short_and_flagged_when_the_spectrum_ends_early():
    f_ps = np.linspace(0, 0.6, 30)  # pixel spacing ~0.83 mm
    f_mtf = np.linspace(0, 1.4, 50)
    result = ratio_ps_mtf_in_band(f_ps, np.ones(30), f_mtf, np.ones((50, 1)))

    assert result.band_truncated is True
    assert result.band == (FEATURE_BAND_MM[0], pytest.approx(0.6))


def test_band_that_cannot_be_covered_raises():
    with pytest.raises(ValueError, match="not covered"):
        ratio_ps_mtf_in_band(np.linspace(0, 0.05, 5), np.ones(5), np.linspace(0, 1, 10), np.ones((10, 1)))


def test_noise_texture_of_a_known_spectrum():
    from bictool.domain.metrics.physical_spectra import noise_texture

    f = np.arange(0, 1.2, 0.01)
    nps = np.exp(-((f - 0.3) ** 2) / (2 * 0.05**2))  # bump at 0.3 1/mm

    f_av, f_peak = noise_texture(f, nps)

    assert f_peak == pytest.approx(0.3, abs=0.011)
    assert f_av == pytest.approx(0.3, abs=0.01)


def test_noise_texture_ignores_power_above_the_band():
    from bictool.domain.metrics.physical_spectra import noise_texture

    f = np.arange(0, 1.6, 0.01)
    nps = np.where(f > 0.9, 100.0, 1.0)  # power only matters inside [0, 0.8]

    f_av, f_peak = noise_texture(f, nps)

    assert f_av == pytest.approx(0.4, abs=0.01)
    assert f_peak <= 0.8


def test_noise_texture_of_an_empty_spectrum_is_nan():
    from bictool.domain.metrics.physical_spectra import noise_texture

    assert np.isnan(noise_texture(np.linspace(0, 1, 11), np.zeros(11))[0])
