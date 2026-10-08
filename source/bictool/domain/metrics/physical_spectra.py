"""Physically scaled MTF / PS / NPS and the PS/MTF ratio features - the
"physical" QA recipe.

The Octave recipe (`mtf.calc_mtf`, `nps.nps_complete_roi`,
`ps_mtf_ratio.ratio_nps_mtf`, faithful ports of the original code behind
Passand 2022) has frequency axes that are not in true spatial frequency and
a PS/MTF evaluation range that depends on the pixel spacing. That was
invisible on the thorax CT it was developed on (pixel spacing 0.64-0.89 mm)
but breaks on head CT (0.3-0.6 mm): features stop being comparable between
scanners and become undefined below ~0.33 mm. Full derivation, evidence and
references: `docs/qa_method/physical_qa_recipe.md`.

This module re-expresses the same measurements on true frequency axes and
evaluates the ratio on one fixed frequency band:

* MTF: `calc_mtf`'s per-edge |FFT(LSF)| is kept as is (same edges, same
  profiles), but its frequency axis uses the profiles' real sample spacing
  (`mtf_sample_spacing_mm`), it is normalised by its DC value (AAPM TG-233)
  instead of the first harmonic, and only the part up to the profile
  Nyquist frequency is kept (the rest is the FFT's mirror image).
* PS / NPS: same windowed periodogram as the Octave port, on the exact DFT
  frequency grid 1/(N*px), radially binned around the true DC position.
* PS/MTF ratio (Passand 2022, Sec. 4.6, Eq. 4.6): both curves normalised to
  1 at the band's lowest frequency, evaluated on `FEATURE_BAND_MM` for every
  scan. LF/HF splits the band at its midpoint, slope is the mean derivative
  [mm], max location is a frequency [1/mm].
* MTF area: integral of the MTF over [0, FEATURE_BAND_MM[1]] [1/mm].

Pure NumPy - no Qt, no file I/O.
"""

from __future__ import annotations

import numpy as np

from bictool.domain.metrics.mtf import IMAGE_INTERP_FACTOR, LP_POINTS, VEC_LEN_MM
from bictool.domain.metrics.nps import NpsResult
from bictool.domain.metrics.ps_mtf_ratio import NpsMtfRatioResult
from bictool.domain.preprocessing._pixel_ops import oround

# Fixed evaluation band [1/mm] for every scan (docs/qa_method/
# physical_qa_recipe.md, Sec. 5.4). Upper limit: the image Nyquist frequency
# 1/(2*px) for any pixel spacing up to 0.625 mm (512^2 at <= 320 mm field of
# view - every routine head CT protocol), and where a standard adult head
# protocol's content ends: MTF 0.1 at 0.62 1/mm, NPS 0 by 0.8 1/mm
# (Friedman et al. 2013, Med Phys 40:051907) - so PS/MTF never divides by ~0.
# Lower limit: the first non-DC PS bin of a ~11 mm ROI (1/(N*px) ~ 0.09).
FEATURE_BAND_MM = (0.1, 0.8)
RATIO_STEP_MM = 0.01


def mtf_sample_spacing_mm(pix_spc: float) -> float:
    """True spacing [mm] of the samples along `calc_mtf`'s edge profiles.

    `calc_mtf` upsamples the ROI with `interp2_octave(., IMAGE_INTERP_FACTOR)`,
    i.e. IMAGE_INTERP_FACTOR refine passes = 2**IMAGE_INTERP_FACTOR (4x), but
    divides the pixel spacing only by IMAGE_INTERP_FACTOR (2) when converting
    VEC_LEN_MM to pixels. Each profile therefore spans 2*vec_len upsampled
    pixels of px/4 (~5 mm, the length Passand 2022 reports), sampled at
    LP_POINTS points - not the 10 mm / LP_POINTS = 0.667 mm `calc_mtf`
    assumes for its frequency axis.
    """
    vec_len = int(oround(VEC_LEN_MM / (pix_spc / IMAGE_INTERP_FACTOR)))
    upsampled_px = pix_spc / (2 ** IMAGE_INTERP_FACTOR)
    return 2 * vec_len * upsampled_px / (LP_POINTS - 1)


def physical_mtf(mtf: np.ndarray, pix_spc: float) -> tuple[np.ndarray, np.ndarray]:
    """`calc_mtf`'s MTF matrix (n_freq x n_edges) on a true frequency axis.

    Returns `(f, mtf)`: f = k / (n * dx) [1/mm] up to the profile Nyquist
    frequency 1/(2*dx), each edge's MTF normalised to 1 at DC.
    """
    mtf = np.asarray(mtf, dtype=np.float64)
    n = mtf.shape[0]
    dx = mtf_sample_spacing_mm(pix_spc)
    f = np.arange(n) / (n * dx)
    keep = f <= 1.0 / (2.0 * dx)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = mtf / mtf[0:1, :]
    return f[keep], normalised[keep]


def mtf_area_in_band(f: np.ndarray, mtf: np.ndarray, f_max: float = FEATURE_BAND_MM[1]) -> np.ndarray:
    """Per-edge area under the MTF from 0 to `f_max` [1/mm] (NaN if the MTF
    axis doesn't reach `f_max`)."""
    if f[-1] < f_max:
        return np.full(mtf.shape[1], np.nan)
    grid = np.arange(0.0, f_max + RATIO_STEP_MM / 2, RATIO_STEP_MM)
    return np.array([np.trapezoid(np.interp(grid, f, mtf[:, k]), grid) for k in range(mtf.shape[1])])


def power_spectrum(img: np.ndarray, px: float) -> NpsResult:
    """NPS/PS of a square ROI - the Octave port's estimator (mean removed,
    radial Hann window, |FFT|^2 * px^2 / (N^2 * mean(window^2))), on the exact
    DFT frequency grid and radially binned around the true DC position with
    1-bin-wide annuli. `var_nps` is the 2D integral (= the ROI variance for
    white noise)."""
    img = np.asarray(img, dtype=np.float64)
    n = img.shape[0]
    if img.shape[1] != n:
        raise ValueError("ROI must be symmetric.")
    centred = img - img.mean()
    var_measured = float(np.var(centred, ddof=1))

    x = np.linspace(-n / 2, n / 2, n)
    r_window = np.sqrt(x[:, None] ** 2 + x[None, :] ** 2)
    window = 0.5 * (1.0 + np.cos(np.pi * r_window / (n / 2)))
    window[r_window > n / 2] = 0.0
    spectrum = np.fft.fftshift(np.fft.fft2(centred * window))
    nps_2d = np.abs(spectrum) ** 2 / n**2 * px**2 / np.mean(window**2)

    f = np.fft.fftshift(np.fft.fftfreq(n, d=px))
    df = 1.0 / (n * px)
    var_nps = float(nps_2d.sum() * df**2)

    # Radial bins of width df around DC (fftshift puts DC at index n // 2),
    # up to and including the Nyquist frequency 1/(2*px).
    k = np.arange(n) - n // 2
    radius = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    bins = np.round(radius).astype(int)
    radii = np.arange(n // 2 + 1)
    profile = np.array([nps_2d[bins == r].mean() for r in radii])
    return NpsResult(
        nps_measured=nps_2d,
        f=f,
        nps_measured_radagv=profile,
        f_avg=radii * df,
        var_measured=var_measured,
        var_nps=var_nps,
    )


def ratio_ps_mtf_in_band(
    f_ps: np.ndarray,
    ps: np.ndarray,
    f_mtf: np.ndarray,
    mtf: np.ndarray,
    band: tuple[float, float] = FEATURE_BAND_MM,
) -> NpsMtfRatioResult:
    """PS/MTF ratio features on a fixed frequency band (`mtf` is the per-edge
    matrix from `physical_mtf`, averaged here).

    If a curve doesn't cover the whole band (pixel spacing > 0.625 mm), the
    band's upper end is lowered to what both cover and `band_truncated` is
    set - the features are then not directly comparable with full-band ones.
    """
    curve = np.nanmean(np.asarray(mtf, dtype=np.float64), axis=1)
    f_lo, f_hi = band
    f_top = min(f_hi, float(f_ps[-1]), float(f_mtf[-1]))
    truncated = f_top < f_hi
    if f_top <= f_lo:
        raise ValueError(f"PS/MTF band {band} not covered (spectra end at {f_top:.3f} 1/mm)")
    f = np.arange(f_lo, f_top + RATIO_STEP_MM / 2, RATIO_STEP_MM)
    ps_band = np.interp(f, f_ps, ps)
    mtf_band = np.interp(f, f_mtf, curve)
    ratio = (ps_band / ps_band[0]) / (mtf_band / mtf_band[0])

    mid = len(f) // 2  # band midpoint (by index - robust to float rounding)
    ratio_lf_hf = float(np.trapezoid(ratio[: mid + 1], f[: mid + 1]) / np.trapezoid(ratio[mid:], f[mid:]))
    slope = float(np.mean(np.diff(ratio)) / RATIO_STEP_MM)
    loc_max = float(f[int(np.argmax(ratio))])
    return NpsMtfRatioResult(
        ratio_lf_hf=ratio_lf_hf,
        slope_avg=slope,
        loc_max=loc_max,
        ratio_nps_mtf=ratio,
        f_int=f,
        band=(f_lo, float(f_top)),
        band_truncated=bool(truncated),
    )


def noise_texture(f: np.ndarray, nps: np.ndarray, f_max: float = FEATURE_BAND_MM[1]) -> tuple[float, float]:
    """Noise-texture frequencies of a radial 1D NPS, AAPM TG-233 Eq. 4-5:
    average frequency f_A = integral(f * NPS) / integral(NPS) and peak
    frequency f_P = argmax NPS [1/mm], both over [0, f_max] so they are
    comparable between pixel spacings (same band logic as the ratio
    features). NaN when the spectrum carries no power there.
    """
    f = np.asarray(f, dtype=np.float64)
    nps = np.asarray(nps, dtype=np.float64)
    keep = f <= f_max + 1e-12
    f, nps = f[keep], nps[keep]
    total = np.trapezoid(nps, f) if len(f) > 1 else 0.0
    if not np.isfinite(total) or total <= 0:
        return float("nan"), float("nan")
    return float(np.trapezoid(f * nps, f) / total), float(f[int(np.argmax(nps))])
