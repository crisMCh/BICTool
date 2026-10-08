"""Noise Power Spectrum (NPS) / Power Spectrum (PS) computation.

Faithful translation of `reference/octave/funcCalcDigitalNps.m` and
`reference/octave/funcCalcNPScompleteROI.m`, restricted to the single call
pattern the live pipeline (`funcImageQualityOctave.m`) actually uses: a 2D
(non-stacked) ROI, `n=2`, `use_window=1`, `average_stack=0`. The general
N-dimensional/stacked-image machinery in the original (never exercised by
any caller) is not translated.

`funcCalcDigitalNps.m` computes a detrending matrix `S` and immediately
discards it (the line `F=(I-S).*h` is commented out in favour of `F=(I).*h`,
with the Octave comment "estefade nemishe" - Persian for "not used") - `S`
is dead code and is not translated here.

`radialAverage.m` similarly computes a distance-weighted `values` array and
then immediately overwrites it with the unweighted `values = IMG(mask)` -
the weighting is dead code and is not translated.

Kept unchanged as the "octave" QA recipe. Its frequency axes are not true
spatial frequency (and the PS/MTF range depends on the pixel spacing) - the
default "physical" recipe (`physical_spectra.py`) corrects that; see
docs/qa_method/physical_qa_recipe.md.
"""

from dataclasses import dataclass

import numpy as np

from bictool.domain.preprocessing._pixel_ops import octave_colon


def radial_average(img, cx, cy, radii):
    """Port of `radialAverage(IMG, cx, cy, w)`: mean pixel value in a
    thickness-2 annulus around each radius in `radii`, centered at
    `(cx, cy)` (1-based, Octave pixel-coordinate convention).
    """
    img = np.asarray(img, dtype=np.float64)
    a, b = img.shape
    x = np.arange(1, a + 1, dtype=np.float64) - cx
    y = np.arange(1, b + 1, dtype=np.float64) - cy
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)

    profile = np.empty(len(radii), dtype=np.float64)
    for k, i in enumerate(radii):
        mask = (i - 1 < r) & (r < i + 1)
        profile[k] = img[mask].mean()
    return profile


def calc_digital_nps(img, px, use_window=True):
    """Port of `funcCalcDigitalNps(I, 2, px, use_window, 0)`.

    Parameters
    ----------
    img : 2D array - square ROI (mean-subtracted by the caller, per the
        original pipeline's usage)
    px : pixel spacing [mm]
    use_window : apply the Hann apodization window (matches the pipeline's
        fixed `use_window=1`)

    Returns
    -------
    nps : 2D float64 array - the (fftshift'd) NPS
    f : 1D float64 array - frequency axis [1/mm], shared by both dimensions
    """
    img = np.asarray(img, dtype=np.float64)
    roi_size = img.shape[0]
    if img.shape[1] != roi_size:
        raise ValueError("ROI must be symmetric.")

    x = np.linspace(-roi_size / 2, roi_size / 2, roi_size)
    f = np.linspace(-0.5, 0.5, roi_size) / px
    r = np.sqrt(x[:, None] ** 2 + x[None, :] ** 2)

    if use_window:
        h = 0.5 * (1.0 + np.cos(np.pi * r / (roi_size / 2)))
        h[r > roi_size / 2] = 0.0
    else:
        h = np.ones_like(img)

    F = img * h
    F = np.fft.fftshift(np.fft.fft2(F))

    nps = np.abs(F) ** 2 / roi_size ** 2 * px ** 2 / (np.sum(h ** 2) / h.size)
    return nps, f


@dataclass
class NpsResult:
    """Port of the `NPSresult` structure returned by `funcCalcNPScompleteROI`."""

    nps_measured: np.ndarray
    f: np.ndarray
    nps_measured_radagv: np.ndarray
    f_avg: np.ndarray
    var_measured: float
    var_nps: float


def nps_complete_roi(img, px):
    """Port of `funcCalcNPScompleteROI(IMG, px, ...)`: NPS/PS for the
    complete ROI (no homogeneous-area selection).
    """
    img = np.asarray(img, dtype=np.float64)
    nps_pix = img - img.mean()
    roi_size = nps_pix.shape[0]

    # Octave's `var()` normalizes by N-1 (sample variance).
    var_measured = float(np.var(nps_pix, ddof=1))

    nps_measured, f = calc_digital_nps(nps_pix, px, use_window=True)
    var_nps = float(np.trapezoid(np.trapezoid(nps_measured, f, axis=0), f))

    radii = octave_colon(0, 1, roi_size / 2 - 1)
    nps_measured_radagv = radial_average(nps_measured, roi_size / 2, roi_size / 2, radii)
    f_avg = np.linspace(0, 1 / px / 2, len(nps_measured_radagv))

    return NpsResult(
        nps_measured=nps_measured,
        f=f,
        nps_measured_radagv=nps_measured_radagv,
        f_avg=f_avg,
        var_measured=var_measured,
        var_nps=var_nps,
    )
