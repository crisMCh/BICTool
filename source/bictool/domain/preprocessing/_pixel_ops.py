"""Faithful Python equivalents of the specific Octave / Octave-Forge (``image``
package v2.14.0) primitives used by the original MTF pipeline
(``reference/octave/funcCalcMTFNew.m``).

Each function here was derived by reading the actual Octave-Forge source
(not MATLAB documentation), since Octave-Forge's implementations of these
functions differ from MATLAB's toolbox in exact algorithmic details.
"""

import numpy as np
import scipy.ndimage as ndi

INT16_MIN = -32768
INT16_MAX = 32767


def oround(x):
    """Octave/MATLAB `round()`: round-half-away-from-zero (numpy's
    `np.round` is round-half-to-even and would disagree on exact .5 values).
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def to_int16(x):
    """Equivalent of Octave's `int16(x)` cast: round-half-away-from-zero,
    then saturate to the int16 range.
    """
    return np.clip(oround(x), INT16_MIN, INT16_MAX).astype(np.int16)


def im2double_int16(img_int16):
    """`im2double` for int16 input: normalizes using the fixed class range,
    not the image's actual min/max.
    """
    return (img_int16.astype(np.float64) + 32768.0) / 65535.0


def double_to_int16(img_double):
    """Inverse of `im2double_int16`: Octave-Forge's `imcast(x, 'int16')`,
    which computes `int16(double(x * uint16(65535)) - 32768)`.

    Critically, this is NOT `round(x*65535 - 32768)`: the `x * uint16(65535)`
    step itself rounds (half-away-from-zero) and saturates to the uint16
    range *before* the `-32768` offset is subtracted, not after. The two
    orderings agree almost everywhere, but disagree exactly on values that
    land on a half-integer tie after scaling by 65535 (e.g. `1/6`), which -
    given this pipeline's narrow stretch windows and heavily-tied Kirsch
    threshold - are common enough to matter.
    """
    scaled = np.clip(oround(img_double * 65535.0), 0.0, 65535.0)
    return (scaled - 32768.0).astype(np.int16)


def interp2_octave(Z, n):
    """Equivalent of Octave's `interp2(Z, n)`: recursively apply a
    single-level bilinear "refine-once" operator `n` times, growing an
    N x M array to (2N-1) x (2M-1) each time - NOT a one-shot bilinear
    resample onto a 2**n-times-finer grid.
    """
    out = np.asarray(Z, dtype=np.float64)
    for _ in range(n):
        out = _interp2_refine_once(out)
    return out


def _interp2_refine_once(Z):
    rows, cols = Z.shape
    out = np.empty((2 * rows - 1, 2 * cols - 1), dtype=np.float64)
    out[0::2, 0::2] = Z
    out[0::2, 1::2] = (Z[:, :-1] + Z[:, 1:]) / 2.0
    out[1::2, 0::2] = (Z[:-1, :] + Z[1:, :]) / 2.0
    out[1::2, 1::2] = (Z[:-1, :-1] + Z[:-1, 1:] + Z[1:, :-1] + Z[1:, 1:]) / 4.0
    return out


def stretchlim(img_d, tol_low=0.01, tol_high=0.99):
    """Order-statistic-index stretch limits (Octave-Forge `stretchlim.m`),
    NOT a percentile interpolation. `img_d` must already be normalized to
    [0, 1] (e.g. via `im2double_int16`).
    """
    flat = np.sort(img_d.ravel())
    n = flat.size
    lo_idx = int(np.floor(tol_low * n))          # 0-based equivalent of Octave's 1-based floor(tol*n)+1
    hi_idx = int(np.ceil(tol_high * n)) - 1       # 0-based equivalent of Octave's 1-based ceil(tol*n)
    lo_idx = min(max(lo_idx, 0), n - 1)
    hi_idx = min(max(hi_idx, 0), n - 1)
    low = min(max(flat[lo_idx], 0.0), 1.0)
    high = min(max(flat[hi_idx], 0.0), 1.0)
    if low == high:
        low, high = 0.0, 1.0
    return low, high


def imadjust_int16(img_int16, tol_low=0.01, tol_high=0.99):
    """Equivalent of Octave-Forge `imadjust(img)` default call on an int16
    image: `imadjust(img, stretchlim(img, 0.01), [0;1], gamma=1)`.
    """
    img_d = im2double_int16(img_int16)
    low, high = stretchlim(img_d, tol_low, tol_high)
    adj = (img_d - low) / (high - low)
    adj = np.clip(adj, 0.0, 1.0)
    return double_to_int16(adj)


def imsmooth_gaussian_int16(img_int16, sigma=0.5):
    """Equivalent of Octave-Forge `imsmooth(img, "Gaussian")` (default
    sigma=0.5): symmetric padding, separable Gaussian convolution ("valid"),
    cast back to int16.
    """
    h = int(np.ceil(3 * sigma))
    taps = np.arange(-h, h + 1, dtype=np.float64)
    f = np.exp(-(taps ** 2) / (2.0 * sigma ** 2))
    f = f / f.sum()

    img_d = img_int16.astype(np.float64)
    padded = np.pad(img_d, h, mode="symmetric")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, f, mode="valid"), axis=1, arr=padded)
    out = np.apply_along_axis(lambda m: np.convolve(m, f, mode="valid"), axis=0, arr=tmp)
    return to_int16(out)


def fspecial_gaussian(hsize, sigma):
    """Equivalent of Octave-Forge `fspecial("gaussian", hsize, sigma)`
    (square, odd `hsize`).
    """
    half = (hsize - 1) // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    y, x = np.meshgrid(coords, coords, indexing="ij")
    dist2 = x ** 2 + y ** 2
    f = np.exp(-dist2 / (2.0 * sigma ** 2))
    return f / f.sum()


def imsharpen_int16(img_int16, radius=1.0, amount=0.8, threshold=0.0):
    """Equivalent of Octave-Forge `imsharpen(img)` defaults
    (Radius=1, Amount=0.8, Threshold=0): unsharp masking via a Gaussian
    kernel, correlation with 'replicate' boundary, cast back to int16.
    """
    hsize = int(np.ceil(max(4 * radius + 1, 3)))
    if hsize % 2 == 0:
        hsize += 1
    kernel = fspecial_gaussian(hsize, radius)

    sharp = im2double_int16(img_int16)
    filtered = ndi.correlate(sharp, kernel, mode="nearest")
    g = sharp - filtered
    if threshold > 0:
        g = np.where(np.abs(g) < threshold, 0.0, g)
    sharp = sharp + amount * g
    return double_to_int16(sharp)


_KIRSCH = np.array([[3.0, 3.0, 3.0], [3.0, 0.0, 3.0], [-5.0, -5.0, -5.0]])
_SOBEL_H = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


def _simple_thinning(strength):
    r, c = strength.shape
    left = np.hstack([np.zeros((r, 1)), strength[:, :-1]])
    right = np.hstack([strength[:, 1:], np.zeros((r, 1))])
    x = (strength > left) & (strength > right)
    up = np.vstack([np.zeros((1, c)), strength[:-1, :]])
    down = np.vstack([strength[1:, :], np.zeros((1, c))])
    y = (strength > up) & (strength > down)
    return x | y


def edge_kirsch(mask_double):
    """Equivalent of Octave-Forge `edge(im, "Kirsch")` (default
    direction="both", thinning=True). `mask_double` is expected to already
    be a double 0/1 array (so `im2double` inside `edge.m` is a no-op).

    `edge.m` normalizes the Kirsch kernel to fractional weights *before*
    correlating (`h1 ./= sum(abs(h1(:)))`, then `imfilter(im, h1,
    "replicate")`), not after. The `strength2` threshold has heavy value
    ties, so matching this exact order of operations (rather than
    correlating with the integer kernel and dividing once at the end, which
    is algebraically equivalent but not bit-identical) is required for
    parity with Octave.
    """
    kirsch = _KIRSCH / np.sum(np.abs(_KIRSCH))
    resp_h = ndi.correlate(mask_double, kirsch, mode="nearest")
    resp_v = ndi.correlate(mask_double, kirsch.T, mode="nearest")
    strength2 = resp_h ** 2 + resp_v ** 2
    thresh2 = 4.0 * np.mean(strength2)
    strength2 = np.where(strength2 <= thresh2, 0.0, strength2)
    return _simple_thinning(strength2)


def imgradient_sobel(img):
    """Equivalent of Octave-Forge `imgradient(img)` (default method
    "sobel"), built on `imgradientxy`'s true-convolution + zero-padding
    semantics (no `im2double` normalization - operates on raw values).
    """
    img_d = img.astype(np.float64)
    gx = ndi.convolve(img_d, _SOBEL_H.T, mode="constant", cval=0.0)
    gy = ndi.convolve(img_d, _SOBEL_H, mode="constant", cval=0.0)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    gdir = np.degrees(np.arctan2(-gy, gx))
    return gmag, gdir


def bwdist(bw):
    """Equivalent of Octave/MATLAB `bwdist(BW)`: Euclidean distance to the
    nearest nonzero (True) pixel of `bw`. `scipy.ndimage.distance_transform_edt`
    computes distance to the nearest *zero* element of its argument, so the
    input must be inverted.
    """
    return ndi.distance_transform_edt(~bw.astype(bool))


def octave_colon(start, step, stop):
    """Equivalent of Octave/MATLAB's scalar range operator `start:step:stop`.

    Element count is `floor((stop-start)/step)` (e.g. `0:1:8.5` has 9
    elements, ending at 8, not 9) - verified against real Octave. A small
    epsilon guards the floor against floating-point error nudging an
    intended-exact ratio (e.g. `42.0`) just under the integer.
    """
    n = int(np.floor((stop - start) / step + 1e-10))
    return start + step * np.arange(n + 1)


def interp1_linear(x, y, xi):
    """Equivalent of Octave/MATLAB `interp1(x, y, xi)`: linear interpolation
    with NaN outside the domain of `x` (the default `extrap` behaviour is
    *not* replicated, since the pipeline never requests it).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    yi = np.interp(xi, x, y)
    yi = np.where((xi < x[0]) | (xi > x[-1]), np.nan, yi)
    return yi
