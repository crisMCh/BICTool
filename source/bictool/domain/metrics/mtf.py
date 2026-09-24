"""Modulation Transfer Function (MTF) computation.

Faithful translation of `reference/octave/funcCalcMTFNew.m`'s numeric
pipeline. Plotting/debug-image code (`DEBUG_PLOT_LEVEL`), the
MATLAB-vs-Octave (`ENV_TYPE`) branching, and file-path/server-host
(`HOST_TYPE`) handling are all dropped - they have no effect on the
returned numeric results. `edgePixels`/gradient-magnitude bookkeeping used
only by the original's (partly already-commented-out) plotting helpers is
likewise not translated.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from bictool.domain.preprocessing._pixel_ops import (
    imadjust_int16,
    imgradient_sobel,
    imsharpen_int16,
    imsmooth_gaussian_int16,
    interp2_octave,
    edge_kirsch,
    oround,
    to_int16,
)
from bictool.domain.segmentation.active_contour import region_seg
from bictool.domain.metrics._line_profile import line_profile
from bictool.domain.metrics._zero_crossing import zero_crossing

IMAGE_INTERP_FACTOR = 2
ACTIVE_CONTOUR_MAX_ITS = 1000
ACTIVE_CONTOUR_ALPHA = 0.2
VEC_LEN_MM = 5
LP_POINTS = 15
INTERP_BUMP_REMOVAL = 4


def calc_mtf(img_roi, act_cont_mask_x, act_cont_mask_y, pix_spc):
    """Port of `funcCalcMTFNew(myimg, ActContMask, PixSpc, ...)`.

    Parameters
    ----------
    img_roi : 2D array - the ROI-cropped, HU-corrected CT image
    act_cont_mask_x, act_cont_mask_y : active-contour seed location
        (before interpolation; matches `ActContMask.x` / `ActContMask.y`)
    pix_spc : pixel spacing [mm] of `img_roi`

    Returns
    -------
    xaxis : 1D float64 array - spatial frequency axis [1/mm]
    MTF : 2D float64 array, shape (n_freq, n_edges) - MTF per edge pixel
    MTFarea : 1D float64 array, shape (n_edges,) - area under each MTF
        curve up to the Nyquist frequency
    """
    # (1) Preprocessing (interpolation + enhancement chain used only to
    # improve edge detection - the original, unfiltered image is restored
    # before line-profile sampling).
    myimg = np.asarray(img_roi, dtype=np.float64)
    myimg = interp2_octave(myimg, IMAGE_INTERP_FACTOR)
    myimg = to_int16(myimg)
    myimgorig = myimg.copy()
    pix_spc = pix_spc / IMAGE_INTERP_FACTOR

    myimg = imadjust_int16(myimg)
    myimg = imadjust_int16(myimg)
    myimg = imsmooth_gaussian_int16(myimg)
    myimg = imsharpen_int16(myimg)

    imgysize, imgxsize = myimg.shape

    # (2) Edge detection using active contours
    mask_x = IMAGE_INTERP_FACTOR ** 2 * act_cont_mask_x
    mask_y = IMAGE_INTERP_FACTOR ** 2 * act_cont_mask_y
    win_width = IMAGE_INTERP_FACTOR * 2

    m = np.zeros(myimg.shape, dtype=np.float64)
    y0, y1 = mask_y - win_width, mask_y + win_width
    x0, x1 = mask_x - win_width, mask_x + win_width
    m[y0 - 1:y1, x0 - 1:x1] = 1.0

    seg = region_seg(myimg, m, ACTIVE_CONTOUR_MAX_ITS, ACTIVE_CONTOUR_ALPHA)

    myimgedge = edge_kirsch(seg.astype(np.float64))

    # Octave's `find` traverses column-major; replicate that ordering so
    # per-edge results line up 1:1 with an Octave run for validation.
    x_edge0, y_edge0 = np.nonzero(myimgedge.T)
    x_edge = x_edge0 + 1
    y_edge = y_edge0 + 1

    # (3) Gradient computation (on the enhanced image, still `myimg`)
    _gmag, gdir = imgradient_sobel(myimg)

    # (4) Line profile estimation - from here on, use the original,
    # unfiltered (but interpolated) image again.
    myimg = myimgorig
    vec_len = int(oround(VEC_LEN_MM / pix_spc))
    pix_spc_lp = (2 * VEC_LEN_MM) / LP_POINTS

    esf_bak = line_profile(myimg, x_edge, y_edge, gdir, vec_len, LP_POINTS, imgxsize, imgysize)

    # Upsample the ESF for the following bump removal
    n = esf_bak.shape[0]
    new_x = np.linspace(0, n - 1, (n - 1) * INTERP_BUMP_REMOVAL + 1)
    esf = CubicSpline(np.arange(n), esf_bak, axis=0)(new_x)
    pix_spc_lp = pix_spc_lp / INTERP_BUMP_REMOVAL

    esf = esf - esf.mean(axis=0, keepdims=True)

    # Low-pass filter the ESF
    b, a = butter(3, 0.49, btype="low")
    esf_lp = filtfilt(b, a, esf, axis=0)
    esf = esf_lp

    # Bump removal
    esf_no_bump = _remove_bumps(esf)

    esf = esf_no_bump
    if esf.shape[1] == 0:
        raise ValueError("No edge columns survived bump removal")

    # Downsample the ESF (was upsampled for better bump removal)
    esf = esf[::INTERP_BUMP_REMOVAL]
    pix_spc_lp = pix_spc_lp * INTERP_BUMP_REMOVAL

    # Extend the ESF to the left and right (edge-value padding) to
    # generate more data points
    esf_len = esf.shape[0]
    left_pad = np.tile(esf[0:1, :], (esf_len, 1))
    right_pad = np.tile(esf[-1:, :], (esf_len, 1))
    esf = np.vstack([left_pad, esf, right_pad])

    lsf = np.diff(esf, axis=0)

    # (7) MTF
    MTF = np.abs(np.fft.fft(lsf, axis=0))
    MTF = MTF / MTF[1, :]

    n_freq = MTF.shape[0]
    xaxis = np.arange(n_freq) / (n_freq - 1) / pix_spc_lp

    MTFarea = _mtf_area(MTF, xaxis)

    return xaxis, MTF, MTFarea


def _remove_bumps(esf):
    n_samples, n_cols = esf.shape
    idx_peak_left = int(oround(0.40 * n_samples))
    idx_peak_right = int(oround(0.60 * n_samples))
    edge_center_distance = int(oround(0.35 * n_samples))

    lsf = np.diff(esf, axis=0)

    columns = []
    idx_left = None
    idx_right = None

    for idx_col in range(n_cols):
        lsf_col = lsf[:, idx_col]
        idx_zero = zero_crossing(lsf_col)

        window = np.abs(lsf_col[idx_peak_left - 1:idx_peak_right])
        idx_center = int(np.argmax(window)) + idx_peak_left

        left_candidates = idx_zero[idx_zero < idx_center]
        if left_candidates.size > 0:
            idx_left = int(left_candidates[-1]) + 1

        right_candidates = idx_zero[idx_zero > idx_center]
        if right_candidates.size > 0:
            idx_right = int(right_candidates[0]) + 1

        if idx_left is None or idx_right is None:
            continue

        if (idx_center - idx_left < edge_center_distance
                and idx_right - idx_center < edge_center_distance):
            tmp = esf[:, idx_col].copy()
            tmp[:idx_left] = tmp[idx_left - 1]
            tmp[idx_right - 1:] = tmp[idx_right - 1]
            columns.append(tmp)

    if not columns:
        return np.zeros((n_samples, 0), dtype=np.float64)
    return np.column_stack(columns)


def _mtf_area(MTF, xaxis):
    n_freq = MTF.shape[0]
    elem_num = (n_freq - 1) // 2 - 1

    n_cols = MTF.shape[1]
    areas = np.full(n_cols, np.nan, dtype=np.float64)
    if elem_num - 1 < 2:
        return areas

    valid_x = xaxis[1:elem_num]
    for k in range(n_cols):
        valid_data = MTF[1:elem_num, k]
        valid = ~np.isnan(valid_data)
        if valid.sum() > 1:
            areas[k] = np.trapezoid(valid_data[valid], valid_x[valid])
    return areas
