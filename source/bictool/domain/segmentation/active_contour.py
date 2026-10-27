"""Chan-Vese region-based active contour segmentation.

Faithful translation of `reference/octave/funcRegionSeg.m` (itself based on
Shawn Lankton's "Active Contours Without Edges" implementation). Plotting
(`showCurveAndPhi`) is intentionally not translated - it has no effect on
the returned segmentation mask.
"""

import numpy as np

from bictool.domain.preprocessing._pixel_ops import bwdist

_EPS = np.finfo(np.float64).eps


def region_seg(image, init_mask, max_its, alpha=0.2, *, octave_step=False):
    """Port of `funcRegionSeg(I, init_mask, max_its, alpha)`.

    Parameters
    ----------
    image : 2D array (grayscale; converted to float64)
    init_mask : 2D boolean/0-1 array, same shape as `image`
    max_its : number of iterations (no automatic convergence test, matching
        the Octave original)
    alpha : weight of the curvature (smoothing) term

    octave_step : reproduce the original step size `dt = 0.45 / max(dphidt)`
        (Octave / Lankton). It turns negative when every point of the narrow
        band wants to join the region - a seed fully inside a bright line -
        and the update then runs backwards: the region shrinks to nothing.
        Default (False): the usual CFL step `dt = 0.45 / max(|dphidt|)`
        (bug fix 2026-10-20, docs/qa_method/physical_qa_recipe.md 5.1);
        the legacy "octave" QA recipe passes True.

    Returns
    -------
    2D boolean array: final segmentation mask (True = foreground)
    """
    I = _im2graydouble(image)
    phi = _mask2phi(init_mask)

    for _ in range(max_its):
        idx = np.flatnonzero((phi <= 1.2) & (phi >= -1.2))
        if idx.size == 0:
            # Matches Octave: arithmetic on an empty array is a silent
            # no-op there, so an empty narrow band freezes `phi` for the
            # rest of the iterations instead of erroring.
            continue

        upts = phi.ravel() <= 0
        vpts = phi.ravel() > 0
        u = I.ravel()[upts].sum() / (upts.sum() + _EPS)
        v = I.ravel()[vpts].sum() / (vpts.sum() + _EPS)

        I_idx = I.ravel()[idx]
        F = (I_idx - u) ** 2 - (I_idx - v) ** 2
        curvature = _get_curvature(phi, idx)

        dphidt = F / np.max(np.abs(F)) + alpha * curvature
        dt = 0.45 / ((np.max(dphidt) if octave_step else np.max(np.abs(dphidt))) + _EPS)

        phi_flat = phi.ravel().copy()
        phi_flat[idx] = phi_flat[idx] + dt * dphidt
        phi = phi_flat.reshape(phi.shape)

        phi = _sussman(phi, 0.5)

    return phi <= 0


def _mask2phi(init_a):
    m = init_a.astype(bool)
    return bwdist(m) - bwdist(~m) + m.astype(np.float64) - 0.5


def _get_curvature(phi, idx):
    dimy, dimx = phi.shape
    y, x = np.unravel_index(idx, (dimy, dimx))

    ym1 = y - 1
    xm1 = x - 1
    yp1 = y + 1
    xp1 = x + 1

    ym1 = np.where(ym1 < 0, 0, ym1)
    xm1 = np.where(xm1 < 0, 0, xm1)
    yp1 = np.where(yp1 > dimy - 1, dimy - 1, yp1)
    xp1 = np.where(xp1 > dimx - 1, dimx - 1, xp1)

    idup = np.ravel_multi_index((yp1, x), (dimy, dimx))
    iddn = np.ravel_multi_index((ym1, x), (dimy, dimx))
    idlt = np.ravel_multi_index((y, xm1), (dimy, dimx))
    idrt = np.ravel_multi_index((y, xp1), (dimy, dimx))
    idul = np.ravel_multi_index((yp1, xm1), (dimy, dimx))
    idur = np.ravel_multi_index((yp1, xp1), (dimy, dimx))
    iddl = np.ravel_multi_index((ym1, xm1), (dimy, dimx))
    iddr = np.ravel_multi_index((ym1, xp1), (dimy, dimx))

    phi_flat = phi.ravel()
    phi_x = -phi_flat[idlt] + phi_flat[idrt]
    phi_y = -phi_flat[iddn] + phi_flat[idup]
    phi_xx = phi_flat[idlt] - 2 * phi_flat[idx] + phi_flat[idrt]
    phi_yy = phi_flat[iddn] - 2 * phi_flat[idx] + phi_flat[idup]
    phi_xy = (-0.25 * phi_flat[iddl] - 0.25 * phi_flat[idur]
              + 0.25 * phi_flat[iddr] + 0.25 * phi_flat[idul])
    phi_x2 = phi_x ** 2
    phi_y2 = phi_y ** 2

    return ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy)
             / (phi_x2 + phi_y2 + _EPS) ** 1.5) * (phi_x2 + phi_y2) ** 0.5


def _im2graydouble(img):
    return img.astype(np.float64)


def _sussman(D, dt):
    a = D - _shift_r(D)
    b = _shift_l(D) - D
    c = D - _shift_d(D)
    d = _shift_u(D) - D

    a_p = np.where(a < 0, 0.0, a)
    a_n = np.where(a > 0, 0.0, a)
    b_p = np.where(b < 0, 0.0, b)
    b_n = np.where(b > 0, 0.0, b)
    c_p = np.where(c < 0, 0.0, c)
    c_n = np.where(c > 0, 0.0, c)
    d_p = np.where(d < 0, 0.0, d)
    d_n = np.where(d > 0, 0.0, d)

    dD = np.zeros_like(D)
    pos = D > 0
    neg = D < 0
    dD[pos] = np.sqrt(np.maximum(a_p[pos] ** 2, b_n[pos] ** 2)
                       + np.maximum(c_p[pos] ** 2, d_n[pos] ** 2)) - 1
    dD[neg] = np.sqrt(np.maximum(a_n[neg] ** 2, b_p[neg] ** 2)
                       + np.maximum(c_n[neg] ** 2, d_p[neg] ** 2)) - 1

    return D - dt * _sussman_sign(D) * dD


def _shift_l(M):
    return np.hstack([M[:, 1:], M[:, -1:]])


def _shift_r(M):
    return np.hstack([M[:, :1], M[:, :-1]])


def _shift_u(M):
    return _shift_l(M.T).T


def _shift_d(M):
    return _shift_r(M.T).T


def _sussman_sign(D):
    return D / np.sqrt(D ** 2 + 1)
