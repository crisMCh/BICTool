"""Line-profile (Edge Scan Function) sampling perpendicular to detected
edges. Faithful translation of `reference/octave/funcLineProfileNew.m`
together with the actually-reachable computational path of
`reference/octave/improfileoct.m` for this call pattern:
    improfileoct(img, [x0 x1], [y0 y1], N)
with `img` a 2D (non-RGB) array, `N` given explicitly (so the
`getn`/degenerate-grid branches of the Octave source never execute), and
method fixed to 'nearest'. Plotting/debug-image code and the `edgePixels`
bookkeeping (used only for plotting in the original) are not translated.
"""

import numpy as np

from bictool.domain.preprocessing._pixel_ops import oround


def _improfile_nearest(img, x0, y0, x1, y1, n_points):
    """Equivalent of `improfileoct(img, [x0 x1], [y0 y1], n_points)` for a
    2D image and method='nearest'. Coordinates are 1-based pixel
    coordinates (Octave convention): x = column, y = row.
    """
    t = np.linspace(0.0, 1.0, n_points)
    xg = x0 + t * (x1 - x0)
    yg = y0 + t * (y1 - y0)

    nrows, ncols = img.shape
    col = np.clip(oround(xg), 1, ncols).astype(np.int64) - 1
    row = np.clip(oround(yg), 1, nrows).astype(np.int64) - 1
    return img[row, col].astype(np.float64)


def line_profile(myimg, x_edge, y_edge, gdir, vec_len, lp_points, imgxsize, imgysize, lines=None):
    """Port of `funcLineProfileNew(myimg,x,y,Gdir,vecLen,LPpoints,imgxsize,imgysize,...)`.

    `x_edge`, `y_edge` are 1-based pixel coordinates (as produced by the
    Octave-order `find` translation in `mtf.py`). `gdir` is the gradient
    direction map (degrees), indexed the same way as `myimg`.

    Returns
    -------
    ESF : 2D float64 array, shape (lp_points, n_used_edges) - one column
        per edge pixel that passed the boundary-margin check.

    `lines`, if given, receives one `(k, x0, y0, x1, y1)` per column: the
    edge's index into `x_edge` and the sampled line's 1-based end points
    (for display only - the numbers are unaffected).
    """
    columns = []

    for k in range(len(x_edge)):
        x = float(x_edge[k])
        y = float(y_edge[k])

        if not (x > vec_len and y > vec_len
                and x + vec_len < imgxsize and y + vec_len < imgysize):
            continue

        angle = gdir[int(y_edge[k]) - 1, int(x_edge[k]) - 1]
        gradvecx = np.cos(np.radians(angle))
        gradvecy = np.sin(np.radians(angle))
        norm = np.hypot(gradvecx, gradvecy)
        gradvecx /= norm
        gradvecy /= norm

        vec_start_x = oround(x - vec_len * gradvecx)
        vec_start_y = oround(y - (-1 * vec_len * gradvecy))
        vec_end_x = oround(x + vec_len * gradvecx)
        vec_end_y = oround(y + (-1 * vec_len * gradvecy))

        lp_tmp = _improfile_nearest(myimg, vec_start_x, vec_start_y, vec_end_x, vec_end_y, lp_points)
        if lines is not None:
            lines.append((k, float(vec_start_x), float(vec_start_y), float(vec_end_x), float(vec_end_y)))

        lp_len = len(lp_tmp)
        first_quarter = lp_tmp[: int(oround(0.25 * lp_len))]
        last_quarter = lp_tmp[int(oround(0.75 * lp_len)):]
        if first_quarter.mean() > last_quarter.mean():
            lp_tmp = np.flipud(lp_tmp)

        columns.append(lp_tmp)

    if not columns:
        return np.zeros((lp_points, 0), dtype=np.float64)
    return np.column_stack(columns)
