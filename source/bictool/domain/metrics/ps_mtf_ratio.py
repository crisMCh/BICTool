"""PS/MTF ratio analysis. Faithful translation of the numeric path of
`reference/octave/funcRatioNpsMtf.m` (plotting code is not translated).

`NPS_int_scale` (a normalized-but-unused rescaling of the interpolated PS
curve, referenced only by the plotting code) is computed in the original but
never contributes to `structNpsMtfRatioResults` - it is not translated here.
"""

from dataclasses import dataclass

import numpy as np

from bictool.domain.preprocessing._pixel_ops import interp1_linear, octave_colon, oround


@dataclass
class NpsMtfRatioResult:
    """Port of the `structNpsMtfRatioResults` structure."""

    ratio_lf_hf: float
    slope_avg: float
    loc_max: float
    ratio_nps_mtf: np.ndarray
    f_int: np.ndarray


def ratio_nps_mtf(f_nps, nps, f_mtf, mtf):
    """Port of `funcRatioNpsMtf(fNPS, NPS, fMTF, MTF, ...)`.

    Parameters
    ----------
    f_nps, nps : 1D arrays - PS/NPS curve (e.g. `NpsResult.f_avg` /
        `.nps_measured_radagv`)
    f_mtf, mtf : the raw `calc_mtf` outputs (`xaxis`, `MTF`) - `mtf` is the
        2D per-edge matrix; the DC component is dropped and the curve is
        averaged across edges here, matching the original.
    """
    f_mtf = np.asarray(f_mtf, dtype=np.float64)[1:]
    mtf_curve = np.asarray(mtf, dtype=np.float64)[1:, :].mean(axis=1)

    f_min = f_mtf[0]
    f_max = f_nps[-1]

    f_int = octave_colon(f_min, 0.01, f_max)
    mtf_int = interp1_linear(f_mtf, mtf_curve, f_int)
    nps_int = interp1_linear(f_nps, nps, f_int)

    ratio = nps_int / mtf_int

    fc = int(oround(len(ratio) / 2))
    area_lf = np.trapezoid(ratio[:fc])
    area_hf = np.trapezoid(ratio[fc:])
    ratio_lf_hf = float(area_lf / area_hf)

    slope_avg = float(np.mean(np.diff(ratio)))

    pos = int(np.argmax(ratio)) + 1
    loc_max = pos / len(ratio)

    return NpsMtfRatioResult(
        ratio_lf_hf=ratio_lf_hf,
        slope_avg=slope_avg,
        loc_max=loc_max,
        ratio_nps_mtf=ratio,
        f_int=f_int,
    )
