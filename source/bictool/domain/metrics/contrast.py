"""RMS contrast. Faithful translation of `reference/octave/funcCalcContrast.m`."""

import numpy as np


def calc_contrast(img):
    """Port of `funcCalcContrast(IMG)`: RMS contrast of the whole image/ROI."""
    img = np.asarray(img, dtype=np.float64)
    img_mean = img.mean()
    x = (img - img_mean) / img_mean
    return float(np.sqrt(np.mean(x ** 2)))
