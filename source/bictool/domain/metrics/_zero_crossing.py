"""Faithful translation of `reference/octave/funcZeroCrossing.m`, restricted
to the single call pattern actually used by the MTF pipeline:
`idxZero = funcZeroCrossing(LSF(:,idxCol))` - i.e. only the first return
value (`ind`), default time vector, default level=0, default linear method
requested. `t0`/`s0` (the interpolated crossing positions) are never used
by the caller and are not translated.
"""

import numpy as np


def zero_crossing(signal):
    """Return the 1-based sample indices (matching Octave's `find` output)
    at which `signal` is exactly zero or crosses zero between two samples.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    ind0 = np.flatnonzero(s == 0) + 1
    s1 = s[:-1] * s[1:]
    ind1 = np.flatnonzero(s1 < 0) + 1
    return np.sort(np.concatenate([ind0, ind1]))
