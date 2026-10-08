"""Per-slice quality assessment: wires the individual metric modules
together into one call, capturing the computational recipe of
`reference/octave/funcImageQualityOctave.m` (which ROI feeds which metric,
how the 7-element SVM feature vector is assembled) without translating that
script itself.

`funcImageQualityOctave.m` also does a lot that does NOT belong here:
`DEBUG_PLOT_LEVEL`/`ENV_TYPE`/`HOST_TYPE` branching, plotting, writing
results to hardcoded per-patient folders, and reading/cropping the DICOM
ROI itself. None of that is translated - this function takes already-cropped
ROI arrays (matching the precedent `calc_mtf` already sets) and returns a
plain, serializable result; DICOM loading, ROI selection, and result export
are left to future Integration/Presentation-layer code driven by a GUI.

Feature-vector assembly (from the real Octave source):

    mtfArea   = mean(MTFarea)          # calc_mtf's per-edge MTFarea, averaged
    npsHomo   = imgROI2_NPS.var_NPS    # ROI2 (homogeneous) - NpsResult.var_nps
    npsStruct = imgROI1_NPS.var_NPS    # ROI1 (structured)  - NpsResult.var_nps
    varStruct = imgROI1_NPS.var_measured   # ROI1 - a DIFFERENT field, easy to
                                            # confuse with var_NPS/var_nps above
    ratioLfHf, slopeAvg, locMax        # from funcRatioNpsMtf(fNPS1, NPS1, fMTF, MTF)
                                        # i.e. ROI1's NPS against ROI1's raw MTF

`ContrastROI1`/`ContrastROI2` are computed and returned by the original for
display, but are NOT part of the classifier's feature vector.
"""

import logging
from dataclasses import dataclass

import numpy as np

from bictool.domain.metrics.contrast import calc_contrast
from bictool.domain.metrics.mtf_seed import calc_mtf_first_viable_seed
from bictool.domain.metrics.nps import NpsResult, nps_complete_roi
from bictool.domain.metrics.physical_spectra import (
    mtf_area_in_band,
    noise_texture,
    physical_mtf,
    power_spectrum,
    ratio_ps_mtf_in_band,
)
from bictool.domain.metrics.ps_mtf_ratio import NpsMtfRatioResult, ratio_nps_mtf
from bictool.domain.results.classification import classify_quality

logger = logging.getLogger(__name__)

# QA recipes. "physical" (default): MTF/PS/NPS on true spatial-frequency axes
# and the PS/MTF ratio on one fixed band, so features mean the same on every
# scanner (`physical_spectra`, docs/qa_method/physical_qa_recipe.md).
# "octave": the faithful port of the original Octave code (Passand 2022) -
# kept to reproduce Octave-era numbers; its frequency axes are mis-scaled and
# its ratio range depends on the pixel spacing.
RECIPE_PHYSICAL = "physical"
RECIPE_OCTAVE = "octave"
RECIPES = (RECIPE_PHYSICAL, RECIPE_OCTAVE)
DEFAULT_RECIPE = RECIPE_PHYSICAL

# `calc_mtf` edge points whose MTF isn't finite (e.g. at the rounded tip of a
# line's contour, where the "across the edge" profile runs along the line and
# the MTF normalization divides by zero) are dropped before anything uses the
# MTF. If more than this fraction of edges is dropped, the contour itself is
# poor and the slice fails instead.
MAX_DROPPED_MTF_EDGE_FRACTION = 0.5


@dataclass
class SliceQualityResult:
    """Every individual metric's result for one slice/ROI pair, plus the
    classification outcome. Plain dataclass of arrays/scalars/nested
    dataclasses - no plotting, no file I/O, serializable as-is.
    """

    mtf_xaxis: np.ndarray
    mtf: np.ndarray
    mtf_area_per_edge: np.ndarray
    mtf_area: float

    nps_roi1: NpsResult
    nps_roi2: NpsResult

    contrast_roi1: float
    contrast_roi2: float

    ratio: NpsMtfRatioResult

    feature_vector: np.ndarray
    predicted_label: int
    decision_value: float

    # The `calc_mtf` seed `(x, y)` that produced the MTF (see `mtf_seed`).
    mtf_seed: tuple[int, int] | None = None
    # Which QA recipe produced the numbers (see RECIPES).
    recipe: str = RECIPE_OCTAVE
    # Noise texture of the homogeneous ROI2 (TG-233 average / peak NPS
    # frequency [1/mm], `physical_spectra.noise_texture`) - physical recipe
    # only, NaN otherwise. Candidate classifier features.
    nps_f_av: float = float("nan")
    nps_f_peak: float = float("nan")


def evaluate_slice_quality(
    img_roi1,
    img_roi2,
    pix_spc,
    act_cont_mask_x=10,
    act_cont_mask_y=10,
    *,
    svm_model,
    mtf_seeds=None,
    recipe=DEFAULT_RECIPE,
):
    """Port of `funcImageQualityOctave.m`'s computational recipe (not the
    script itself - see the module docstring).

    Parameters
    ----------
    img_roi1 : the "structured" ROI (e.g. septum/falx) - used for both MTF
        and one of the two NPS/PS measurements.
    img_roi2 : the "homogeneous" ROI (e.g. ventricles/white matter) - used
        for the other NPS measurement.
    pix_spc : pixel spacing [mm] of both ROIs.
    act_cont_mask_x, act_cont_mask_y : active-contour seed location for
        `calc_mtf`; default to 10/10, the fixed value used for every real
        slice in `reference/octave/brain_calls.m`.
    mtf_seeds : optional sequence of `(x, y)` seeds tried in order until
        `calc_mtf` finds usable edges (see
        `bictool.domain.metrics.mtf_seed.mtf_seed_candidates`); replaces
        `act_cont_mask_x/y` when given.
    recipe : "physical" (default) or "octave" - see RECIPES.

    Returns
    -------
    SliceQualityResult
    """
    logger.debug(
        "evaluate_slice_quality: img_roi1 shape=%s, img_roi2 shape=%s, pix_spc=%s, act_cont_mask=(%s, %s)",
        img_roi1.shape,
        img_roi2.shape,
        pix_spc,
        act_cont_mask_x,
        act_cont_mask_y,
    )

    logger.debug("Step 1/6: calc_mtf(img_roi1, ...)")
    seeds = list(mtf_seeds) if mtf_seeds else [(act_cont_mask_x, act_cont_mask_y)]
    if recipe not in RECIPES:
        raise ValueError(f"Unknown QA recipe {recipe!r} - expected one of {RECIPES}")
    xaxis, mtf, mtf_area_per_edge, mtf_seed = calc_mtf_first_viable_seed(img_roi1, seeds, pix_spc)
    if recipe == RECIPE_PHYSICAL:
        xaxis, mtf = physical_mtf(mtf, pix_spc)
        mtf_area_per_edge = mtf_area_in_band(xaxis, mtf)
    mtf, mtf_area_per_edge = _drop_invalid_mtf_edges(mtf, mtf_area_per_edge)
    mtf_area = float(np.mean(mtf_area_per_edge))
    logger.debug(
        "Step 1/6 done: mtf_area=%s (n_edges=%d, seed=%s)", mtf_area, mtf.shape[1] if mtf.size else 0, mtf_seed
    )

    logger.debug("Step 2/6: nps_complete_roi(img_roi1, ...) [structured ROI]")
    spectrum = power_spectrum if recipe == RECIPE_PHYSICAL else nps_complete_roi
    nps_roi1 = spectrum(img_roi1, pix_spc)
    logger.debug("Step 2/6 done: nps_roi1.var_nps=%s, var_measured=%s", nps_roi1.var_nps, nps_roi1.var_measured)

    logger.debug("Step 3/6: nps_complete_roi(img_roi2, ...) [homogeneous ROI]")
    nps_roi2 = spectrum(img_roi2, pix_spc)
    logger.debug("Step 3/6 done: nps_roi2.var_nps=%s", nps_roi2.var_nps)

    logger.debug("Step 4/6: ratio_nps_mtf(nps_roi1.f_avg, nps_roi1.nps_measured_radagv, xaxis, mtf)")
    ratio_features = ratio_ps_mtf_in_band if recipe == RECIPE_PHYSICAL else ratio_nps_mtf
    ratio = ratio_features(nps_roi1.f_avg, nps_roi1.nps_measured_radagv, xaxis, mtf)
    logger.debug("Step 4/6 done: ratio_lf_hf=%s, slope_avg=%s, loc_max=%s", ratio.ratio_lf_hf, ratio.slope_avg, ratio.loc_max)

    logger.debug("Step 5/6: calc_contrast(img_roi1), calc_contrast(img_roi2)")
    contrast_roi1 = calc_contrast(img_roi1)
    contrast_roi2 = calc_contrast(img_roi2)
    logger.debug("Step 5/6 done: contrast_roi1=%s, contrast_roi2=%s", contrast_roi1, contrast_roi2)

    nps_f_av, nps_f_peak = float("nan"), float("nan")
    if recipe == RECIPE_PHYSICAL:
        nps_f_av, nps_f_peak = noise_texture(nps_roi2.f_avg, nps_roi2.nps_measured_radagv)

    feature_vector = np.array([
        mtf_area,
        nps_roi2.var_nps,       # npsHomo
        nps_roi1.var_nps,       # npsStruct
        nps_roi1.var_measured,  # varStruct
        ratio.ratio_lf_hf,
        ratio.slope_avg,
        ratio.loc_max,
    ])
    logger.debug("Step 6/6: classify_quality(feature_vector=%s)", feature_vector.tolist())
    predicted_label, decision_value = classify_quality(feature_vector, svm_model)
    logger.debug("Step 6/6 done: predicted_label=%s, decision_value=%s", predicted_label, decision_value)

    return SliceQualityResult(
        mtf_xaxis=xaxis,
        mtf=mtf,
        mtf_area_per_edge=mtf_area_per_edge,
        mtf_area=mtf_area,
        nps_roi1=nps_roi1,
        nps_roi2=nps_roi2,
        contrast_roi1=contrast_roi1,
        contrast_roi2=contrast_roi2,
        ratio=ratio,
        feature_vector=feature_vector,
        predicted_label=predicted_label,
        decision_value=decision_value,
        mtf_seed=mtf_seed,
        recipe=recipe,
        nps_f_av=nps_f_av,
        nps_f_peak=nps_f_peak,
    )


def _drop_invalid_mtf_edges(mtf, mtf_area_per_edge):
    """`mtf` / `mtf_area_per_edge` without edge columns whose MTF isn't finite."""
    valid = np.isfinite(mtf).all(axis=0) & np.isfinite(mtf_area_per_edge)
    n_edges = valid.size
    n_dropped = int(n_edges - valid.sum())
    if n_dropped == 0:
        return mtf, mtf_area_per_edge
    if n_dropped > MAX_DROPPED_MTF_EDGE_FRACTION * n_edges:
        raise ValueError(f"MTF invalid on {n_dropped} of {n_edges} edge points - ROI1 edge is unusable")
    logger.info("MTF: dropped %d of %d edge points with a non-finite MTF", n_dropped, n_edges)
    return mtf[:, valid], mtf_area_per_edge[valid]
