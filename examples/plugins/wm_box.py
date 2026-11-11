"""Example BICTool plug-in: a metric with its own ROI (see docs/developer, how-to/metric-with-own-roi.md).

On every measured slice it places one square ROI of its own in the white
matter - the largest square (up to 15 mm) centred inside BICTool's
white-matter mask - and reports its mean and pixel SD. The ROI is shown on
the image when "Plug-in ROIs" is ticked (Run QA box). It also reports the
median SD over the series, and names slices whose SD is far above that
median as a possible artefact.

To try it, copy this file into the plug-in folder (`plugins/` next to
BICTool.exe, or `~/.bictool/plugins` when run from source) and restart
BICTool; it appears under Settings > Plug-ins.

An example only - these values are not validated quality measures.
"""

import numpy as np
from scipy import ndimage

from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(
    name="wm-box",
    version="1.0",
    requires_api=">=0.3,<1",
    description="Example: mean and SD in a white-matter square of its own",
)

MIN_SIDE_MM, MAX_SIDE_MM = 5.0, 15.0
HIGH_SD_FACTOR = 2.0  # "possible artefact": SD above this times the series median


def _white_matter_square(tissue, brain_mask, spacing_mm: float) -> np.ndarray | None:
    """The largest square (odd side, at most MAX_SIDE_MM) centred inside the white matter."""
    white = tissue[plugins.WHITE_MATTER] & brain_mask
    if not white.any():
        return None
    # Chessboard distance d at a pixel = a square of side 2d - 1 around it lies inside the mask.
    distance = ndimage.distance_transform_cdt(white, metric="chessboard")
    row, col = np.unravel_index(int(np.argmax(distance)), distance.shape)
    half = min(int(distance[row, col]) - 1, int(MAX_SIDE_MM / spacing_mm) // 2)
    if (2 * half + 1) * spacing_mm < MIN_SIDE_MM:
        return None
    roi = np.zeros(white.shape, dtype=bool)
    roi[row - half : row + half + 1, col - half : col + half + 1] = True
    return roi


def _sd(image, roi) -> float:
    return float(image[roi].std(ddof=1)) if roi is not None else float("nan")


@plugins.hookimpl
def bictool_slice_metrics(ct_slice):
    roi = _white_matter_square(ct_slice.tissue(), ct_slice.brain_mask(), ct_slice.pixel_spacing_mm)
    if roi is None:
        return [plugins.MetricValue("wm_mean", float("nan"), "HU"), plugins.MetricValue("wm_sd", float("nan"), "HU")]
    return [
        plugins.MetricValue("wm_mean", float(ct_slice.image[roi].mean()), "HU", "Mean in the white-matter square"),
        plugins.MetricValue("wm_sd", _sd(ct_slice.image, roi), "HU", "Pixel SD in the white-matter square"),
        plugins.RoiMarker("white-matter square", roi),
    ]


def _series_sds(study) -> dict[int, float]:
    sds = {}
    for index in study.measured_slices:
        roi = _white_matter_square(study.tissue(index), study.brain_mask(index), study.pixel_spacing_mm)
        if roi is not None:
            sds[index] = _sd(study.image(index), roi)
    return sds


@plugins.hookimpl
def bictool_study_metrics(study):
    sds = list(_series_sds(study).values())
    return [plugins.MetricValue("wm_sd_median", float(np.median(sds)) if sds else float("nan"), "HU")]


@plugins.hookimpl
def bictool_artefact_hints(study):
    sds = _series_sds(study)
    if len(sds) < 3:
        return []
    median = float(np.median(list(sds.values())))
    high = tuple(index for index, sd in sds.items() if sd > HIGH_SD_FACTOR * median)
    return [plugins.ArtefactHint("White-matter SD far above the series median", high)] if high else []
