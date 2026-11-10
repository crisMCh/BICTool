"""Example BICTool plug-in: extra metrics (see docs/plugins.md).

Adds the mean and SD of both ROIs of every measured pair, and the median ROI2
SD over the series. To try it, copy this file into the plug-in folder
(`plugins/` next to BICTool.exe, or `~/.bictool/plugins` when run from
source) and restart BICTool; it appears under Settings > Plug-ins.

An example only - these values are not validated quality measures.
"""

import numpy as np

from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(
    name="roi-stats",
    version="1.0",
    requires_api=">=0.2,<1",
    description="Example: mean and SD of both ROIs",
)


@plugins.hookimpl
def bictool_pair_metrics(pair):
    roi1, roi2 = pair.roi1_pixels, pair.roi2_pixels
    return [
        plugins.MetricValue("roi1_mean", float(roi1.mean()), "HU", "Mean of ROI1 (on the structure)"),
        plugins.MetricValue("roi2_mean", float(roi2.mean()), "HU", "Mean of ROI2 (homogeneous tissue)"),
        plugins.MetricValue("roi2_sd", float(roi2.std(ddof=1)), "HU", "Pixel SD of ROI2"),
    ]


@plugins.hookimpl
def bictool_study_metrics(study):
    sds = [float(p.roi2_pixels.std(ddof=1)) for p in study.pairs if p.pair == "falx"]
    return [plugins.MetricValue("falx_roi2_sd_median", float(np.median(sds)) if sds else float("nan"), "HU")]
