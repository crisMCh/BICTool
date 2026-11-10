"""Example BICTool plug-in: a WM/GM method (see docs/plugins.md).

Splits the brain into CSF, white and grey matter with two Otsu thresholds on
the HU histogram inside a plausible range. It appears in the Brain Tissue
box's Method drop-down as "Otsu (example)"; its settings dialog is built from
`OtsuSettings`. To try it, copy this file into the plug-in folder and restart
BICTool.

An example only - not validated against the built-in methods.
"""

from dataclasses import dataclass

import numpy as np

from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(
    name="otsu-wmgm",
    version="1.0",
    requires_api=">=0.2,<1",
    description="Example: WM/GM by two Otsu thresholds",
)


@dataclass(frozen=True)
class OtsuSettings:
    csf_below_hu: float = 15.0     # pixels below this are CSF
    tissue_above_hu: float = 50.0  # pixels above this are left out (vessels, calcification)
    bins: int = 64


def _otsu(values: np.ndarray, bins: int) -> float:
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(counts)
    mean = np.cumsum(counts * centers)
    total, total_mean = weight[-1], mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total_mean * weight - mean * total) ** 2 / (weight * (total - weight))
    return float(centers[np.nanargmax(between[:-1])])


def classify(slice_hu, brain_mask, settings: OtsuSettings, context):
    tissue = brain_mask & (slice_hu >= settings.csf_below_hu) & (slice_hu <= settings.tissue_above_hu)
    if tissue.sum() < 2 * settings.bins:
        empty = np.zeros(slice_hu.shape, dtype=bool)
        return {plugins.WHITE_MATTER: empty, plugins.GRAY_MATTER: empty, plugins.CSF: brain_mask & (slice_hu < settings.csf_below_hu)}
    threshold = _otsu(slice_hu[tissue], settings.bins)
    return {
        plugins.WHITE_MATTER: tissue & (slice_hu < threshold),
        plugins.GRAY_MATTER: tissue & (slice_hu >= threshold),
        plugins.CSF: brain_mask & (slice_hu < settings.csf_below_hu),
    }


@plugins.hookimpl
def bictool_tissue_methods():
    return [plugins.TissueMethod("otsu", "Otsu (example)", classify, OtsuSettings(), "Two Otsu thresholds")]
