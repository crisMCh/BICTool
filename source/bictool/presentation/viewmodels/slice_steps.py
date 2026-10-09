"""Results > Per slice > Advanced: one slice's intermediate QA steps as
plot-ready panels.

Built from a `SliceQualityResult` with `steps` (`evaluate_slice_quality(...,
keep_steps=True)`):

* **MTF steps** - ROI1 (upsampled, as the MTF sees it) with the active-contour
  outline, the detected edge pixels, a sample of the line profiles and the
  start region; the edge profiles (ESF) and their derivative (LSF); the MTF
  of every edge with their mean.
* **MTF spread** - the edge pixels coloured by their MTF area, the histogram of
  the areas, and how many edges survived each step.
* **ROIs & 2D spectra** - both ROIs, the 2D PS of ROI1 and the 2D NPS of ROI2.
* **PS/MTF** - the ratio curve with the feature band, its low/high halves and
  the maximum.

Only data and colour *roles* - drawing is the view's job. Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from bictool.domain.metrics.mtf import MtfSteps
from bictool.domain.metrics.physical_spectra import FEATURE_BAND_MM, mtf_sample_spacing_mm
from bictool.domain.results.study_quality import RECIPE_PHYSICAL

# Brain window for every HU image (as in the viewer).
WINDOW_WIDTH, WINDOW_CENTER = 80.0, 40.0
MAX_CURVES = 40          # individual edge curves drawn (evenly sampled)
MAX_PROFILE_LINES = 12   # line profiles drawn on ROI1
HISTOGRAM_BINS = 15

# Colour roles (the view maps them to its palette).
ROLE_CONTOUR, ROLE_EDGE, ROLE_PROFILE, ROLE_SEED, ROLE_DROPPED = "contour", "edge", "profile", "seed", "dropped"
ROLE_MEAN, ROLE_FAINT, ROLE_BAND, ROLE_MARK = "mean", "faint", "band", "mark"
ROLE_LOW, ROLE_HIGH = "low", "high"  # the two halves of the PS/MTF feature band


@dataclass(frozen=True)
class ImagePanel:
    title: str
    image: np.ndarray                      # 2D, display values 0..1
    caption: str = ""
    contour: np.ndarray | None = None      # (n, 2) row/col of an outline (ROLE_CONTOUR)
    boxes: tuple[tuple[int, int, int, int], ...] = ()          # (row0, row1, col0, col1), ROLE_SEED
    points: tuple[tuple[np.ndarray, str], ...] = ()            # ((n, 2) row/col, role)
    lines: np.ndarray | None = None        # (n, 4) row0, col0, row1, col1 (ROLE_PROFILE)
    coloured_points: np.ndarray | None = None  # (n, 2) row/col coloured by `point_values`
    point_values: np.ndarray | None = None
    value_range: tuple[float, float] | None = None  # of `point_values` (colour scale)
    extent: tuple[float, float, float, float] | None = None  # (x0, x1, y0, y1) axis values, e.g. 1/mm


@dataclass(frozen=True)
class CurvePanel:
    title: str
    x_label: str
    y_label: str
    x: np.ndarray
    faint: tuple[np.ndarray, ...] = ()     # individual curves (ROLE_FAINT)
    mean: np.ndarray | None = None         # ROLE_MEAN
    bands: tuple[tuple[float, float, str], ...] = ()  # shaded x-ranges (x0, x1, role)
    marks: tuple[float, ...] = ()          # vertical lines (ROLE_MARK)
    note: str = ""


@dataclass(frozen=True)
class HistogramPanel:
    title: str
    x_label: str
    edges: np.ndarray                      # bin edges
    counts: np.ndarray
    marks: tuple[float, ...] = ()          # e.g. the mean
    note: str = ""


@dataclass(frozen=True)
class SliceStepsView:
    roi1_edges: ImagePanel
    esf: CurvePanel
    lsf: CurvePanel
    mtf: CurvePanel
    edge_map: ImagePanel
    area_histogram: HistogramPanel
    edge_counts: tuple[str, ...]
    roi1: ImagePanel
    roi2: ImagePanel
    ps_2d: ImagePanel
    nps_2d: ImagePanel
    ratio: CurvePanel
    ratio_features: tuple[str, ...] = field(default=())


def window(image: np.ndarray, width: float = WINDOW_WIDTH, center: float = WINDOW_CENTER) -> np.ndarray:
    """HU -> 0..1 display values for the given window."""
    low = center - width / 2
    return np.clip((np.asarray(image, dtype=np.float64) - low) / width, 0.0, 1.0)


def log_display(spectrum: np.ndarray) -> np.ndarray:
    """A 2D spectrum on a log scale, 0..1 between its 1st and 99.5th percentile."""
    values = np.log10(np.maximum(np.asarray(spectrum, dtype=np.float64), 1e-12))
    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def outline(mask: np.ndarray) -> np.ndarray:
    """(n, 2) row/col of the boundary pixels of a bool mask."""
    mask = np.asarray(mask, dtype=bool)
    return np.argwhere(mask & ~ndi.binary_erosion(mask))


def sample_columns(n: int, limit: int = MAX_CURVES) -> np.ndarray:
    """Up to `limit` evenly spread indices of `n` columns."""
    if n <= limit:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, limit).round().astype(int))


def slice_steps_view(result, pixel_spacing_mm: float) -> SliceStepsView | None:
    """All panels for one slice result, or None without kept steps."""
    steps = getattr(result, "steps", None)
    if steps is None:
        return None
    mtf_steps: MtfSteps = steps.mtf
    physical = getattr(result, "recipe", None) == RECIPE_PHYSICAL
    dx = mtf_sample_spacing_mm(pixel_spacing_mm)
    n_samples = mtf_steps.esf.shape[0]
    position = (np.arange(n_samples) - (n_samples - 1) / 2) * dx
    shown = sample_columns(mtf_steps.esf.shape[1])

    # ---- MTF steps ----
    kept_profiles = mtf_steps.kept
    line_sample = kept_profiles[sample_columns(len(kept_profiles), MAX_PROFILE_LINES)]
    # The edge pixel behind each used MTF column (same order as the result's MTF columns).
    final_points = (
        mtf_steps.edge_points[mtf_steps.profile_edges[kept_profiles]] if len(kept_profiles) else np.zeros((0, 2))
    )
    roi1_edges = ImagePanel(
        title="ROI1 as the MTF sees it",
        image=window(mtf_steps.image),
        caption=f"{mtf_steps.upsample}x upsampled, W{WINDOW_WIDTH:g}/L{WINDOW_CENTER:g}. Outline: active contour; "
        "dots: edge pixels (grey = dropped); lines: sampled profiles; box: start region",
        contour=outline(mtf_steps.segmentation),
        boxes=(mtf_steps.seed_box,),
        points=((mtf_steps.edge_points, ROLE_DROPPED), (final_points, ROLE_EDGE)),
        lines=mtf_steps.profile_lines[line_sample] if len(line_sample) else None,
    )
    esf_raw = mtf_steps.esf_raw[:, kept_profiles] if len(kept_profiles) else np.zeros((n_samples, 0))
    esf = CurvePanel(
        title="Edge profiles (ESF) across the edge",
        x_label="Distance from the edge [mm]",
        y_label="HU",
        x=position,
        faint=tuple(esf_raw[:, k] for k in shown),
        mean=esf_raw.mean(axis=1) if esf_raw.shape[1] else None,
        note=f"{esf_raw.shape[1]} edges used, {len(shown)} drawn; bold: mean",
    )
    lsf_x = (position[:-1] + position[1:]) / 2
    lsf = CurvePanel(
        title="Line spread (LSF) - derivative of the processed ESF",
        x_label="Distance from the edge [mm]",
        y_label="d(HU)",
        x=lsf_x,
        faint=tuple(mtf_steps.lsf[:, k] for k in shown),
        mean=mtf_steps.lsf.mean(axis=1) if mtf_steps.lsf.shape[1] else None,
        note="mean removed, low-pass filtered, bumps removed",
    )
    mtf_x, mtf = np.asarray(result.mtf_xaxis), np.asarray(result.mtf)
    if not physical:  # the legacy axis mirrors around its middle - show the first half
        half = len(mtf_x) // 2
        mtf_x, mtf = mtf_x[:half], mtf[:half]
    mtf_panel = CurvePanel(
        title="MTF of every edge",
        x_label="Spatial frequency [1/mm]",
        y_label="MTF",
        x=mtf_x,
        faint=tuple(mtf[:, k] for k in sample_columns(mtf.shape[1])),
        mean=np.nanmean(mtf, axis=1) if mtf.shape[1] else None,
        bands=((0.0, FEATURE_BAND_MM[1], ROLE_BAND),) if physical else (),
        note=(
            f"shaded: 0-{FEATURE_BAND_MM[1]:g} 1/mm, the range of the MTF area" if physical
            else "legacy axis (first half)"
        ),
    )

    # ---- MTF spread ----
    areas = np.asarray(result.mtf_area_per_edge, dtype=np.float64)
    finite = np.isfinite(areas)
    value_range = (float(np.min(areas[finite])), float(np.max(areas[finite]))) if finite.any() else (0.0, 1.0)
    edge_map = ImagePanel(
        title="Edge pixels coloured by their MTF area",
        image=window(mtf_steps.image),
        caption=f"dark = low (blurry), bright = high (sharp): {value_range[0]:.3g} - {value_range[1]:.3g}",
        contour=outline(mtf_steps.segmentation),
        points=((mtf_steps.edge_points, ROLE_DROPPED),),
        coloured_points=final_points,
        point_values=areas,
        value_range=value_range,
    )
    counts, bin_edges = np.histogram(areas[finite], bins=HISTOGRAM_BINS) if finite.any() else (np.zeros(0), np.zeros(1))
    histogram = HistogramPanel(
        title="MTF area per edge",
        x_label="MTF area" + (" [1/mm]" if physical else ""),
        edges=bin_edges,
        counts=counts,
        marks=(float(result.mtf_area),),
        note=f"line: the slice's MTF area (mean) {result.mtf_area:.3g}, SD {np.std(areas[finite]):.2g}"
        if finite.any() else "",
    )
    edge_counts = (
        f"Edge pixels found by the active contour: {len(mtf_steps.edge_points)}",
        f"... with a full profile inside the ROI: {len(mtf_steps.profile_edges)}",
        f"... kept by the bump removal: {mtf_steps.n_after_bumps}",
        f"... with a finite MTF (used): {mtf.shape[1]}",
        f"Active-contour start (seed): {result.mtf_seed}" if result.mtf_seed else "",
    )

    # ---- ROIs and 2D spectra ----
    roi1 = ImagePanel("ROI1 (structure)", window(steps.roi1), f"W{WINDOW_WIDTH:g}/L{WINDOW_CENTER:g}")
    roi2 = ImagePanel("ROI2 (homogeneous)", window(steps.roi2), f"W{WINDOW_WIDTH:g}/L{WINDOW_CENTER:g}")
    ps_2d = _spectrum_panel("2D PS of ROI1", result.nps_roi1)
    nps_2d = _spectrum_panel("2D NPS of ROI2", result.nps_roi2)

    # ---- PS/MTF ----
    ratio = result.ratio
    f_int = np.asarray(ratio.f_int)
    marks, bands = (), ()
    features = (
        f"LF/HF ratio: {ratio.ratio_lf_hf:.4g}",
        f"Mean slope: {ratio.slope_avg:.4g}",
        f"Maximum at: {ratio.loc_max:.4g}" + (" 1/mm" if physical else ""),
    )
    if physical and len(f_int):
        split = float(f_int[len(f_int) // 2])
        bands = ((float(f_int[0]), split, ROLE_LOW), (split, float(f_int[-1]), ROLE_HIGH))
        marks = (float(ratio.loc_max),)
        if ratio.band_truncated:
            features += (f"Band cut short to {ratio.band[1]:.3g} 1/mm (coarse pixel spacing)",)
    ratio_panel = CurvePanel(
        title="PS/MTF ratio",
        x_label="Spatial frequency [1/mm]",
        y_label="PS/MTF (normalised)",
        x=f_int,
        mean=np.asarray(ratio.ratio_nps_mtf),
        bands=bands,
        marks=marks,
        note=(
            "LF/HF ratio = area under the curve in the low half (left shade) / high half (right shade) "
            "of the feature band; line: the maximum" if physical else "legacy recipe"
        ),
    )
    return SliceStepsView(
        roi1_edges=roi1_edges, esf=esf, lsf=lsf, mtf=mtf_panel,
        edge_map=edge_map, area_histogram=histogram, edge_counts=tuple(line for line in edge_counts if line),
        roi1=roi1, roi2=roi2, ps_2d=ps_2d, nps_2d=nps_2d,
        ratio=ratio_panel, ratio_features=features,
    )


def _spectrum_panel(title: str, spectrum) -> ImagePanel:
    f = np.asarray(spectrum.f, dtype=np.float64)
    extent = (float(f[0]), float(f[-1]), float(f[0]), float(f[-1])) if f.size else None
    return ImagePanel(
        title=title,
        image=log_display(spectrum.nps_measured),
        caption="log scale, centre = 0 1/mm, axes in 1/mm. Noise: a round blob; lines or a stretched shape: "
        "directional structure (edges, streaks)",
        extent=extent,
    )
