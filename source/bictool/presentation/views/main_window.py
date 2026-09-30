"""Main application window (View layer).

Structural port of `design/bictool_console_mockup.html`'s single fixed window:
a menu bar, a large viewer/results pane on the left, and a sidebar (Series /
Segmentation + ROI Selection / Run Metrics) on the right.

Wired to real functionality for a first end-to-end test:
- **DICOM import**: File > Open DICOM Folder... recursively finds every
  series/stack under a local folder (see `integration.dicom_file_loader`'s
  module docstring for the discovery pipeline - series with inconsistent
  image geometry are transparently split into separate loadable candidates
  rather than ever erroring); File > Open DICOM File... loads a single file
  as a one-slice volume. The sidebar's "Series" dropdown always shows what
  was found (auto-selects the best guess: CT modality, most slices) and lets
  the user pick a different one - switching it calls `load_series` directly
  (cheap - the file list is already known from discovery, no rescanning).
  The scan itself runs on a background thread via `DicomLoadWorker` (same
  Worker/Scheduler pattern as metric computation) and pixel data is decoded
  lazily per slice (`DicomVolume.get_slice_hu`) - loading a 500-slice study
  does not mean scanning or decoding 500 slices' pixel data up front.
  Patient/study/series identity, scanner/institution, acquisition technique
  (kVp/mA), geometry (slice thickness/location, pixel spacing), and current
  display state (WL/WW, zoom%) are shown as four screen-corner text overlays
  directly on the image (`_update_viewer_overlay` / `RoiViewport.
  set_overlay_text`), not in the sidebar - matching field-for-field what
  RadiAnt/OsiriX/MicroDicom show by default (see `_update_viewer_overlay`'s
  own docstring for the research), freeing the sidebar for segmentation
  controls. Refreshes on every volume load, slice change, window-preset
  click, and zoom change, so nothing on it is ever a stale reading from a
  previous series/slice/display state.
- **ROI selection**: draw-then-assign. Any left-drag on `RoiViewport` draws
  a rectangle (no arming step first - see the class's own docstring);
  "Set ROI1"/"Set ROI2" (sidebar, ROI Selection box) then save *that*
  rectangle into the respective slot, shown in a small ROI/Position/Size
  table rather than free-text labels (a prior version's coordinate text
  didn't fit its column and grew the sidebar, visibly shrinking the viewer
  pane - see project memory). Specific to the currently displayed slice -
  changing slice clears both ROIs, the drawn rectangle, and un-confirms,
  since Run Metrics always operates on the slice on screen (FR-04's "quality
  score per image slice", not a whole-volume score). Panning is a
  middle-button drag (left-drag is reserved for drawing); hovering over the
  image live-shows cursor position and HU value in the bottom-right corner
  overlay, the same spot that shows the rectangle's size while drawing one.
- **Segmentation (FR-02)**: "Segment Slice" runs a pretrained 16-structure
  CNN (`domain.segmentation.head_ct_unet`) on the current slice via a
  `SegmentationWorker` off the GUI thread. A plain (uncompiled) CPU forward
  pass on this architecture measured ~157s per slice - `_unet_architecture
  .build_unet` now compiles the model with `jit_compile=True` (XLA), which
  fuses that down to ~1-2s per slice after a one-time ~2s trace on the
  model's first call each session (see that module's docstring for the
  measurement). It's still an explicit trigger rather than a per-checkbox
  auto-run - and its result is still cached per slice index - since even a
  couple of seconds per checkbox toggle would be a poor default, and a
  multi-hundred-slice study's total cost is still worth batching rather
  than paying slice-by-slice. "Segment Volume" is the batch counterpart
  (`VolumeSegmentationWorker`): runs every not-yet-cached slice in the
  background, live-updating the currently displayed slice as soon as its
  own result lands, showing "N/total" progress, and supporting mid-run
  cancellation (checked between slices, not mid-`predict()` - a single call
  can't be interrupted) - a few minutes for a full multi-hundred-slice
  study at this per-slice cost, not blocking, so it still runs as a
  background job with progress/cancel rather than a modal wait.
  `_maybe_auto_segment_volume` auto-starts this same "Segment Volume" batch,
  with no user action at all, the moment the study's primary series (the
  one `discover_series` auto-selects and loads by default - see
  `_load_selected_series`) finishes loading, if anatomy weights are
  configured - per explicit request, since that primary series is normally
  the one the user actually wants segmented. Any other series the user
  picks from the "Series" dropdown still only segments on an explicit
  Segment Slice/Volume click; the auto-start silently does nothing (no
  error dialog) if weights aren't configured or TensorFlow isn't
  installed, since it's a convenience, not a user-initiated action. Which
  structures to overlay is chosen in a separate, non-modal "Select
  Segments..." window (`SegmentationSelectionDialog`) - kept out of this
  sidebar so a long checkbox list doesn't crowd it - showing exactly the
  16 raw structures the model produces (see `_SEGMENTATION_MAPPING`; no
  WP2-specific renaming/combining, no Brainstem placeholder, per explicit
  request). The selection itself (`_selected_segmentation_names`) is a
  whole-volume-scoped choice, not a per-slice one - it survives slice
  navigation (re-applied from that slice's cached result, if segmented)
  and is only cleared on close study/
  new volume load. "Check for Lesion" (Stroke Detection box) is
  the same mechanism wrapping a different, transfer-learned model
  (`aisd_stroke_unet`) - kept visually and architecturally separate since
  it's a diagnostic-adjacent capability, not a QC/image-quality
  measurement, and labeled investigational in the UI. Neither model's
  weights are bundled with BICTool - both are user-supplied external files
  configured via `infrastructure.config` (env vars or a small JSON file).
- **Run Metrics**: dispatches a real `bictool.application.worker.Worker`
  (i.e. `domain.core_api.evaluate_slice_quality`) via `Scheduler`, and
  renders the real `SliceQualityResult` - not example values, including
  real MTF/NPS curves via `PySide6.QtCharts` (`_update_line_chart`).
- **Viewing**: clinical windowing presets (dropdown - Brain/Subdural/Bone/
  Stroke/Soft tissue, see `viewmodels.windowing` for sourcing), zoom
  (Ctrl+wheel)/pan (middle-button drag, since left-drag draws ROIs) that
  persists across slice navigation, and a segmentation/metric overlay
  mechanism (`RoiViewport.set_overlay`) for FR-06 - not yet wired to
  anything since FR-02 (automatic segmentation) doesn't exist and this
  project avoids shipping fake/example data in the UI.
- **Trace panel**: a persistent, always-visible log panel under the main
  content, fed by `infrastructure.logging_config.QtLogHandler` - every
  `logging` call anywhere in the app (DICOM loading, ROI selection, metric
  computation step-by-step) shows up here live, including full tracebacks
  on error. This is a debugging aid added while diagnosing real Run-All
  failures, not a permanent UI element - see project memory before removing
  or hiding it by default.

Still placeholder/out of scope: observer-study integration, the study-level
batch quality score half of FR-04 (only per-slice scoring exists). Napari
was evaluated and deliberately not adopted - see project memory for the
analysis.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydicom
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QButtonGroup,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bictool.application.brain_tissue_volume_worker import BrainTissueVolumeWorker
from bictool.application.dicom_load_worker import DicomLoadWorker
from bictool.application.scheduler import Scheduler
from bictool.application.worker import Worker
from bictool.domain.metrics.mtf_seed import mtf_seed_candidates
from bictool.domain.results.study_quality import SliceQualityResult
from bictool.domain.results.study_summary import ACCEPTABLE, UNACCEPTABLE, study_verdict, summarize_metrics
from bictool.domain.roi_selection.homogeneous_roi import FALLBACK, VENTRICLE, WHITE_MATTER, find_roi2_candidate
from bictool.domain.roi_selection.structured_roi import (
    DURAL_FOLDS,
    SEPTUM_PELLUCIDUM,
    STRUCTURES,
    RoiCandidate,
    estimate_midline_direction,
    find_roi1_candidate,
)
from bictool.domain.segmentation.brain_tissue import (
    DEFAULT_AIR_HU_CUTOFF,
    DEFAULT_BONE_HU_CUTOFF,
    DEFAULT_CSF_RANGE,
    DEFAULT_GRAY_MATTER_RANGE,
    DEFAULT_MIN_COMPONENT_SIZE,
    DEFAULT_SMOOTHING_SIZE,
    DEFAULT_WHITE_MATTER_RANGE,
    classify_brain_tissue,
    extract_brain_mask,
)
from bictool.domain.segmentation.segmentation_brain_mask import brain_mask_from_segmentation
from bictool.domain.segmentation.brain_tissue_adaptive import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PLAUSIBLE_RANGE,
)
from bictool.domain.segmentation.brain_tissue_adaptive import (
    DEFAULT_MIN_COMPONENT_SIZE as ADAPTIVE_DEFAULT_MIN_COMPONENT_SIZE,
)
from bictool.domain.segmentation.brain_tissue_adaptive import classify_brain_tissue_adaptive
from bictool.domain.segmentation.brain_tissue_fcm import (
    DEFAULT_FUZZINESS,
    DEFAULT_MAX_ITER,
    DEFAULT_SPATIAL_WEIGHT,
)
from bictool.domain.segmentation.brain_tissue_fcm import (
    DEFAULT_MIN_COMPONENT_SIZE as FCM_DEFAULT_MIN_COMPONENT_SIZE,
)
from bictool.domain.segmentation.brain_tissue_fcm import classify_brain_tissue_fcm
from bictool.infrastructure.config import load_config
from bictool.infrastructure.logging_config import QtLogHandler, configure_logging
from bictool.integration.dicom_file_loader import DicomVolume, SeriesCandidate, load_series
from bictool.integration.results_export import (
    ExportContext,
    StudyExportContext,
    export_metrics_csv,
    export_metrics_excel,
    export_study_csv,
    export_study_excel,
)
from bictool.presentation.viewmodels.chart_data import mtf_display_curve
from bictool.presentation.viewmodels.roi_selection import RoiRect, crop_roi
from bictool.presentation.viewmodels.segmentation_overlay import decode_masks_for_checkboxes
from bictool.presentation.viewmodels.windowing import DEFAULT_WINDOW_PRESET, WINDOW_PRESETS
from bictool.presentation.views.adaptive_settings_dialog import AdaptiveSettings, AdaptiveSettingsDialog
from bictool.presentation.views.brain_tissue_settings_dialog import BrainTissueSettings, BrainTissueSettingsDialog
from bictool.presentation.views.custom_window_dialog import CustomWindowDialog
from bictool.presentation.views.dicom_tags_dialog import DicomTagsDialog
from bictool.presentation.views.fcm_settings_dialog import FcmSettings, FcmSettingsDialog
from bictool.presentation.views.export_dialog import SCOPE_STUDY, ExportDialog
from bictool.presentation.views.roi_viewport import RoiViewport
from bictool.presentation.views.segmentation_selection_dialog import SegmentationSelectionDialog

if TYPE_CHECKING:
    # Type-hint-only - the real imports are deferred to first use inside
    # `_on_segment_slice_clicked`/`_on_check_stroke_clicked` (see there for
    # why): importing TensorFlow/Keras (transitively, via these modules)
    # costs a real ~3s at import time, which would otherwise tax every
    # single app launch even for users who never touch segmentation.
    from bictool.application.segmentation_worker import SegmentationWorker
    from bictool.application.stroke_worker import StrokeWorker
    from bictool.application.volume_segmentation_worker import VolumeSegmentationWorker
    # `NiftiConversionWorker` doesn't import TensorFlow, but does import
    # `dicom2nifti` (a native/SWIG-backed dependency) - deferred for the
    # same "don't tax every launch for a feature not every user touches"
    # reason, not TF's cost specifically.
    from bictool.application.nifti_conversion_worker import NiftiConversionWorker
    from bictool.domain.segmentation._nifti_alignment import DicomNiftiVolume

logger = logging.getLogger(__name__)

BG = "#11151a"
SURFACE = "#181d23"
SURFACE_RAISED = "#20262d"
SURFACE_SUNKEN = "#0c0f13"
BORDER = "#2b323a"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"
ACCENT = "#49c8de"
ACCENT_2 = "#d98bd8"
GOOD = "#5fbf83"
CRITICAL = "#e0685c"

STYLESHEET = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-size: 12px; }}
QMenuBar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item:selected {{ background: {SURFACE_RAISED}; }}
QMenu {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {BG}; }}
QGroupBox {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 14px; padding: 10px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {TEXT_MUTED}; }}
QPushButton {{
    background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 11px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; }}
QPushButton#runAllBtn {{ background: {ACCENT}; color: #08181c; font-weight: 600; border: none; }}
QPushButton#runAllBtn:disabled {{ background: {SURFACE_RAISED}; color: {TEXT_MUTED}; }}
QPushButton#navTabBtn {{
    border: none; border-bottom: 2px solid transparent; border-radius: 0;
    background: transparent; padding: 6px 16px; color: {TEXT_MUTED}; font-weight: 600;
}}
QPushButton#navTabBtn:hover {{ color: {TEXT}; border-color: {BORDER}; }}
QPushButton#navTabBtn:checked {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton#navTabBtn:disabled {{ color: {TEXT_MUTED}; border-color: transparent; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; }}
QLabel[role="mono"] {{ font-family: "IBM Plex Mono", Consolas, monospace; }}
QLabel[role="statusDone"] {{ color: {GOOD}; font-family: "IBM Plex Mono", Consolas, monospace; }}
QLabel[role="statusError"] {{ color: {CRITICAL}; font-family: "IBM Plex Mono", Consolas, monospace; }}
QLabel[role="statusQueued"] {{ color: {TEXT_MUTED}; font-family: "IBM Plex Mono", Consolas, monospace; }}
QFrame#sunken {{ background: {SURFACE_SUNKEN}; border: 1px solid {BORDER}; border-radius: 6px; }}
"""

_METRIC_ROWS = [("mtf", "MTF"), ("nps", "NPS"), ("ratio", "PS/MTF"), ("contrast", "Contrast")]
_FEATURE_NAMES = ["mtf_area", "nps_homo", "nps_struct", "var_struct", "ratio_lf_hf", "slope_avg", "loc_max"]
_STATUS_COLORS = {"statusDone": GOOD, "statusError": CRITICAL, "statusQueued": TEXT_MUTED}

# The 16 raw structures `head_ct_unet` produces, shown directly (no WP2
# curation/combination, no primary/"Other structures" split) per explicit
# request - must stay in sync with that module's `_CLASS_NAMES` (kept as a
# literal copy here, not imported, since `head_ct_unet` pulls in TensorFlow
# at import time and this module is loaded eagerly at app startup - see the
# TYPE_CHECKING import block below for the same reasoning applied to the
# worker classes). No Brainstem entry - it isn't one of these 16 (see
# `head_ct_unet`'s docstring for that gap, still real, just not a checkbox
# here anymore). Index 8/11 order (Insular Cortex / Central Sulcus, not the
# reverse) confirmed against the model's own training source
# (`jasonccai/HeadCTSegmentation`'s `z_controlboard.py`) and independently
# against `reference/segmentation/Prediction_2CP_Bad2_2_head_40_axial.nii.gz`'s
# actual per-label anatomy - see `head_ct_unet.py`'s docstring for the detail.
_SEGMENTATION_MAPPING: dict[str, list[str]] = {
    "Brain Parenchyma": ["Brain Parenchyma"],
    "Subarachnoid Space": ["Subarachnoid Space"],
    "Dural Folds": ["Dural Folds"],
    "Septum Pellucidum": ["Septum Pellucidum"],
    "Cerebellum": ["Cerebellum"],
    "Caudate Nucleus": ["Caudate Nucleus"],
    "Lentiform Nucleus": ["Lentiform Nucleus"],
    "Insular Cortex": ["Insular Cortex"],
    "Internal Capsule": ["Internal Capsule"],
    "Ventricular System": ["Ventricular System"],
    "Central Sulcus": ["Central Sulcus"],
    "Frontal Lobe": ["Frontal Lobe"],
    "Parietal Lobe": ["Parietal Lobe"],
    "Occipital Lobe": ["Occipital Lobe"],
    "Temporal Lobe": ["Temporal Lobe"],
    "Thalamus": ["Thalamus"],
}
# One color per structure (16, matching `_SEGMENTATION_MAPPING` 1:1 - no
# repeats) - evenly spaced by hue (HLS, L=0.60/S=0.55 held fixed for
# consistent vibrancy/contrast against the dark theme) rather than the
# original 7-color palette, which repeated every 7 structures and made
# several unrelated structures share an identical overlay color (2026-09-25
# report: "many of them have the same or very similar color"). Hue bands
# around ROI1's cyan (~190°), ROI2's pink (~310°), and the stroke overlay's
# red (~5°, CRITICAL) are skipped (±18°) so a segmentation overlay is never
# confusable with those other overlay kinds either.
_SEGMENTATION_COLOR_PALETTE = [
    "#d19b61", "#d1b861", "#cdd161", "#afd161",
    "#92d161", "#74d161", "#61d16b", "#61d188",
    "#61d1a6", "#619cd1", "#617ed1", "#6161d1",
    "#7f61d1", "#9c61d1", "#b961d1", "#d16188",
]
# Assigning the (already hue-spread) palette in plain dict order still left
# *adjacent* structures looking similar (2026-09-25 report: "many similar
# colored labels are close to each other") - `_SEGMENTATION_MAPPING`'s
# order clusters anatomically-neighboring structures together (e.g. the
# four lobes, or Caudate/Lentiform Nucleus/Insular Cortex/Internal Capsule
# around the basal ganglia), so consecutive palette entries (only ~22.5°
# apart in hue) kept landing right next to each other in the image too.
# Reading the palette via a bit-reversal permutation instead (a standard
# technique for this exact problem) guarantees consecutive structures
# always draw from opposite/far-apart parts of the hue circle - see
# `test_main_window_segmentation.py`'s test asserting a minimum hue
# separation between every consecutive pair.
_SEGMENTATION_PALETTE_ORDER = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
# One color per display name, computed once (not per-widget) since the
# `SegmentationSelectionDialog`'s checkboxes and `RoiViewport`'s overlays
# both need the same display-name -> color association, independent of any
# widget lifetime.
_SEGMENTATION_COLORS: dict[str, str] = {
    name: _SEGMENTATION_COLOR_PALETTE[_SEGMENTATION_PALETTE_ORDER[i % len(_SEGMENTATION_PALETTE_ORDER)]]
    for i, name in enumerate(_SEGMENTATION_MAPPING)
}

# HU-threshold-only white/gray matter overlays (domain.segmentation.brain_tissue)
# - a separate feature from the CNN Segmentation box above, so its colors are
# picked to stay distinct from both that palette and ROI1/ROI2's cyan/pink.
_BRAIN_TISSUE_COLORS: dict[str, str] = {
    "CSF": "#5cc9a0",
    "White Matter": "#4c8ce0",
    "Gray Matter": "#e0a24c",
    "Brain Mask": "#8d96a0",  # neutral gray - the skull-strip result itself, not a tissue class
}

# The three WM/GM classification methods, selected from Tools -> WM/GM
# Classification (a radio-style submenu, not the sidebar) - each is a
# separate domain module with its own module docstring explaining its
# approach/trade-offs, and its own settings dialog (also under Tools).
# Fuzzy C-Means is the default per explicit request (2026-09-25) after
# comparing all three candidates the user tried; GMM+MRF (a fourth
# candidate briefly implemented for comparison) was removed entirely -
# see brain_tissue_fcm.py/brain_tissue_adaptive.py's own docstrings for
# the surviving two, and brain_tissue.py for the original threshold method.
_BRAIN_TISSUE_METHOD_FCM = "Fuzzy C-Means"
_BRAIN_TISSUE_METHOD_THRESHOLD = "HU Thresholding"
_BRAIN_TISSUE_METHOD_ADAPTIVE = "Adaptive Thresholding"
_BRAIN_TISSUE_METHODS = [
    _BRAIN_TISSUE_METHOD_FCM,
    _BRAIN_TISSUE_METHOD_THRESHOLD,
    _BRAIN_TISSUE_METHOD_ADAPTIVE,
]
_BRAIN_TISSUE_DEFAULT_METHOD = _BRAIN_TISSUE_METHOD_FCM


_ROI2_REGION_LABELS = {
    VENTRICLE: "ventricle",
    WHITE_MATTER: "white matter",
    FALLBACK: "fallback - no clean homogeneous area nearby",
}


_STUDY_TABLE_COLUMNS = ["Slice", "MTF area [1/mm]", "NPS ROI2 [HU²mm²]", "PS/MTF LF/HF", "Decision", "Verdict"]


def _candidate_to_rect(candidate: RoiCandidate) -> RoiRect:
    return RoiRect(candidate.col0, candidate.row0, candidate.col1, candidate.row1)


def _format_slice_ranges(slice_indices: list[int]) -> str:
    """Sorted 0-based indices -> 1-based display ranges, e.g. [19, 20, 21, 30] -> "20–22, 31"."""
    parts = []
    start = previous = slice_indices[0]
    for index in slice_indices[1:] + [None]:
        if index is not None and index == previous + 1:
            previous = index
            continue
        parts.append(f"{start + 1}" if start == previous else f"{start + 1}–{previous + 1}")
        if index is not None:
            start = previous = index
    return ", ".join(parts)


class MainWindowView(QMainWindow):
    """Application shell: menu bar + viewer/results pane + sidebar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BICTool")
        self.resize(1180, 880)  # taller than before to fit the always-visible trace bar
        self.setStyleSheet(STYLESHEET)

        self._scheduler = Scheduler()
        self._loaded_volume: DicomVolume | None = None
        self._current_slice_index: int = 0
        self._roi_rects: dict[str, RoiRect] = {}
        # ROIs are per slice: `_roi_rects` is the displayed slice's effective
        # ROIs (see `_effective_rois`) - hand-set ones (`_manual_rois`,
        # Set ROI1/ROI2) override automatically selected ones.
        self._manual_rois: dict[int, dict[str, RoiRect]] = {}
        # Transient line shown in the image's bottom-right corner overlay -
        # live cursor/pixel-value while hovering, or the in-progress
        # rectangle's size while drawing (see `_on_cursor_moved`/
        # `_on_draft_changed`); empty when neither is happening.
        self._interaction_line: str = ""
        self._active_worker: Worker | None = None
        # Run All = QA on every slice with both ROIs, one `Worker` at a time
        # (see `_run_next_qa_slice`). `_qa_jobs` is the queue while a batch
        # runs (None otherwise); results/errors are kept per slice for the
        # Results page and the study summary.
        self._qa_jobs: list[int] | None = None
        self._qa_total = 0
        self._qa_active_slice: int | None = None
        self._qa_cancel_requested = False
        self._qa_results: dict[int, SliceQualityResult] = {}
        self._qa_errors: dict[int, str] = {}
        self._active_dicom_load_worker: DicomLoadWorker | None = None
        self._series_candidates: list[SeriesCandidate] = []
        # The most recently computed result (and the non-identifying context
        # describing where it came from) - what Export Results (FR-07)
        # actually writes out. `None` until Run All succeeds at least once.
        self._last_result: SliceQualityResult | None = None
        self._last_export_context: ExportContext | None = None

        # Segmentation (FR-02) and stroke detection - both wrap external
        # pretrained models (see domain.segmentation's module docstring for
        # what/why). Models are built+weight-loaded lazily on first use
        # (not at startup) and reused across slices; results are cached per
        # slice index for the lifetime of the loaded volume - a single
        # inference call is now ~1-2s on CPU (see `_unet_architecture
        # .build_unet`'s docstring for the ~157s-to-~1-2s XLA fix), so this
        # is more about avoiding pointless recompute on repeated navigation
        # than about tolerating an otherwise-unusable wait.
        self._segmentation_config = load_config()
        self._anatomy_model = None
        self._stroke_model = None
        self._segmentation_cache: dict[int, object] = {}  # slice index -> head_ct_unet.SegmentationResult
        # Auto ROIs (ROI Selection box): per slice, an ROI1 and an ROI2
        # candidate (or the user's moved version), computed for every segmented
        # slice while the checkbox is on. Derived from `_segmentation_cache`
        # (and, for white-matter ROI2s, `_brain_tissue_cache`), so cleared
        # along with it. `_auto_roi2_moved` = slices whose ROI2 the user moved
        # by hand - not re-placed when their ROI1 moves.
        self._auto_roi1_candidates: dict[int, RoiCandidate] = {}
        self._auto_roi2_candidates: dict[int, RoiCandidate] = {}
        self._auto_roi2_moved: set[int] = set()
        self._auto_midline = None  # volume-wide midline direction (structured_roi.estimate_midline_direction)
        self._has_hand_drawn_draft = False  # see `_on_draft_changed`
        self._active_auto_tissue_worker: BrainTissueVolumeWorker | None = None
        self._stroke_cache: dict[int, object] = {}  # slice index -> aisd_stroke_unet.StrokeSegmentationResult
        # HU-threshold-only white/gray matter classification (domain.
        # segmentation.brain_tissue) - a separate, non-ML feature, so unlike
        # anatomy/stroke there's no model to load and no config-gated
        # enablement; the checkboxes just run this synchronously per slice
        # (milliseconds, not the CNN's ~160s) and cache the result.
        self._brain_tissue_cache: dict[int, object] = {}  # slice index -> brain_tissue.BrainTissueResult
        # Which of the three methods is active - chosen from Tools -> WM/GM
        # Classification (an app-level preference, like the settings below,
        # not per-study state - see `_reset_brain_tissue_display`).
        self._brain_tissue_method = _BRAIN_TISSUE_DEFAULT_METHOD
        # Tunable via Tools → HU Thresholding Settings… (`_on_brain_tissue_settings_clicked`)
        # - starts at brain_tissue.py's own defaults.
        self._brain_tissue_default_settings = BrainTissueSettings(
            air_hu_cutoff=DEFAULT_AIR_HU_CUTOFF,
            bone_hu_cutoff=DEFAULT_BONE_HU_CUTOFF,
            csf_low=DEFAULT_CSF_RANGE[0],
            csf_high=DEFAULT_CSF_RANGE[1],
            white_matter_low=DEFAULT_WHITE_MATTER_RANGE[0],
            white_matter_high=DEFAULT_WHITE_MATTER_RANGE[1],
            gray_matter_low=DEFAULT_GRAY_MATTER_RANGE[0],
            gray_matter_high=DEFAULT_GRAY_MATTER_RANGE[1],
            smoothing_size=DEFAULT_SMOOTHING_SIZE,
            min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
        )
        self._brain_tissue_settings = self._brain_tissue_default_settings
        # Tunable via Tools → Fuzzy C-Means Settings…
        self._fcm_default_settings = FcmSettings(
            fuzziness=DEFAULT_FUZZINESS,
            max_iter=DEFAULT_MAX_ITER,
            spatial_weight=DEFAULT_SPATIAL_WEIGHT,
            min_component_size=FCM_DEFAULT_MIN_COMPONENT_SIZE,
        )
        self._fcm_settings = self._fcm_default_settings
        # Tunable via Tools → Adaptive Threshold Settings…
        self._adaptive_default_settings = AdaptiveSettings(
            block_size=DEFAULT_BLOCK_SIZE,
            plausible_low=DEFAULT_PLAUSIBLE_RANGE[0],
            plausible_high=DEFAULT_PLAUSIBLE_RANGE[1],
            min_component_size=ADAPTIVE_DEFAULT_MIN_COMPONENT_SIZE,
        )
        self._adaptive_settings = self._adaptive_default_settings
        self._active_segmentation_worker: SegmentationWorker | None = None
        self._active_stroke_worker: StrokeWorker | None = None
        self._active_volume_segmentation_worker: VolumeSegmentationWorker | None = None
        # Segmentation runs on a NIfTI conversion of the loaded DICOM series,
        # not the raw DICOM pixel arrays directly - the model expects
        # orientation-canonicalized input the way its own training/reference
        # data was prepared (see domain.segmentation._nifti_alignment's
        # module docstring for the empirical evidence). Built lazily, once per
        # loaded study, on first segmentation request - not eagerly on every
        # volume load, since many users may never segment at all, and QA/
        # display never need it (they stay on DicomVolume throughout).
        self._dicom_nifti_volume: DicomNiftiVolume | None = None
        self._active_nifti_conversion_worker: NiftiConversionWorker | None = None
        # Backup of the primary series' segmentation state (see
        # `_on_volume_loaded`), so switching to another series and back
        # restores rather than re-segmenting from scratch - explicitly
        # scoped to the primary series only (2026-09-24 request), not every
        # series visited, to avoid keeping every series' full NIfTI volume +
        # per-slice masks resident in memory at once.
        self._primary_series_segmentation_cache: dict[int, object] | None = None
        self._primary_series_dicom_nifti_volume: DicomNiftiVolume | None = None
        self._current_series_is_primary = False
        self._segmentation_masks_for_current_slice: dict[str, np.ndarray] | None = None
        # Which structures the user wants overlaid - a whole-volume-scoped
        # choice (per explicit request), not a per-slice one: surviving
        # slice navigation is the point, and it's only cleared (back to
        # "all selected", its default - see below) in `_reset_segmentation_
        # display` (close study / new volume load). Lives here, not on the
        # (optional, lazily-created) selection dialog, since it must
        # persist even while that dialog is closed.
        # Defaults to every structure selected (per explicit request) - the
        # "Select Segments…" dialog is for narrowing down which ones show,
        # not for opting in from nothing; `_show_segments_cb` ("Show
        # Segments") is the separate, single switch for whether *any* of
        # them are drawn at all.
        self._selected_segmentation_names: set[str] = set(_SEGMENTATION_MAPPING)
        self._segment_selection_dialog: SegmentationSelectionDialog | None = None
        # Overlay opacity (0.0-1.0), set from the "Select Segments…" dialog's
        # vertical slider - applies to every currently-shown structure at
        # once, not per-structure. Default matches `RoiViewport.set_overlay`'s
        # own prior hardcoded default (0.6), so introducing the slider
        # doesn't change anything until the user actually moves it.
        self._segmentation_opacity: float = 0.6
        self._stroke_mask_for_current_slice: np.ndarray | None = None

        self._build_menu_bar()

        central = QWidget()
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(1)
        content_layout.addWidget(self._build_viewer_pane(), 3)
        content_layout.addWidget(self._build_sidebar(), 0)
        outer_layout.addWidget(content, 1)

        outer_layout.addWidget(self._build_trace_bar())

        self.setCentralWidget(central)

        self._log_handler = QtLogHandler()
        self._log_handler.message_logged.connect(self._append_trace_line)
        configure_logging(extra_handlers=[self._log_handler])
        logger.info("BICTool window ready.")

    # ---- trace / log panel ----

    def _build_trace_bar(self) -> QWidget:
        box = QGroupBox("Trace / Log")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        self._trace_log = QPlainTextEdit()
        self._trace_log.setReadOnly(True)
        self._trace_log.setMaximumBlockCount(5000)  # cap memory growth on a long session
        self._trace_log.setFixedHeight(160)
        self._trace_log.setStyleSheet(
            f"font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 10px; "
            f"background: {SURFACE_SUNKEN}; color: {TEXT}; border: 1px solid {BORDER};"
        )
        layout.addWidget(self._trace_log)
        return box

    def _append_trace_line(self, message: str) -> None:
        self._trace_log.appendPlainText(message)

    # ---- menu bar ----

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        self._open_folder_action = file_menu.addAction(
            "Open DICOM Folder (Series)…", self._on_open_dicom_folder_clicked
        )
        self._open_file_action = file_menu.addAction(
            "Open DICOM File (Single Slice)…", self._on_open_dicom_file_clicked
        )

        # Not spec'd by any FR/NFR - scoped deliberately small (per user
        # request) to two study-level actions distinct from File's
        # open/import and View's display controls. A batch "study-level
        # quality score" (FR-04's currently-unmet second half) was discussed
        # and explicitly deferred to a separate task - see project memory.
        study_menu = menu_bar.addMenu("&Study")
        self._close_study_action = study_menu.addAction("Close Study", self._on_close_study_clicked)
        self._close_study_action.setEnabled(False)
        self._close_study_action.setToolTip("Open a study first")
        self._reload_series_action = study_menu.addAction("Reload Series", self._on_reload_series_clicked)
        self._reload_series_action.setEnabled(False)
        self._reload_series_action.setToolTip("Open a study first")

        view_menu = menu_bar.addMenu("&View")
        self._view_tags_action = view_menu.addAction("DICOM Tags…", self._on_view_dicom_tags_clicked)
        self._view_tags_action.setEnabled(False)
        self._view_tags_action.setToolTip("Open a study first")

        # Configuration/adjustment dialogs - distinct from View's display
        # toggles and File's I/O. None of these actions need a study-loaded
        # gate except Custom Window, which is disabled/re-enabled the same
        # way the viewer toolbar's own preset dropdown already is (see
        # `_show_view_page`) - the settings dialogs and the method choice
        # below are plain numeric-parameters forms/preferences, usable
        # (and worth setting up) before any study is even opened.
        tools_menu = menu_bar.addMenu("&Tools")

        # One radio-style submenu picks which of the three WM/GM
        # classification methods is active; each method's own tunable
        # parameters live in its own settings dialog, listed right below it
        # here rather than inside the submenu itself (a checkable action and
        # a "open a dialog" action are different enough interactions that
        # mixing them in one menu would be confusing).
        wm_gm_menu = tools_menu.addMenu("WM/GM Classification")
        self._brain_tissue_method_actions: dict[str, QAction] = {}
        method_group = QActionGroup(self)
        method_group.setExclusive(True)
        for method in _BRAIN_TISSUE_METHODS:
            action = wm_gm_menu.addAction(method)
            action.setCheckable(True)
            action.setChecked(method == self._brain_tissue_method)
            action.triggered.connect(lambda checked, m=method: self._on_brain_tissue_method_selected(m))
            method_group.addAction(action)
            self._brain_tissue_method_actions[method] = action

        tools_menu.addAction("HU Thresholding Settings…", self._on_brain_tissue_settings_clicked)
        tools_menu.addAction("Fuzzy C-Means Settings…", self._on_fcm_settings_clicked)
        tools_menu.addAction("Adaptive Threshold Settings…", self._on_adaptive_settings_clicked)
        tools_menu.addSeparator()
        self._custom_window_action = tools_menu.addAction("Custom Window…", self._on_custom_window_clicked)
        self._custom_window_action.setEnabled(False)
        self._custom_window_action.setToolTip("Open a study first")

        export_menu = menu_bar.addMenu("&Export")
        self._export_action = export_menu.addAction("Export Results…", self._open_export_dialog)
        # Disabled rather than a silent no-op click - a clickable-looking
        # control that does nothing when clicked is an error-prevention/
        # visibility-of-system-status problem (NFR-07) - stays disabled
        # until Run All has actually produced a result to export.
        self._export_action.setEnabled(False)
        self._export_action.setToolTip("Run metrics first")

    def _on_close_study_clicked(self) -> None:
        logger.info("Study closed by user")
        self._loaded_volume = None
        self._series_candidates = []
        # A future study's primary series is unrelated to this one's - don't
        # let its saved segmentation state leak into the next study opened.
        self._primary_series_segmentation_cache = None
        self._primary_series_dicom_nifti_volume = None
        self._current_series_is_primary = False
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        self._series_combo.setEnabled(False)
        self._series_combo.blockSignals(False)

        self._roi_viewport.clear()  # also clears every seg_*/stroke_lesion overlay - see RoiViewport.clear_all_overlays
        self._roi_rects = {}
        self._manual_rois = {}
        self._reset_roi_table()
        self._confirm_rois_cb.setChecked(False)
        self._confirm_rois_cb.setEnabled(False)
        self._set_roi1_btn.setEnabled(False)
        self._set_roi2_btn.setEnabled(False)

        self._reset_segmentation_display()
        self._reset_stroke_display()
        self._reset_brain_tissue_display()

        self._slice_slider.setEnabled(False)
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(0)
        self._slice_slider.blockSignals(False)
        self._slice_label.setText("No slices loaded")
        self._study_status_label.setText("No study loaded — File → Open DICOM Folder/File…")

        self._reset_qa_results()

        self._close_study_action.setEnabled(False)
        self._close_study_action.setToolTip("Open a study first")
        self._reload_series_action.setEnabled(False)
        self._reload_series_action.setToolTip("Open a study first")
        self._view_tags_action.setEnabled(False)
        self._view_tags_action.setToolTip("Open a study first")
        self._custom_window_action.setEnabled(False)
        self._custom_window_action.setToolTip("Open a study first")

        self._show_view_page(0)

    def _on_reload_series_clicked(self) -> None:
        if not self._series_candidates:
            return
        index = self._series_combo.currentIndex()
        if index < 0:
            index = 0
        logger.info("Reloading series from disk: %s", self._series_candidates[index].label)
        self._load_selected_series(index, force_reload=True)

    def _on_view_dicom_tags_clicked(self) -> None:
        if self._loaded_volume is None:
            return
        path = self._loaded_volume.slice_paths[self._current_slice_index]
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            logger.exception("Failed to read DICOM tags from %s", path)
            QMessageBox.critical(self, "Failed to read DICOM tags", str(exc))
            return
        title = f"DICOM Tags — Slice {self._current_slice_index + 1} / {self._loaded_volume.num_slices}"
        DicomTagsDialog(dataset, title=title, parent=self).exec()

    def _on_brain_tissue_settings_clicked(self) -> None:
        dialog = BrainTissueSettingsDialog(self._brain_tissue_settings, self._brain_tissue_default_settings, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._brain_tissue_settings = dialog.settings()
        # Every already-computed result was made with the old parameters -
        # discard it so the next checkbox check/slice view recomputes fresh,
        # exactly like `_reset_brain_tissue_display` does on a new study.
        self._brain_tissue_cache = {}
        self._refresh_brain_tissue_for_current_slice()
        self._restart_auto_roi2()

    def _open_export_dialog(self) -> None:
        if self._last_result is None or self._last_export_context is None:
            return
        # Opened from the Study summary tab -> whole study preselected.
        dialog = ExportDialog(
            study_available=bool(self._qa_results),
            default_scope=SCOPE_STUDY if self._showing_study_summary() else "slice",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.scope() == SCOPE_STUDY:
            self._perform_study_export(dialog.selected_items(), dialog.destination())
        else:
            self._perform_export(dialog.selected_items(), dialog.destination())

    @staticmethod
    def _safe_file_stem(series_label: str) -> str:
        # Filesystem-safe stem - the series label has "·"/"×" and spaces
        # from its human-readable form (see `dicom_file_loader._build_candidate`);
        # collapsed runs of "_" (one per replaced character otherwise) so
        # e.g. "CT · 1 slice(s) · 512×434" doesn't become "CT___1_slice_s_..._".
        return re.sub(r"_+", "_", "".join(c if c.isalnum() else "_" for c in series_label)).strip("_")

    def _perform_study_export(self, items: set[str], destination: Path) -> None:
        """Study averages + one row per slice, from every QA result (failed
        slices listed, not averaged). Chart images are per slice - not here.
        """
        if not self._qa_results or self._loaded_volume is None:
            return
        context = StudyExportContext(
            series_label=self._series_combo.currentText() or "Series",
            slice_count=self._loaded_volume.num_slices,
            pixel_spacing_mm=self._loaded_volume.pixel_spacing_mm,
            acquisition_date=self._loaded_volume.acquisition_date,
            failed_slices=[index + 1 for index in sorted(self._qa_errors)],
        )
        stem = f"{self._safe_file_stem(context.series_label)}_study"
        try:
            destination.mkdir(parents=True, exist_ok=True)
            written: list[Path] = []
            if "excel" in items:
                path = destination / f"{stem}_metrics.xlsx"
                export_study_excel(self._qa_results, context, path)
                written.append(path)
            if "csv" in items:
                summary_path = destination / f"{stem}_summary.csv"
                slices_path = destination / f"{stem}_slices.csv"
                export_study_csv(self._qa_results, context, summary_path, slices_path)
                written += [summary_path, slices_path]
        except OSError as exc:
            logger.exception("Study export failed")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        logger.info("Exported study (%d slice(s)) to %d file(s) in %s", len(self._qa_results), len(written), destination)
        QMessageBox.information(self, "Export complete", f"Wrote {len(written)} file(s) to:\n{destination}")

    def _perform_export(self, items: set[str], destination: Path) -> None:
        result = self._last_result
        context = self._last_export_context
        stem = f"{self._safe_file_stem(context.series_label)}_slice{context.slice_number:03d}"
        try:
            destination.mkdir(parents=True, exist_ok=True)
            written: list[Path] = []
            if "excel" in items:
                path = destination / f"{stem}_metrics.xlsx"
                export_metrics_excel(result, context, path)
                written.append(path)
            if "csv" in items:
                path = destination / f"{stem}_metrics.csv"
                export_metrics_csv(result, context, path)
                written.append(path)
            if "jpg" in items:
                charts = [
                    ("mtf", self._mtf_chart_view),
                    ("nps", self._nps_chart_view),
                    ("ps", self._ps_chart_view),
                    ("ratio", self._ratio_chart_view),
                ]
                for name, chart_view in charts:
                    path = destination / f"{stem}_{name}.jpg"
                    chart_view.grab().save(str(path), "JPG")
                    written.append(path)
        except OSError as exc:
            logger.exception("Export failed")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        logger.info("Exported %d file(s) to %s", len(written), destination)
        QMessageBox.information(self, "Export complete", f"Wrote {len(written)} file(s) to:\n{destination}")

    def _on_open_dicom_folder_clicked(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open DICOM series folder")
        if path:
            self._load_volume_from(path)

    def _on_open_dicom_file_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM file", "", "DICOM files (*.dcm);;All files (*)")
        if path:
            self._load_volume_from(path)

    def _load_volume_from(self, path: str) -> None:
        # A recursive folder scan is real disk I/O over potentially hundreds
        # of files - run it on a background thread (like Worker does for
        # evaluate_slice_quality) so the window stays responsive instead of
        # freezing for however long the scan takes.
        logger.info("Open DICOM: %s", path)
        self._open_folder_action.setEnabled(False)
        self._open_file_action.setEnabled(False)
        self._study_status_label.setText("Loading DICOM data…")

        loader = DicomLoadWorker(path)
        loader.signals.finished.connect(self._on_series_discovered)
        loader.signals.error.connect(self._on_volume_load_error)
        loader.signals.progress.connect(self._on_volume_load_progress)
        self._active_dicom_load_worker = loader  # keep a reference alive while it runs
        self._scheduler.submit(loader)

    def _on_volume_load_progress(self, files_scanned: int) -> None:
        self._study_status_label.setText(f"Loading DICOM data… ({files_scanned} files scanned)")

    def _on_series_discovered(self, candidates: list[SeriesCandidate]) -> None:
        logger.info("Discovered %d series candidate(s): %s", len(candidates), [c.label for c in candidates])
        self._active_dicom_load_worker = None
        self._open_folder_action.setEnabled(True)
        self._open_file_action.setEnabled(True)

        # A fresh study discovery - even without an explicit "Close Study"
        # first (e.g. opening a new folder straight over an already-loaded
        # one) - means whatever's backed up as "the primary series'"
        # segmentation belongs to a study that's going away; without this,
        # `_on_volume_loaded`'s restore-on-return-to-primary logic below
        # could otherwise resurrect a *previous, unrelated* study's cached
        # results under the new study's own primary series.
        self._primary_series_segmentation_cache = None
        self._primary_series_dicom_nifti_volume = None
        self._current_series_is_primary = False

        self._series_candidates = candidates
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        self._series_combo.addItems([c.label for c in candidates])
        self._series_combo.blockSignals(False)
        # Only meaningful to interact with when there's an actual choice -
        # with one candidate it just documents what was found.
        self._series_combo.setEnabled(len(candidates) > 1)

        self._load_selected_series(0)

    def _on_series_combo_changed(self, index: int) -> None:
        if index < 0:  # combo cleared/being repopulated
            return
        logger.info("Series selection changed to: %s", self._series_candidates[index].label)
        self._load_selected_series(index)

    def _load_selected_series(self, index: int, *, force_reload: bool = False) -> None:
        volume = load_series(self._series_candidates[index])
        # Index 0 is always `discover_series`'s own best-guess default (CT
        # modality, most slices - see this module's docstring) - the one
        # auto-loaded when a study is first opened, and the only one
        # `_maybe_auto_segment_volume` ever triggers for automatically, per
        # explicit request - any other series (picked from the combo, or a
        # reload of one) only segments on an explicit Segment Slice/Volume
        # click, even if the user later navigates back to index 0.
        self._on_volume_loaded(volume, is_primary_series=(index == 0), force_reload=force_reload)

    def _on_volume_loaded(
        self, volume: DicomVolume, *, is_primary_series: bool = True, force_reload: bool = False
    ) -> None:
        logger.info(
            "Volume loaded: %d slice(s), patient=%s, pixel_spacing=%.4f mm",
            volume.num_slices,
            volume.patient_id,
            volume.pixel_spacing_mm,
        )
        self._active_dicom_load_worker = None
        self._open_folder_action.setEnabled(True)
        self._open_file_action.setEnabled(True)
        self._close_study_action.setEnabled(True)
        self._close_study_action.setToolTip("")
        self._reload_series_action.setEnabled(True)
        self._reload_series_action.setToolTip("Re-reads the current series from disk")
        self._view_tags_action.setEnabled(True)
        self._view_tags_action.setToolTip("")
        self._custom_window_action.setEnabled(True)
        self._custom_window_action.setToolTip("")

        # A different volume (new series, reload, or a fresh open) means any
        # cached segmentation/stroke results (keyed by slice index) belong
        # to slices that no longer exist under these indices - clear before
        # `_show_slice` below, or a stale result could silently reappear
        # under the new volume's same slice index. Before clearing, if the
        # series we're LEAVING was the primary one, preserve its
        # segmentation state (not stroke - not asked for) so switching back
        # to it later restores rather than re-segmenting from scratch (per
        # explicit request - primary series only, not every series visited,
        # to avoid keeping every series' full NIfTI volume + masks resident
        # at once).
        if self._current_series_is_primary and self._loaded_volume is not None:
            self._primary_series_segmentation_cache = self._segmentation_cache
            self._primary_series_dicom_nifti_volume = self._dicom_nifti_volume
        self._reset_segmentation_display()
        self._reset_stroke_display()
        self._reset_brain_tissue_display()
        self._manual_rois = {}
        self._confirm_rois_cb.setChecked(False)
        self._reset_qa_results()
        self._update_segmentation_availability()
        self._current_series_is_primary = is_primary_series

        if is_primary_series and force_reload:
            # The underlying file(s) may have changed - the previous cache
            # (if any) is no longer trustworthy.
            self._primary_series_segmentation_cache = None
            self._primary_series_dicom_nifti_volume = None
        elif is_primary_series and self._primary_series_segmentation_cache is not None:
            # Returning to the primary series after visiting another one -
            # restore instead of starting empty. `_maybe_auto_segment_volume`
            # below still runs, but `_on_segment_volume_clicked` already
            # no-ops ("Done") once every slice is present, or resumes just
            # the slices that weren't finished yet if the batch was
            # cancelled mid-way by switching away earlier.
            self._segmentation_cache = self._primary_series_segmentation_cache
            self._dicom_nifti_volume = self._primary_series_dicom_nifti_volume

        self._loaded_volume = volume
        self._current_slice_index = 0
        self._slice_slider.setMinimum(0)
        self._slice_slider.setMaximum(volume.num_slices - 1)
        self._slice_slider.setEnabled(volume.num_slices > 1)
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(0)
        self._slice_slider.blockSignals(False)
        # Set ROI1/2 stay disabled until a rectangle is actually drawn (see
        # `_on_draft_changed`) - `_show_slice` below triggers that via
        # `RoiViewport.set_image`'s `draft_changed.emit(None)`.
        self._show_slice(0)  # also refreshes the corner overlay (see `_update_viewer_overlay`)
        self._show_view_page(0)

        if is_primary_series:
            self._maybe_auto_segment_volume()

    def _maybe_auto_segment_volume(self) -> None:
        """Automatically starts a background whole-volume segmentation for
        the study's primary series (per explicit request) - the user
        shouldn't have to click Segment Volume for the main scan. Only ever
        called for the primary series (see `_load_selected_series`); any
        other series (picked from the "Series" combo, or reloaded) still
        requires an explicit Segment Slice/Volume click.

        Silently does nothing if anatomy weights aren't configured or
        TensorFlow isn't installed - this is a convenience auto-start, not
        a user-initiated action, so it must never pop up an error dialog on
        its own the way the button's own click handler does. A genuinely
        broken *configured* weights path still surfaces the usual dialog
        via `_on_segment_volume_clicked` below - that's an actionable
        misconfiguration, not an expected "nothing set up yet" state.
        """
        weights_path = self._segmentation_config.get("segmentation", {}).get("anatomy_weights_path")
        if not weights_path:
            return
        try:
            # Deferred to first actual use - see the TYPE_CHECKING import
            # block's comment for why (TensorFlow's import cost).
            from bictool.domain.segmentation import head_ct_unet  # noqa: F401
        except ImportError:
            return
        self._on_segment_volume_clicked()

    def _on_volume_load_error(self, message: str) -> None:
        logger.error("Failed to open DICOM data: %s", message)
        self._active_dicom_load_worker = None
        self._open_folder_action.setEnabled(True)
        self._open_file_action.setEnabled(True)
        self._study_status_label.setText("No study loaded — File → Open DICOM Folder/File…")
        QMessageBox.critical(self, "Failed to open DICOM data", message)

    def _show_slice(self, index: int) -> None:
        self._current_slice_index = index
        self._slice_label.setText(f"Slice {index + 1} / {self._loaded_volume.num_slices}")
        self._roi_viewport.set_image(self._loaded_volume.get_slice_hu(index))  # also clears every overlay
        self._refresh_rois_for_current_slice()
        self._refresh_segmentation_for_current_slice()
        self._refresh_stroke_for_current_slice()
        self._refresh_brain_tissue_for_current_slice()
        self._show_auto_rois_for_current_slice()
        self._refresh_metrics_box()
        if index in self._results_slices:
            self._show_result_for_slice(index)
        self._update_viewer_overlay()  # top-left corner shows the current slice number

    def _on_viewport_slice_scroll(self, direction: int) -> None:
        new_index = self._slice_slider.value() + direction
        new_index = max(self._slice_slider.minimum(), min(self._slice_slider.maximum(), new_index))
        self._slice_slider.setValue(new_index)

    def _on_window_preset_clicked(self, preset_name: str) -> None:
        width, center = WINDOW_PRESETS[preset_name]
        self._roi_viewport.set_window(width, center)
        self._update_viewer_overlay()  # bottom-right corner shows the current WL/WW

    def _on_custom_window_clicked(self) -> None:
        if self._loaded_volume is None:
            return
        current_width, current_center = self._roi_viewport.current_window()
        dialog = CustomWindowDialog(current_width, current_center, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        width, center = dialog.window()
        self._roi_viewport.set_window(width, center)
        self._update_viewer_overlay()  # bottom-right corner shows the current WL/WW

    # ---- left pane: viewer / results ----

    def _build_viewer_pane(self) -> QWidget:
        pane = QWidget()
        outer = QVBoxLayout(pane)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        toolbar = QHBoxLayout()
        self._viewer_tab_btn = QPushButton("Viewer")
        self._viewer_tab_btn.setObjectName("navTabBtn")
        self._viewer_tab_btn.setCheckable(True)
        self._viewer_tab_btn.setChecked(True)
        self._results_tab_btn = QPushButton("Results")
        self._results_tab_btn.setObjectName("navTabBtn")
        self._results_tab_btn.setCheckable(True)
        self._results_tab_btn.setEnabled(False)
        self._results_tab_btn.setToolTip("Run metrics to see results")
        self._viewer_tab_btn.clicked.connect(lambda: self._show_view_page(0))
        self._results_tab_btn.clicked.connect(lambda: self._show_view_page(1))
        toolbar.addWidget(self._viewer_tab_btn)
        toolbar.addWidget(self._results_tab_btn)
        toolbar.addStretch(1)

        # Windowing only makes sense while looking at the image, not the
        # Results page - wrapped in one widget so `_show_view_page` can hide
        # it. (Set ROI1/2 live in the sidebar's ROI Selection box instead -
        # see `_build_roi_box` - since they're an ROI-selection action, not
        # a viewing control.)
        self._viewer_controls_widget = QWidget()
        controls_layout = QHBoxLayout(self._viewer_controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        controls_layout.addWidget(QLabel("Window:"))
        self._window_preset_combo = QComboBox()
        self._window_preset_combo.addItems(list(WINDOW_PRESETS))
        self._window_preset_combo.setCurrentText(DEFAULT_WINDOW_PRESET)
        self._window_preset_combo.currentTextChanged.connect(self._on_window_preset_clicked)
        controls_layout.addWidget(self._window_preset_combo)
        toolbar.addWidget(self._viewer_controls_widget)
        outer.addLayout(toolbar)

        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self._build_viewer_page())
        self._view_stack.addWidget(self._build_results_page())
        outer.addWidget(self._view_stack, 1)

        return pane

    def _show_view_page(self, index: int) -> None:
        self._view_stack.setCurrentIndex(index)
        self._viewer_tab_btn.setChecked(index == 0)
        self._results_tab_btn.setChecked(index == 1)
        # Windowing/ROI-selection/segmentation controls, and the ability to
        # (re-)run metrics, only make sense while looking at the image
        # (index 0) - there's no ROI to run against from the Results page.
        # The metric status/result rows themselves stay visible on both.
        is_viewer_page = index == 0
        self._viewer_controls_widget.setVisible(is_viewer_page)
        self._segmentation_roi_widget.setVisible(is_viewer_page)
        self._run_all_btn.setVisible(is_viewer_page)
        self._refresh_metrics_box()

    def _build_viewer_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self._roi_viewport = RoiViewport()
        self._roi_viewport.setMinimumHeight(420)
        self._roi_viewport.draft_changed.connect(self._on_draft_changed)
        self._roi_viewport.draft_moved.connect(self._on_draft_moved)
        self._roi_viewport.cursor_moved.connect(self._on_cursor_moved)
        self._roi_viewport.slice_scroll.connect(self._on_viewport_slice_scroll)
        # Zoom% is part of the bottom-right corner overlay - keep it live as
        # the user actually zooms (Ctrl+wheel, double-click reset, or a
        # window resize re-fitting the image), not just at load time.
        self._roi_viewport.zoom_changed.connect(lambda _percent: self._update_viewer_overlay())
        layout.addWidget(self._roi_viewport, 1)

        slice_row = QHBoxLayout()
        self._slice_label = QLabel("No slices loaded")
        self._slice_label.setProperty("role", "mono")
        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setEnabled(False)
        self._slice_slider.valueChanged.connect(self._show_slice)
        slice_row.addWidget(self._slice_label)
        slice_row.addWidget(self._slice_slider, 1)
        layout.addLayout(slice_row)

        self._study_status_label = QLabel("No study loaded — File → Open DICOM Folder/File…")
        self._study_status_label.setProperty("role", "muted")
        layout.addWidget(self._study_status_label)

        return page

    def _on_draft_changed(self, rect: RoiRect | None) -> None:
        """A rectangle was just drawn (or cleared, e.g. by a slice change) -
        drawing no longer directly produces ROI1/ROI2 (see `RoiViewport`'s
        draw-then-assign model); it just makes Set ROI1/Set ROI2 available
        to save it. Also drives the live "Selecting: WxH px" line in the
        image's bottom-right corner overlay while dragging.
        """
        self._has_hand_drawn_draft = rect is not None
        self._update_set_roi_buttons()
        self._interaction_line = f"Selecting: {rect.width}×{rect.height} px" if rect is not None else ""
        self._update_viewer_overlay()

    def _on_cursor_moved(self, reading: tuple[int, int, float] | None) -> None:
        """Live "what's under the cursor" readout, shown the same way real
        DICOM viewers do - in the image's bottom-right corner overlay,
        alongside WL/WW/zoom (`_update_viewer_overlay`). `None` while
        actively drawing a rectangle (see `RoiViewport.mouseMoveEvent`, which
        only emits one of `draft_changed`/`cursor_moved` at a time) or after
        the cursor leaves the image/viewport.
        """
        if reading is None:
            self._interaction_line = ""
        else:
            x, y, value = reading
            self._interaction_line = f"X: {x}  Y: {y}  Val: {value:.0f} HU"
        self._update_viewer_overlay()

    def _on_set_roi_clicked(self, key: str) -> None:
        rect = self._roi_viewport.commit_current_as(key)
        if rect is None:
            return  # button is disabled whenever there's nothing to commit - defensive only
        logger.info("%s set: x=%d-%d, y=%d-%d (%dx%d)", key, rect.x0, rect.x1, rect.y0, rect.y1, rect.width, rect.height)
        self._manual_rois.setdefault(self._current_slice_index, {})[key] = rect
        self._refresh_rois_for_current_slice()
        # Redrawing/re-saving an ROI after confirmation should force a fresh review.
        self._confirm_rois_cb.setChecked(False)
        self._update_confirm_availability()

    def _update_confirm_availability(self) -> None:
        # One confirmation covers every slice: available as soon as any slice
        # has both ROIs (and no QA batch is running).
        ready = self._loaded_volume is not None and bool(self._qa_ready_slices()) and self._qa_jobs is None
        self._confirm_rois_cb.setEnabled(ready or self._confirm_rois_cb.isChecked())

    def _effective_rois(self, slice_index: int) -> dict[str, RoiRect]:
        """A slice's ROIs: hand-set ones (Set ROI1/ROI2) over automatic ones."""
        rois = {}
        auto_on = self._auto_rois_cb.isChecked()
        for key, candidates in (("roi1", self._auto_roi1_candidates), ("roi2", self._auto_roi2_candidates)):
            manual = self._manual_rois.get(slice_index, {}).get(key)
            if manual is not None:
                rois[key] = manual
            elif auto_on and slice_index in candidates:
                rois[key] = _candidate_to_rect(candidates[slice_index])
        return rois

    def _qa_ready_slices(self) -> list[int]:
        """Every slice with both ROIs - what one confirmation + Run All covers."""
        slices = set(self._manual_rois) | set(self._auto_roi1_candidates)
        return sorted(index for index in slices if len(self._effective_rois(index)) == 2)

    def _refresh_rois_for_current_slice(self) -> None:
        """Shows the displayed slice's ROIs on the image and in the Position/Size table."""
        rois = self._effective_rois(self._current_slice_index) if self._loaded_volume is not None else {}
        self._roi_rects = rois
        self._roi_viewport.clear_rois()
        for key in ("roi1", "roi2"):
            if key in rois:
                self._roi_viewport.set_roi(key, rois[key])
                self._update_roi_table_row(key, rois[key])
            else:
                self._reset_roi_table_row(key)
        self._update_confirm_availability()
        self._update_set_roi_buttons()

    # ---- auto ROIs (domain.roi_selection: structured_roi -> ROI1, homogeneous_roi -> ROI2) ----

    def _auto_roi1_structure(self) -> str:
        return SEPTUM_PELLUCIDUM if self._auto_roi1_septum_rb.isChecked() else DURAL_FOLDS

    def _update_auto_rois_availability(self) -> None:
        confirmed = self._confirm_rois_cb.isChecked()
        available = self._loaded_volume is not None and bool(self._segmentation_cache) and not confirmed
        self._auto_rois_cb.setEnabled(available)
        self._auto_roi1_septum_rb.setEnabled(not confirmed)
        self._auto_roi1_dural_rb.setEnabled(not confirmed)
        if confirmed:
            self._auto_rois_cb.setToolTip("ROIs are confirmed - uncheck Confirmed to change them")
        else:
            self._auto_rois_cb.setToolTip("" if available else "Needs anatomy segmentation first")

    def _segmentation_masks_by_slice(self) -> dict[int, dict[str, np.ndarray]]:
        return {index: result.masks for index, result in self._segmentation_cache.items()}

    def _compute_auto_roi1_for_slice(self, slice_index: int) -> None:
        result = self._segmentation_cache.get(slice_index)
        if result is None:
            return
        candidate = find_roi1_candidate(
            result.masks,
            self._auto_roi1_structure(),
            self._loaded_volume.pixel_spacing_mm,
            midline_direction=self._auto_midline,
            image=self._loaded_volume.get_slice_hu(slice_index),
        )
        if candidate is None:
            self._auto_roi1_candidates.pop(slice_index, None)
            self._auto_roi2_candidates.pop(slice_index, None)
            self._auto_roi2_moved.discard(slice_index)
            return
        self._auto_roi1_candidates[slice_index] = candidate
        self._compute_auto_roi2_for_slice(slice_index)

    def _compute_auto_roi2_for_slice(self, slice_index: int) -> None:
        """ROI2 next to this slice's ROI1. Stays pending (absent) while it needs
        white matter and the WM/GM classification for this slice hasn't
        arrived from `_start_auto_tissue_worker` yet. A ROI2 the user moved
        by hand is kept as is.
        """
        if slice_index in self._auto_roi2_moved:
            return
        roi1 = self._auto_roi1_candidates.get(slice_index)
        if roi1 is None:
            self._auto_roi2_candidates.pop(slice_index, None)
            return
        tissue = self._brain_tissue_cache.get(slice_index)
        candidate = find_roi2_candidate(
            roi1,
            self._segmentation_cache[slice_index].masks,
            self._loaded_volume.pixel_spacing_mm,
            None if tissue is None else tissue.masks,
        )
        if candidate is None:
            self._auto_roi2_candidates.pop(slice_index, None)
        else:
            self._auto_roi2_candidates[slice_index] = candidate

    def _recompute_auto_rois(self) -> None:
        """ROI1 (and ROI2 where possible right away) for every segmented slice,
        not just the displayed one - the per-slice groundwork for local and
        study-level QA. White-matter ROI2s follow as the background WM/GM
        classification lands.
        """
        self._stop_auto_tissue_worker()
        self._auto_roi1_candidates = {}
        self._auto_roi2_candidates = {}
        self._auto_roi2_moved = set()
        self._auto_midline = estimate_midline_direction(self._segmentation_masks_by_slice())
        for slice_index in self._segmentation_cache:
            self._compute_auto_roi1_for_slice(slice_index)
        logger.info(
            "Auto ROIs (%s): ROI1 on %d of %d segmented slice(s), ROI2 ready on %d",
            self._auto_roi1_structure(),
            len(self._auto_roi1_candidates),
            len(self._segmentation_cache),
            len(self._auto_roi2_candidates),
        )
        self._start_auto_tissue_worker()

    def _start_auto_tissue_worker(self) -> None:
        """Classifies WM/GM (the method selected in Tools) in the background for
        every slice whose ROI2 is waiting on white matter. If one is already
        running, its `finished` handler calls this again to pick up slices
        that became pending meanwhile.
        """
        if self._active_auto_tissue_worker is not None or self._loaded_volume is None:
            return
        pending = [
            index
            for index in sorted(self._auto_roi1_candidates)
            if index not in self._auto_roi2_candidates and index not in self._brain_tissue_cache
        ]
        if not pending:
            return
        classify = self._tissue_classifier()
        brain_masks = brain_mask_from_segmentation(self._segmentation_masks_by_slice(), self._loaded_volume.num_slices)
        get_slice_hu = self._loaded_volume.get_slice_hu
        worker = BrainTissueVolumeWorker(lambda index: classify(get_slice_hu(index), brain_masks[index]), pending)
        worker.signals.slice_finished.connect(self._on_auto_tissue_slice_finished)
        worker.signals.finished.connect(self._on_auto_tissue_finished)
        worker.signals.error.connect(self._on_auto_tissue_error)
        self._active_auto_tissue_worker = worker  # keep a reference alive while it runs
        logger.info("Auto ROIs: classifying WM/GM for %d slice(s) in the background", len(pending))
        self._scheduler.submit(worker)

    def _on_auto_tissue_slice_finished(self, slice_index: int, result) -> None:
        self._brain_tissue_cache.setdefault(slice_index, result)
        if not self._auto_rois_cb.isChecked():
            return
        self._compute_auto_roi2_for_slice(slice_index)
        if slice_index == self._current_slice_index:
            self._show_auto_rois_for_current_slice()
        else:
            self._update_auto_rois_status()

    def _on_auto_tissue_finished(self) -> None:
        self._active_auto_tissue_worker = None
        if self._auto_rois_cb.isChecked():
            self._start_auto_tissue_worker()

    def _on_auto_tissue_error(self, slice_index: int, message: str) -> None:
        logger.error("Auto ROIs: WM/GM classification failed on slice %d: %s", slice_index + 1, message)
        self._active_auto_tissue_worker = None
        self._auto_rois_status_label.setText(f"WM/GM classification failed on slice {slice_index + 1}: {message}")

    def _stop_auto_tissue_worker(self) -> None:
        # cancel() only stops future slices; blockSignals silences the one
        # already running (same pattern as the segmentation workers).
        worker = self._active_auto_tissue_worker
        if worker is not None:
            worker.cancel()
            worker.signals.blockSignals(True)
            self._active_auto_tissue_worker = None

    def _restart_auto_roi2(self) -> None:
        """The WM/GM method or its settings changed: white-matter ROI2s are
        stale. Ventricle ROI2s and hand-moved ones stay.
        """
        if not self._auto_rois_cb.isChecked() or self._confirm_rois_cb.isChecked():
            return  # confirmed ROIs are locked
        self._stop_auto_tissue_worker()
        for slice_index in list(self._auto_roi1_candidates):
            self._compute_auto_roi2_for_slice(slice_index)
        self._start_auto_tissue_worker()
        self._show_auto_rois_for_current_slice()

    def _show_auto_rois_for_current_slice(self) -> None:
        if self._loaded_volume is None:
            return
        self._refresh_rois_for_current_slice()
        if self._auto_rois_cb.isChecked():
            self._update_auto_rois_status()

    def _update_auto_rois_status(self) -> None:
        name = "septum pellucidum" if self._auto_roi1_structure() == SEPTUM_PELLUCIDUM else "dural folds"
        slices = sorted(self._auto_roi1_candidates)
        if not slices:
            self._auto_rois_status_label.setText(f"No {name} ROI1 candidate on any segmented slice.")
            return
        roi2 = self._auto_roi2_candidates.get(self._current_slice_index)
        if self._current_slice_index not in self._auto_roi1_candidates:
            here = f"No {name} on this slice."
        elif roi2 is None:
            here = "ROI2 here: waiting for WM/GM classification…"
        else:
            here = f"ROI2 here: {_ROI2_REGION_LABELS[roi2.structure]}."
        waiting = sum(1 for index in slices if index not in self._auto_roi2_candidates)
        waiting_text = f", {waiting} waiting for WM/GM" if waiting else ""
        self._auto_rois_status_label.setText(
            f"{here} ROI1 on {len(slices)} slice(s): {_format_slice_ranges(slices)}{waiting_text}."
        )

    def _on_auto_rois_toggled(self, checked: bool) -> None:
        if checked:
            self._recompute_auto_rois()
            self._show_auto_rois_for_current_slice()
        else:
            self._clear_auto_rois()
            self._refresh_rois_for_current_slice()

    def _on_auto_roi1_structure_toggled(self, _button, checked: bool) -> None:
        if not checked or not self._auto_rois_cb.isChecked():
            return  # the group fires once for the unchecked button too - act only on the newly checked one
        self._recompute_auto_rois()
        self._show_auto_rois_for_current_slice()

    def _on_auto_rois_segmentation_landed(self, slice_index: int) -> None:
        """A slice just got segmented: Auto ROIs may become available, and
        while they're on, that slice gets its ROIs right away.
        """
        self._update_auto_rois_availability()
        if not self._auto_rois_cb.isChecked():
            return
        self._auto_midline = estimate_midline_direction(self._segmentation_masks_by_slice())
        self._compute_auto_roi1_for_slice(slice_index)
        self._start_auto_tissue_worker()
        if slice_index == self._current_slice_index:
            self._show_auto_rois_for_current_slice()
        else:
            self._update_auto_rois_status()

    def _on_draft_moved(self, key: str, rect: RoiRect) -> None:
        """The user dragged a committed ROI (only possible while not
        confirmed): remember it for this slice. Moving an automatic ROI1
        re-places an automatic ROI2 around it (unless ROI2 was itself moved
        by hand). A hand-drawn draft ("") isn't an ROI yet.
        """
        index = self._current_slice_index
        if not key:
            return
        manual = self._manual_rois.get(index, {})
        if key in manual:
            manual[key] = rect
        elif key == "roi1" and index in self._auto_roi1_candidates:
            structure = self._auto_roi1_candidates[index].structure
            self._auto_roi1_candidates[index] = RoiCandidate(rect.y0, rect.x0, rect.width, structure)
            if index not in self._auto_roi2_moved and "roi2" not in manual:
                self._compute_auto_roi2_for_slice(index)
        elif key == "roi2" and index in self._auto_roi2_candidates:
            region = self._auto_roi2_candidates[index].structure
            self._auto_roi2_candidates[index] = RoiCandidate(rect.y0, rect.x0, rect.width, region)
            self._auto_roi2_moved.add(index)
        self._show_auto_rois_for_current_slice()

    def _clear_auto_rois(self) -> None:
        self._stop_auto_tissue_worker()
        self._auto_roi1_candidates = {}
        self._auto_roi2_candidates = {}
        self._auto_roi2_moved = set()
        self._auto_rois_status_label.setText("")

    def _reset_auto_rois(self) -> None:
        self._clear_auto_rois()
        self._auto_rois_cb.blockSignals(True)
        self._auto_rois_cb.setChecked(False)
        self._auto_rois_cb.blockSignals(False)
        self._update_auto_rois_availability()

    def _update_set_roi_buttons(self) -> None:
        """Set ROI1/ROI2 are available while there's a hand-drawn draft (it
        stays committable to the other slot after the first commit) and the
        ROIs aren't confirmed.
        """
        confirmed = self._confirm_rois_cb.isChecked()
        for button in (self._set_roi1_btn, self._set_roi2_btn):
            button.setEnabled(self._has_hand_drawn_draft and not confirmed)
            if confirmed:
                button.setToolTip("ROIs are confirmed - uncheck Confirmed to change them")
            else:
                button.setToolTip("" if self._has_hand_drawn_draft else "Draw a rectangle on the image first")

    def _build_results_page(self) -> QWidget:
        """Two tabs: "Per slice" - one slice's QA result at a time, scrolled
        through like the image viewer (slider, mouse wheel); "Study summary"
        - the study-level verdict and a per-slice table.
        """
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        self._results_tabs = QTabWidget()
        outer.addWidget(self._results_tabs)

        per_slice = QWidget()
        self._results_per_slice_page = per_slice
        per_slice.installEventFilter(self)  # mouse wheel = next/previous slice
        layout = QVBoxLayout(per_slice)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        nav_row = QHBoxLayout()
        self._results_prev_btn = QPushButton("◀")
        self._results_prev_btn.clicked.connect(lambda: self._step_results_slice(-1))
        self._results_slider = QSlider(Qt.Orientation.Horizontal)
        self._results_slider.setEnabled(False)
        self._results_slider.valueChanged.connect(self._on_results_slider_changed)
        self._results_next_btn = QPushButton("▶")
        self._results_next_btn.clicked.connect(lambda: self._step_results_slice(1))
        self._results_slice_label = QLabel("No results yet")
        self._results_slice_label.setProperty("role", "mono")
        nav_row.addWidget(self._results_prev_btn)
        nav_row.addWidget(self._results_slider, 1)
        nav_row.addWidget(self._results_next_btn)
        nav_row.addWidget(self._results_slice_label)
        layout.addLayout(nav_row)
        # Sorted slice indices that have a result or an error - the slider's positions.
        self._results_slices: list[int] = []

        score_row = QHBoxLayout()
        self._result_verdict_label = QLabel("—")
        self._result_verdict_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px; font-weight: 700;")
        self._result_decision_label = QLabel("")
        self._result_decision_label.setProperty("role", "mono")
        score_row.addWidget(self._result_verdict_label)
        score_row.addWidget(self._result_decision_label)
        score_row.addStretch(1)
        layout.addLayout(score_row)

        # 2x2 grid: MTF/NPS/PS are all "spectrum vs spatial frequency"
        # curves for the two ROIs (NPS = homogeneous ROI2's true noise
        # spectrum, PS = structured ROI1's spectrum per WP2's terminology),
        # PS/MTF is the combined ratio curve used for the accept/reject
        # decision.
        charts_grid = QGridLayout()
        charts_grid.setSpacing(10)
        self._mtf_chart_view = self._make_chart_view("MTF curve — ROI1 (mean, first half)", "Spatial frequency [1/mm]", "MTF")
        self._nps_chart_view = self._make_chart_view("NPS curve — ROI2 (homogeneous)", "Spatial frequency [1/mm]", "NPS [HU²mm²]")
        self._ps_chart_view = self._make_chart_view("PS curve — ROI1 (structured)", "Spatial frequency [1/mm]", "PS [HU²mm²]")
        self._ratio_chart_view = self._make_chart_view("PS/MTF ratio", "Spatial frequency [1/mm]", "PS/MTF")
        charts_grid.addWidget(self._mtf_chart_view, 0, 0)
        charts_grid.addWidget(self._nps_chart_view, 0, 1)
        charts_grid.addWidget(self._ps_chart_view, 1, 0)
        charts_grid.addWidget(self._ratio_chart_view, 1, 1)
        for chart_view in (self._mtf_chart_view, self._nps_chart_view, self._ps_chart_view, self._ratio_chart_view):
            chart_view.installEventFilter(self)  # wheel over a chart scrolls slices too
        # Stretch factor 1 so the charts actually grow to fill available
        # vertical space - previously a trailing addStretch(1) after the
        # feature box soaked up all of it instead, leaving the charts
        # pinned to their minimum size no matter how tall the window was.
        layout.addLayout(charts_grid, 1)

        self._advanced_toggle_btn = QPushButton("▸ Advanced: feature vector")
        self._advanced_toggle_btn.setCheckable(True)
        self._advanced_toggle_btn.setChecked(False)
        self._advanced_toggle_btn.clicked.connect(self._on_advanced_toggle_clicked)
        layout.addWidget(self._advanced_toggle_btn)

        feature_box = QGroupBox("Feature vector — classify_quality input")
        feature_form = QFormLayout(feature_box)
        self._feature_labels: dict[str, QLabel] = {}
        for name in _FEATURE_NAMES:
            value_label = QLabel("—")
            value_label.setProperty("role", "mono")
            feature_form.addRow(name, value_label)
            self._feature_labels[name] = value_label
        feature_box.setVisible(False)  # hidden by default, behind the Advanced toggle
        self._feature_box = feature_box
        layout.addWidget(feature_box)  # natural size - no stretch, so charts_grid gets the extra room
        self._results_tabs.addTab(per_slice, "Per slice")

        summary = QWidget()
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(0, 8, 0, 0)
        self._study_verdict_label = QLabel("—")
        self._study_verdict_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px; font-weight: 700;")
        summary_layout.addWidget(self._study_verdict_label)
        self._study_table = QTableWidget(0, len(_STUDY_TABLE_COLUMNS))
        self._study_table.setHorizontalHeaderLabels(_STUDY_TABLE_COLUMNS)
        self._study_table.verticalHeader().setVisible(False)
        self._study_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._study_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._study_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._study_table.cellClicked.connect(self._on_study_table_clicked)
        summary_layout.addWidget(self._study_table, 1)
        self._results_tabs.addTab(summary, "Study summary")
        # Connected only now - adding the first tab emits currentChanged too.
        self._results_tabs.currentChanged.connect(self._on_results_tab_changed)
        return page

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override signature
        # Scroll through per-slice results with the mouse wheel, like the image viewer.
        if (
            event.type() == QEvent.Type.Wheel
            and self._results_slices
            and (watched is self._results_per_slice_page or watched in self._results_chart_views())
        ):
            self._step_results_slice(-1 if event.angleDelta().y() > 0 else 1)
            return True
        return super().eventFilter(watched, event)

    def _results_chart_views(self) -> tuple:
        return (self._mtf_chart_view, self._nps_chart_view, self._ps_chart_view, self._ratio_chart_view)

    def _step_results_slice(self, step: int) -> None:
        if self._results_slices:
            self._results_slider.setValue(self._results_slider.value() + step)

    def _on_results_slider_changed(self, position: int) -> None:
        if not 0 <= position < len(self._results_slices):
            return
        index = self._results_slices[position]
        self._show_result_for_slice(index)
        # Keep the image viewer on the same slice, so switching back shows its ROIs.
        if self._loaded_volume is not None and self._slice_slider.value() != index:
            self._slice_slider.setValue(index)

    def _on_study_table_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._results_slices):
            self._results_slider.setValue(row)
            self._results_tabs.setCurrentIndex(0)

    def _refresh_results_page(self) -> None:
        """Rebuilds the slice list/slider and the study summary from `_qa_results`.
        Failed slices aren't listed - only counted in the study verdict line.
        """
        self._results_slices = sorted(self._qa_results)
        self._results_slider.blockSignals(True)
        self._results_slider.setRange(0, max(len(self._results_slices) - 1, 0))
        self._results_slider.blockSignals(False)
        self._results_slider.setEnabled(len(self._results_slices) > 1)

        self._study_table.setRowCount(len(self._results_slices))
        for row, index in enumerate(self._results_slices):
            result = self._qa_results[index]
            acceptable = result.predicted_label == 1
            cells = [
                str(index + 1),
                f"{result.mtf_area:.3f}",
                f"{result.nps_roi2.var_nps:.3f}",
                f"{result.ratio.ratio_lf_hf:.2f}",
                f"{result.decision_value:+.2f}",
                "Acceptable" if acceptable else "Unacceptable",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == len(cells) - 1:
                    item.setForeground(QColor(GOOD if acceptable else CRITICAL))
                self._study_table.setItem(row, column, item)

        verdict = study_verdict(self._qa_results.values())
        if verdict is None:
            self._study_verdict_label.setText("—")
            self._study_verdict_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px; font-weight: 700;")
            return
        color = {ACCEPTABLE: GOOD, UNACCEPTABLE: CRITICAL}.get(verdict.verdict, TEXT_MUTED)
        failed = f" ({len(self._qa_errors)} failed, not shown)" if self._qa_errors else ""
        self._study_verdict_label.setText(
            f"Study: {verdict.verdict} — {verdict.acceptable} of {verdict.total} slice(s) acceptable{failed}"
        )
        self._study_verdict_label.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: 700;")

    def _show_result_for_slice(self, index: int) -> None:
        """The Per-slice tab's content for `index` (a slice with a result)."""
        if index not in self._results_slices:
            return
        position = self._results_slices.index(index)
        self._results_slider.blockSignals(True)
        self._results_slider.setValue(position)
        self._results_slider.blockSignals(False)
        self._results_prev_btn.setEnabled(position > 0)
        self._results_next_btn.setEnabled(position < len(self._results_slices) - 1)
        self._results_slice_label.setText(
            f"Slice {index + 1} / {self._loaded_volume.num_slices}  ({position + 1} of {len(self._results_slices)})"
        )

        result = self._qa_results[index]

        # "Acceptable"/"Unacceptable" matches WP1's own terminology for this
        # decision (e.g. "acceptable images show higher [PS/MTF] ratios...
        # unacceptable images show lower ratios and faster decay") rather
        # than an invented label.
        is_acceptable = result.predicted_label == 1
        self._result_verdict_label.setText("Acceptable" if is_acceptable else "Unacceptable")
        verdict_color = GOOD if is_acceptable else CRITICAL
        self._result_verdict_label.setStyleSheet(f"color: {verdict_color}; font-size: 15px; font-weight: 700;")
        self._result_decision_label.setText(f"decision {result.decision_value:+.2f}")
        for name, value in zip(_FEATURE_NAMES, result.feature_vector):
            self._feature_labels[name].setText(f"{value:.4g}")

        # Plot the mean across edges, truncated to the first half (see
        # `mtf_display_curve`'s docstring for why) - not one series per
        # edge, which for a busy ROI producing dozens of detected edges
        # (a known, separate issue - see project memory on the MTF
        # bump-removal mismatch) made the chart an unreadable tangle.
        mtf_x, mtf_y = mtf_display_curve(result.mtf_xaxis, result.mtf)
        self._update_line_chart(self._mtf_chart_view, mtf_x, [mtf_y] if mtf_y.size else [])
        # NPS = ROI2 (homogeneous, true noise spectrum); PS = ROI1
        # (structured ROI, WP2's "PS") - both computed by the same
        # `nps_complete_roi` function, just on different ROIs.
        self._update_line_chart(self._nps_chart_view, result.nps_roi2.f_avg, [result.nps_roi2.nps_measured_radagv])
        self._update_line_chart(self._ps_chart_view, result.nps_roi1.f_avg, [result.nps_roi1.nps_measured_radagv])
        self._update_line_chart(self._ratio_chart_view, result.ratio.f_int, [result.ratio.ratio_nps_mtf])

        # Export covers the slice shown here.
        self._last_result = result
        self._last_export_context = ExportContext(
            series_label=self._series_combo.currentText() or "Series",
            slice_number=index + 1,
            slice_count=self._loaded_volume.num_slices,
            pixel_spacing_mm=self._loaded_volume.pixel_spacing_mm,
            acquisition_date=self._loaded_volume.acquisition_date,
        )

    def _on_advanced_toggle_clicked(self, checked: bool) -> None:
        self._feature_box.setVisible(checked)
        self._advanced_toggle_btn.setText(("▾" if checked else "▸") + " Advanced: feature vector")

    def _make_chart_view(self, title: str, x_label: str, y_label: str) -> QChartView:
        chart = QChart()
        chart.setTitle(title)
        chart.legend().setVisible(False)
        chart.setBackgroundBrush(QBrush(QColor(SURFACE_SUNKEN)))
        chart.setBackgroundRoundness(6)
        chart.setTitleBrush(QBrush(QColor(TEXT_MUTED)))

        x_axis = QValueAxis()
        x_axis.setTitleText(x_label)
        x_axis.setRange(0.0, 1.0)
        # QValueAxis's default label format ("") lets Qt pick decimal
        # precision dynamically per range/tick, which for these curves'
        # values (many decimal places from raw float64 data) can produce
        # tick labels too wide for the plot area - Qt then silently elides
        # them to "...". A fixed, short format plus a fixed tick count
        # guarantees labels that always fit.
        x_axis.setLabelFormat("%.2f")
        x_axis.setTickCount(5)
        y_axis = QValueAxis()
        y_axis.setTitleText(y_label)
        y_axis.setRange(0.0, 1.0)
        y_axis.setLabelFormat("%.2f")
        y_axis.setTickCount(5)
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

        view = QChartView(chart)
        view.setMinimumHeight(190)  # 4 charts now share a 2x2 grid, not 2 side by side
        view.setMinimumWidth(300)  # enough room for 5 formatted tick labels without crowding
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        return view

    def _update_line_chart(self, chart_view: QChartView, x_values: np.ndarray, y_columns: list[np.ndarray]) -> None:
        """Replaces `chart_view`'s series with one `QLineSeries` per column in
        `y_columns` (all sharing `x_values`), rescaling the axes to fit.
        `y_columns` may hold more than one array - e.g. MTF has one curve
        per detected edge in the ROI - all drawn overlaid on the same axes.
        """
        chart = chart_view.chart()
        for series in list(chart.series()):
            chart.removeSeries(series)

        x_axis = chart.axes(Qt.Orientation.Horizontal)[0]
        y_axis = chart.axes(Qt.Orientation.Vertical)[0]

        if len(x_values) == 0 or not y_columns:
            x_axis.setRange(0.0, 1.0)
            y_axis.setRange(0.0, 1.0)
            return

        finite_x = x_values[np.isfinite(x_values)]
        finite_y = np.concatenate([col[np.isfinite(col)] for col in y_columns])
        x_min, x_max = (float(finite_x.min()), float(finite_x.max())) if finite_x.size else (0.0, 1.0)
        y_min, y_max = (float(finite_y.min()), float(finite_y.max())) if finite_y.size else (0.0, 1.0)
        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            y_max = y_min + 1.0

        for y_values in y_columns:
            series = QLineSeries()
            for x, y in zip(x_values, y_values):
                if np.isfinite(x) and np.isfinite(y):
                    series.append(float(x), float(y))
            chart.addSeries(series)
            series.attachAxis(x_axis)
            series.attachAxis(y_axis)

        x_axis.setRange(x_min, x_max)
        y_axis.setRange(y_min, y_max)

    # ---- right pane: sidebar ----

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(360)
        sidebar.setMaximumWidth(420)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_study_info_box())

        # Wrapped in one widget (not just a bare layout) so `_show_view_page`
        # can hide Segmentation + ROI Selection together on the Results page
        # - Run Metrics below stays visible on both pages. Stacked full-width
        # (not side by side) - ROI Selection's table needs its own row's
        # worth of width; splitting the sidebar in half is what caused the
        # coordinate-text overflow bug in the first place (see project memory).
        self._segmentation_roi_widget = QWidget()
        column = QVBoxLayout(self._segmentation_roi_widget)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._build_segmentation_box())
        # HU-threshold-only, no-ML feature - kept visually separate from the
        # CNN Segmentation box above (different method, different purpose)
        # per explicit request (2026-09-24).
        column.addWidget(self._build_brain_tissue_box())
        column.addWidget(self._build_roi_box())
        # Placed after (not merged with) the QC-relevant Segmentation/ROI
        # boxes - stroke detection is a diagnostic-adjacent capability, not
        # an image-quality measurement, kept visually separate for that
        # reason (see domain.segmentation.aisd_stroke_unet's docstring).
        # Hidden for now (per explicit request, 2026-09-24 - not needed
        # yet) - still built and added normally (kept parented/referenced,
        # not skipped) so every internal widget reference
        # (`_check_stroke_btn` etc.) and all the existing stroke-detection
        # wiring/tests stay valid; only its visibility changes.
        self._stroke_box = self._build_stroke_box()
        self._stroke_box.setVisible(False)
        column.addWidget(self._stroke_box)
        layout.addWidget(self._segmentation_roi_widget)

        layout.addWidget(self._build_metrics_box(), 1)

        return sidebar

    def _build_study_info_box(self) -> QWidget:
        # Patient/study/technical identity now lives on the image itself as
        # corner overlay text (see `_update_viewer_overlay` and
        # `RoiViewport.set_overlay_text`) - the standard DICOM-viewer
        # convention (RadiAnt/OsiriX/Weasis), and it frees up this sidebar
        # for upcoming segmentation controls. Only the series picker (an
        # actual interactive control, not a readout) stays here.
        box = QGroupBox("Series")
        form = QFormLayout(box)

        # `discover_series` can find more than one coherent stack under one
        # folder (real scout/localizer series, or a nominal series that
        # turned out to hold more than one image geometry - see
        # `integration.dicom_file_loader`'s module docstring). Auto-selects
        # the best guess (CT, most slices) but always lets the user see what
        # else was found and pick a different one, rather than the tool
        # silently guessing among genuinely different acquisitions.
        self._series_combo = QComboBox()
        self._series_combo.setEnabled(False)
        self._series_combo.currentIndexChanged.connect(self._on_series_combo_changed)
        form.addRow("Series", self._series_combo)
        return box

    def _update_viewer_overlay(self) -> None:
        """Refreshes the four corner text labels on the image itself (see
        `RoiViewport.set_overlay_text`) from the current volume/slice/series.

        Content and grouping (identity top, technical/exposure bottom, one
        line per logical field) follow real DICOM viewers researched for
        this - RadiAnt's `overlay.xml` defaults and MicroDicom's corner
        layout both put patient/study/series identity in the top corners and
        acquisition/exposure/display technical detail in the bottom corners;
        MicroDicom's own screenshot (patient name, ID+sex, study/series
        description top-left, institution+scanner+acquisition time top-right,
        ST/SL/modality/transfer-syntax/image-and-series-number bottom-left,
        kV/mA/WL-WW/zoom bottom-right) is what this now matches field-for-
        field, with CTDIvol/DLP added on top per explicit request even
        though dose isn't part of the standard overlay convention.

        Called on every volume load, every slice change (slice number and
        SL genuinely vary per slice), every window-preset click (WL/WW), and
        every zoom change (`RoiViewport.zoom_changed`) - nothing here is
        cached, so switching series/slice/preset/zoom can never leave a
        stale reading on screen.
        """
        volume = self._loaded_volume
        if volume is None:
            for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
                self._roi_viewport.set_overlay_text(corner, [])
            return

        series_label = self._series_combo.currentText() or "Series"
        slice_index = self._current_slice_index

        self._roi_viewport.set_overlay_text(
            "top_left",
            [
                volume.patient_name or volume.patient_id or "—",
                f"ID: {volume.patient_id or '—'}   Sex: {volume.patient_sex or '—'}",
                volume.study_description or "—",
                volume.series_description or series_label,
            ],
        )
        self._roi_viewport.set_overlay_text(
            "top_right",
            [
                volume.institution_name or "—",
                volume.manufacturer_model or "—",
                f"Acquired: {self._format_dicom_date(volume.acquisition_date)} {volume.acquisition_time or ''}".strip(),
            ],
        )

        slice_location = (
            volume.slice_locations_mm[slice_index]
            if 0 <= slice_index < len(volume.slice_locations_mm)
            else None
        )
        st = f"{volume.slice_thickness_mm:.2f} mm" if volume.slice_thickness_mm is not None else "—"
        sl = f"{slice_location:.1f} mm" if slice_location is not None else "—"
        self._roi_viewport.set_overlay_text(
            "bottom_left",
            [
                f"Series {volume.series_number or '—'} · Image {slice_index + 1}/{volume.num_slices}",
                f"ST: {st}   SL: {sl}",
                f"{volume.modality or '—'} · {volume.transfer_syntax_name or '—'}",
                f"Pixel spacing: {volume.pixel_spacing_mm:.3f} mm",
                f"Recon: {volume.reconstruction or '—'}",
            ],
        )

        window_width, window_center = self._roi_viewport.current_window()
        zoom_percent = self._roi_viewport.current_zoom_percent()
        exposure = (
            f"{volume.tube_current_ma:.0f} mA" if volume.tube_current_ma is not None else "—"
        ) + "  " + (f"{volume.kvp:.0f} kV" if volume.kvp is not None else "—")
        # CTDIvol/DLP aren't part of the standard corner-overlay convention
        # (they're dose metadata, not identity/technical display state) -
        # included anyway per explicit request.
        ctdi = f"{volume.ctdi_vol_mgy:.1f} mGy" if volume.ctdi_vol_mgy is not None else "—"
        dlp = f"{volume.dlp_mgy_cm:.0f} mGy·cm" if volume.dlp_mgy_cm is not None else "—"
        bottom_right_lines = [
            exposure,
            f"WL: {window_center:.0f}   WW: {window_width:.0f}",
            f"Zoom: {zoom_percent:.0f}%",
            f"CTDIvol: {ctdi}   DLP: {dlp}",
        ]
        # Live cursor/pixel-value hover readout, or the in-progress
        # rectangle's size while drawing (see `_on_cursor_moved`/
        # `_on_draft_changed`) - omitted entirely rather than shown blank
        # when neither is currently happening.
        if self._interaction_line:
            bottom_right_lines.append(self._interaction_line)
        self._roi_viewport.set_overlay_text("bottom_right", bottom_right_lines)

        study_label = volume.study_id or volume.patient_id or "DICOM data"
        self._study_status_label.setText(f"Loaded: {study_label} ({volume.num_slices} slice(s))")

    @staticmethod
    def _format_dicom_date(raw: str | None) -> str:
        """DICOM `DA` values are `YYYYMMDD` with no separators - reformatted
        to `DD-Mon-YYYY` for display (matching MicroDicom's overlay, e.g.
        "11-September-2025"). Falls back to the raw string for anything that
        doesn't match the expected shape rather than raising on odd data.
        """
        if not raw or len(raw) != 8 or not raw.isdigit():
            return raw or "—"
        try:
            return datetime.strptime(raw, "%Y%m%d").strftime("%d-%B-%Y")
        except ValueError:
            return raw

    def _build_segmentation_box(self) -> QWidget:
        # FR-02: wires in `domain.segmentation.head_ct_unet`, a pretrained
        # 16-structure model - see that module's docstring for the model
        # itself and `_SEGMENTATION_MAPPING` above for its 16 raw classes,
        # shown directly with no WP2-specific renaming (not the earlier
        # generic Skull/Brain/Ventricles/Vessels placeholder either). Explicit
        # triggers (not per-checkbox auto-run) since a single inference call
        # is ~1-2s on CPU (see `_unet_architecture.build_unet`'s docstring) -
        # small per click, but still worth an explicit trigger rather than
        # firing on every checkbox toggle. "Segment Volume"
        # (`VolumeSegmentationWorker`) is the batch counterpart to "Segment
        # Slice" - a few minutes for a real multi-hundred-slice study at this
        # per-slice cost, so it still reports live progress and supports
        # mid-run cancellation rather than blocking with no feedback.
        # "Select Segments..." opens `SegmentationSelectionDialog` - a
        # separate, non-modal window (per explicit user request, to avoid
        # cluttering this sidebar with a long checkbox list) for narrowing
        # down which structures to overlay (all 16 are selected by default -
        # see `_selected_segmentation_names`'s own comment), and a vertical
        # slider for their shared opacity; the choice itself lives here
        # (`_selected_segmentation_names`), not in that dialog, and persists
        # across slice navigation for the whole loaded volume (also per
        # explicit request - see `_refresh_segmentation_for_current_slice`).
        # The sidebar's own "Show Segments" checkbox (per a later request)
        # is a quick on/off switch for whatever's currently selected,
        # without needing to open that dialog just to hide/show it.
        box = QGroupBox("Segmentation")
        layout = QVBoxLayout(box)

        header = QVBoxLayout()
        self._segmentation_status_label = QLabel("—")
        self._segmentation_status_label.setProperty("role", "muted")
        header.addWidget(self._segmentation_status_label)

        button_row = QHBoxLayout()
        self._segment_slice_btn = QPushButton("Segment Slice")
        self._segment_slice_btn.setEnabled(False)
        self._segment_slice_btn.setToolTip("Open a study first")
        self._segment_slice_btn.clicked.connect(self._on_segment_slice_clicked)
        button_row.addWidget(self._segment_slice_btn)
        self._segment_volume_btn = QPushButton("Segment Volume")
        self._segment_volume_btn.setEnabled(False)
        self._segment_volume_btn.setToolTip("Open a study first")
        self._segment_volume_btn.clicked.connect(self._on_segment_volume_clicked)
        button_row.addWidget(self._segment_volume_btn)
        self._cancel_segment_volume_btn = QPushButton("Cancel")
        self._cancel_segment_volume_btn.setVisible(False)
        self._cancel_segment_volume_btn.clicked.connect(self._on_cancel_segment_volume_clicked)
        button_row.addWidget(self._cancel_segment_volume_btn)
        header.addLayout(button_row)
        layout.addLayout(header)

        segments_row = QHBoxLayout()
        self._select_segments_btn = QPushButton("Select Segments…")
        self._select_segments_btn.setEnabled(False)
        self._select_segments_btn.setToolTip("Open a study first")
        self._select_segments_btn.clicked.connect(self._on_select_segments_clicked)
        segments_row.addWidget(self._select_segments_btn)
        # A quick on/off switch for the *currently selected* structures'
        # overlays (per explicit request) - separate from the selection
        # itself (chosen in the "Select Segments…" dialog, all 16 selected
        # by default): toggling this never changes
        # `_selected_segmentation_names`, just whether those
        # already-chosen structures are actually drawn right now, so
        # flipping it back on shows exactly what was showing before.
        self._show_segments_cb = QCheckBox("Show Segments")
        self._show_segments_cb.setChecked(False)  # per explicit request (2026-09-24) - opt in, not on by default
        self._show_segments_cb.setEnabled(False)
        self._show_segments_cb.setToolTip("Open a study first")
        self._show_segments_cb.toggled.connect(self._on_show_segments_toggled)
        segments_row.addWidget(self._show_segments_cb)
        layout.addLayout(segments_row)

        return box

    def _on_select_segments_clicked(self) -> None:
        if self._segment_selection_dialog is None:
            dialog = SegmentationSelectionDialog(_SEGMENTATION_MAPPING, _SEGMENTATION_COLORS, parent=self)
            dialog.selection_changed.connect(self._on_segmentation_selection_changed)
            dialog.opacity_changed.connect(self._on_segmentation_opacity_changed)
            self._segment_selection_dialog = dialog
        # Reflects whatever's already been segmented for this volume (may be
        # nothing yet, if opened before the first Segment Slice/Volume) -
        # cheap and idempotent to set on every open rather than tracking a
        # separate "has data" flag just for this.
        self._segment_selection_dialog.set_enabled_all(bool(self._segmentation_cache))
        self._segment_selection_dialog.show()
        self._segment_selection_dialog.raise_()
        self._segment_selection_dialog.activateWindow()

    def _ensure_nifti_volume_then(self, on_ready) -> None:
        """Runs `on_ready()` once `self._dicom_nifti_volume` exists for the
        current study - immediately if it's already built (cached for the
        lifetime of the loaded volume, cleared in `_reset_segmentation_
        display`), or after a background `NiftiConversionWorker` finishes
        building it otherwise. Both `_on_segment_slice_clicked` and
        `_on_segment_volume_clicked` need this same "build once, on first
        use" step before they can actually run inference - see
        `integration.dicom_to_nifti`'s module docstring for why
        segmentation needs a NIfTI conversion at all.
        """
        if self._dicom_nifti_volume is not None:
            on_ready()
            return
        if self._active_nifti_conversion_worker is not None:
            return  # already converting (e.g. a fast double-click) - it'll call on_ready via the queued signal below only once, so just wait

        from bictool.application.nifti_conversion_worker import NiftiConversionWorker

        logger.info("Converting DICOM series to NIfTI for segmentation...")
        self._segmentation_status_label.setText("Converting to NIfTI…")
        self._segment_slice_btn.setEnabled(False)
        self._segment_volume_btn.setEnabled(False)
        worker = NiftiConversionWorker(self._loaded_volume.slice_paths, self._loaded_volume.get_slice_hu)
        worker.signals.finished.connect(lambda result: self._on_nifti_conversion_finished(result, on_ready))
        worker.signals.error.connect(self._on_nifti_conversion_error)
        self._active_nifti_conversion_worker = worker  # keep a reference alive while it runs
        self._scheduler.submit(worker)

    def _on_nifti_conversion_finished(self, result, on_ready) -> None:
        logger.info("NIfTI conversion finished.")
        self._active_nifti_conversion_worker = None
        self._dicom_nifti_volume = result
        self._segmentation_status_label.setText("—")
        self._segment_slice_btn.setEnabled(True)
        self._segment_volume_btn.setEnabled(True)
        on_ready()

    def _on_nifti_conversion_error(self, message: str) -> None:
        logger.error("NIfTI conversion failed: %s", message)
        self._active_nifti_conversion_worker = None
        self._segment_slice_btn.setEnabled(True)
        self._segment_volume_btn.setEnabled(True)
        self._segmentation_status_label.setText("Error")
        self._segmentation_status_label.setToolTip(message)

    def _on_segmentation_selection_changed(self, display_name: str, checked: bool) -> None:
        if checked:
            self._selected_segmentation_names.add(display_name)
            self._apply_segment_overlay(display_name)
        else:
            self._selected_segmentation_names.discard(display_name)
            self._roi_viewport.clear_overlay(f"seg_{display_name}")

    def _apply_segment_overlay(self, display_name: str) -> None:
        """Shows `display_name`'s overlay for the current slice, at the
        current opacity - unless the "Show Segments" master switch is off
        (in which case this is a no-op; the caller's selection state is
        still updated as normal, it just isn't drawn right now) or this
        slice simply hasn't been segmented yet (no cached mask for this
        structure). Shared by every place that (re-)applies overlays:
        an individual checkbox toggling on, a batch of new results landing
        (`_apply_segmentation_masks`), the master switch being turned back
        on, and the opacity slider moving.
        """
        if not self._show_segments_cb.isChecked():
            return
        masks = self._segmentation_masks_for_current_slice
        if masks is None or display_name not in masks:
            return
        self._roi_viewport.set_overlay(
            f"seg_{display_name}",
            masks[display_name],
            color=_SEGMENTATION_COLORS[display_name],
            opacity=self._segmentation_opacity,
            label=display_name,
        )

    def _on_show_segments_toggled(self, checked: bool) -> None:
        """The main sidebar's quick on/off switch - shows or hides every
        currently *selected* structure's overlay without touching the
        selection itself (`_selected_segmentation_names`), so turning it
        back on reproduces exactly what was showing before.
        """
        for display_name in self._selected_segmentation_names:
            if checked:
                self._apply_segment_overlay(display_name)
            else:
                self._roi_viewport.clear_overlay(f"seg_{display_name}")

    def _on_segmentation_opacity_changed(self, opacity: float) -> None:
        self._segmentation_opacity = opacity
        for display_name in self._selected_segmentation_names:
            self._apply_segment_overlay(display_name)

    def _on_segment_slice_clicked(self) -> None:
        if self._loaded_volume is None:
            return
        if self._active_volume_segmentation_worker is not None:
            return  # defensive only - both buttons are disabled while a volume batch runs
        cached = self._segmentation_cache.get(self._current_slice_index)
        if cached is not None:
            self._apply_segmentation_result(cached, from_cache=True)
            return

        weights_path = self._segmentation_config.get("segmentation", {}).get("anatomy_weights_path")
        if not weights_path:
            QMessageBox.warning(
                self,
                "Segmentation model not configured",
                "No anatomy segmentation weights are configured. Set BICTOOL_HEADCT_WEIGHTS "
                "or the segmentation.anatomy_weights_path config key - see "
                "domain.segmentation.head_ct_unet for where to obtain the weights file.",
            )
            return

        try:
            # Deferred to first actual use - see the TYPE_CHECKING import
            # block's comment for why (TensorFlow's import cost).
            # `SegmentationWorker` isn't used directly here any more (that
            # moved to `_do_segment_slice`) but is still imported here too -
            # this whole try/except exists to fail fast with one clear
            # message if *either* module is unavailable, before spending
            # time on a NIfTI conversion the click can't actually finish.
            from bictool.application.segmentation_worker import SegmentationWorker  # noqa: F401
            from bictool.domain.segmentation import head_ct_unet
        except ImportError as exc:
            QMessageBox.critical(
                self, "TensorFlow not installed", f"Segmentation requires TensorFlow (pip install bictool[ml]): {exc}"
            )
            return

        if self._anatomy_model is None:
            self._anatomy_model = head_ct_unet.build_model()
        try:
            head_ct_unet.load_weights(self._anatomy_model, weights_path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Segmentation weights not found", str(exc))
            return

        self._ensure_nifti_volume_then(self._do_segment_slice)

    def _do_segment_slice(self) -> None:
        # `SegmentationWorker`/`head_ct_unet` imports and the model/weights
        # were already resolved in `_on_segment_slice_clicked` before this
        # was scheduled (possibly after a NIfTI-conversion detour) - re-import
        # here is just picking the already-imported module back up, not a
        # second TensorFlow load.
        from bictool.application.segmentation_worker import SegmentationWorker

        if not self._dicom_nifti_volume.is_segmentable(self._current_slice_index):
            self._segmentation_status_label.setText("Not segmentable")
            self._segmentation_status_label.setToolTip(
                "This slice has no corresponding processed volume slice (e.g. a scout/localizer "
                "sharing this series' geometry) - see integration.dicom_to_nifti."
            )
            return

        logger.info("Segmenting slice %d...", self._current_slice_index)
        self._segmentation_status_label.setText("Segmenting…")
        self._segment_slice_btn.setEnabled(False)
        self._segment_volume_btn.setEnabled(False)
        slice_hu = self._dicom_nifti_volume.get_nifti_slice_hu(self._current_slice_index)
        worker = SegmentationWorker(self._anatomy_model, slice_hu)
        worker.signals.finished.connect(self._on_segmentation_finished)
        worker.signals.error.connect(self._on_segmentation_error)
        self._active_segmentation_worker = worker  # keep a reference alive while it runs
        self._scheduler.submit(worker)

    def _on_segmentation_finished(self, result) -> None:
        logger.info("Segmentation finished for slice %d.", self._current_slice_index)
        self._active_segmentation_worker = None
        # `result` was computed on a NIfTI-space slice (`_dicom_nifti_volume
        # .get_nifti_slice_hu`) - map it back onto this DICOM slice's own
        # pixel grid before it reaches the cache/overlays/selection dialog,
        # so every consumer downstream keeps working in DICOM pixel space
        # exactly as before. This mapping is pure array logic with no Qt
        # involvement, so it lives on the Domain-layer alignment object
        # itself (`DicomNiftiVolume.remap_result_to_dicom_space`), not here.
        result = self._dicom_nifti_volume.remap_result_to_dicom_space(result)
        self._segmentation_cache[self._current_slice_index] = result
        self._apply_segmentation_result(result, from_cache=False)
        # This slice now has real CNN segmentation data, which
        # `_brain_mask_for_current_slice` prefers over the HU-threshold
        # fallback - drop any already-cached (fallback-based) Brain Tissue
        # result and refresh, so "Brain Mask" becomes available/correct
        # immediately rather than only on the next slice re-visit.
        self._brain_tissue_cache.pop(self._current_slice_index, None)
        self._refresh_brain_tissue_for_current_slice()
        self._on_auto_rois_segmentation_landed(self._current_slice_index)

    def _apply_segmentation_result(self, result, *, from_cache: bool) -> None:
        self._segment_slice_btn.setEnabled(True)
        self._segment_volume_btn.setEnabled(True)
        self._segmentation_status_label.setText("Done (cached)" if from_cache else "Done")
        self._apply_segmentation_masks(result)

    def _apply_segmentation_masks(self, result) -> None:
        """Decodes `result`'s raw model-class masks into checkbox display
        names and (re-)applies overlays for whichever ones are currently
        selected (`_selected_segmentation_names` - a persisted, whole-volume
        choice, not tied to this particular slice) - shared by the
        single-slice path (`_apply_segmentation_result`, which also updates
        the status label/buttons) and the mid-volume-batch live-update path
        (`_on_volume_segmentation_slice_finished`), which must leave the
        in-progress batch's own status text/button state alone. Also
        enables the selection dialog's checkboxes, if it's open - it starts
        disabled until at least one slice has real data to show.
        """
        decoded = decode_masks_for_checkboxes(result, _SEGMENTATION_MAPPING)
        self._segmentation_masks_for_current_slice = decoded
        if self._segment_selection_dialog is not None:
            self._segment_selection_dialog.set_enabled_all(True)
        for display_name in self._selected_segmentation_names:
            self._apply_segment_overlay(display_name)

    def _on_segmentation_error(self, message: str) -> None:
        logger.error("Segmentation failed: %s", message)
        self._active_segmentation_worker = None
        self._segment_slice_btn.setEnabled(True)
        self._segment_volume_btn.setEnabled(True)
        self._segmentation_status_label.setText("Error")
        self._segmentation_status_label.setToolTip(message)

    # ---- anatomy segmentation: whole-volume batch (background, cancellable) ----

    def _on_segment_volume_clicked(self) -> None:
        if self._loaded_volume is None:
            return
        if self._active_volume_segmentation_worker is not None:
            return  # defensive only - the button is disabled while a batch already runs
        total_slices = self._loaded_volume.num_slices
        if all(i in self._segmentation_cache for i in range(total_slices)):
            self._segmentation_status_label.setText("Done (all slices already segmented)")
            return

        weights_path = self._segmentation_config.get("segmentation", {}).get("anatomy_weights_path")
        if not weights_path:
            QMessageBox.warning(
                self,
                "Segmentation model not configured",
                "No anatomy segmentation weights are configured. Set BICTOOL_HEADCT_WEIGHTS "
                "or the segmentation.anatomy_weights_path config key - see "
                "domain.segmentation.head_ct_unet for where to obtain the weights file.",
            )
            return

        try:
            # Deferred to first actual use - see the TYPE_CHECKING import
            # block's comment for why (TensorFlow's import cost).
            # `VolumeSegmentationWorker` isn't used directly here any more
            # (that moved to `_do_segment_volume`) but is still imported
            # here too - see `_on_segment_slice_clicked`'s matching comment.
            from bictool.application.volume_segmentation_worker import VolumeSegmentationWorker  # noqa: F401
            from bictool.domain.segmentation import head_ct_unet
        except ImportError as exc:
            QMessageBox.critical(
                self, "TensorFlow not installed", f"Segmentation requires TensorFlow (pip install bictool[ml]): {exc}"
            )
            return

        if self._anatomy_model is None:
            self._anatomy_model = head_ct_unet.build_model()
        try:
            head_ct_unet.load_weights(self._anatomy_model, weights_path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Segmentation weights not found", str(exc))
            return

        self._ensure_nifti_volume_then(self._do_segment_volume)

    def _do_segment_volume(self) -> None:
        from bictool.application.volume_segmentation_worker import VolumeSegmentationWorker

        total_slices = self._loaded_volume.num_slices
        pending_indices = [
            i
            for i in range(total_slices)
            if i not in self._segmentation_cache and self._dicom_nifti_volume.is_segmentable(i)
        ]
        not_segmentable_count = sum(
            1
            for i in range(total_slices)
            if i not in self._segmentation_cache and not self._dicom_nifti_volume.is_segmentable(i)
        )
        if not pending_indices:
            self._segmentation_status_label.setText(
                "Done (all segmentable slices already segmented)" if not_segmentable_count else
                "Done (all slices already segmented)"
            )
            return

        logger.info(
            "Segmenting volume: %d of %d slice(s) pending (%d not segmentable)...",
            len(pending_indices),
            total_slices,
            not_segmentable_count,
        )
        self._segmentation_status_label.setText(f"Segmenting volume… 0/{len(pending_indices)}")
        self._segment_slice_btn.setEnabled(False)
        self._segment_volume_btn.setEnabled(False)
        self._cancel_segment_volume_btn.setVisible(True)
        worker = VolumeSegmentationWorker(
            self._anatomy_model, self._dicom_nifti_volume.get_nifti_slice_hu, pending_indices
        )
        worker.signals.slice_finished.connect(self._on_volume_segmentation_slice_finished)
        worker.signals.progress.connect(self._on_volume_segmentation_progress)
        worker.signals.finished.connect(self._on_volume_segmentation_finished)
        worker.signals.cancelled.connect(self._on_volume_segmentation_cancelled)
        worker.signals.error.connect(self._on_volume_segmentation_error)
        self._active_volume_segmentation_worker = worker  # keep a reference alive while it runs
        self._scheduler.submit(worker)

    def _on_cancel_segment_volume_clicked(self) -> None:
        if self._active_volume_segmentation_worker is not None:
            self._active_volume_segmentation_worker.cancel()
            self._segmentation_status_label.setText("Cancelling…")

    def _on_volume_segmentation_slice_finished(self, slice_index: int, result) -> None:
        result = self._dicom_nifti_volume.remap_result_to_dicom_space(result)
        self._segmentation_cache[slice_index] = result
        if slice_index == self._current_slice_index:
            # Live-updates the currently displayed slice's checkboxes/
            # overlay as soon as its own result is ready, without waiting
            # for the whole batch - and without touching the batch's own
            # progress text/buttons (`_apply_segmentation_result` does both,
            # which would be wrong mid-batch).
            self._apply_segmentation_masks(result)
            # Same reasoning as `_on_segmentation_finished`: this slice now
            # has real CNN data, so "Brain Mask" may just have become
            # available - refresh immediately rather than on next re-visit.
            self._brain_tissue_cache.pop(slice_index, None)
            self._refresh_brain_tissue_for_current_slice()
        self._on_auto_rois_segmentation_landed(slice_index)

    def _on_volume_segmentation_progress(self, completed: int, total: int) -> None:
        self._segmentation_status_label.setText(f"Segmenting volume… {completed}/{total}")

    def _on_volume_segmentation_finished(self) -> None:
        logger.info("Volume segmentation finished.")
        self._active_volume_segmentation_worker = None
        self._segment_slice_btn.setEnabled(True)
        self._segment_volume_btn.setEnabled(True)
        self._cancel_segment_volume_btn.setVisible(False)
        self._segmentation_status_label.setText("Done (volume)")

    def _on_volume_segmentation_cancelled(self, completed: int, total: int) -> None:
        logger.info("Volume segmentation cancelled after %d/%d slice(s).", completed, total)
        self._active_volume_segmentation_worker = None
        self._segment_slice_btn.setEnabled(True)
        self._segment_volume_btn.setEnabled(True)
        self._cancel_segment_volume_btn.setVisible(False)
        self._segmentation_status_label.setText(f"Cancelled ({completed}/{total} segmented)")

    def _on_volume_segmentation_error(self, slice_index: int, message: str) -> None:
        logger.error("Volume segmentation failed on slice %d: %s", slice_index, message)
        self._active_volume_segmentation_worker = None
        self._segment_slice_btn.setEnabled(True)
        self._segment_volume_btn.setEnabled(True)
        self._cancel_segment_volume_btn.setVisible(False)
        self._segmentation_status_label.setText(f"Error (slice {slice_index + 1})")
        self._segmentation_status_label.setToolTip(message)

    # ---- stroke detection (investigational, not a QC/image-quality measurement) ----

    def _build_brain_tissue_box(self) -> QWidget:
        # HU-threshold-only CSF/white/gray matter classification
        # (domain.segmentation.brain_tissue) - no ML, so unlike Segmentation/
        # Stroke there's no weights config to gate on: the CSF/White Matter/
        # Gray Matter checkboxes are simply enabled once a study is loaded
        # (see `_update_segmentation_availability`) and computed
        # synchronously, on demand, per slice (milliseconds - see
        # brain_tissue.py's docstring) rather than via a background worker.
        # "Brain Mask" is the exception - see
        # `_update_brain_mask_checkbox_availability`.
        box = QGroupBox("Brain Tissue")
        layout = QVBoxLayout(box)

        self._csf_cb = QCheckBox("CSF")
        self._csf_cb.setEnabled(False)
        self._csf_cb.setToolTip("Open a study first")
        self._csf_cb.setStyleSheet(f"QCheckBox {{ color: {_BRAIN_TISSUE_COLORS['CSF']}; }}")
        self._csf_cb.toggled.connect(lambda checked: self._on_brain_tissue_checkbox_toggled("CSF", checked))
        layout.addWidget(self._csf_cb)

        self._white_matter_cb = QCheckBox("White Matter")
        self._white_matter_cb.setEnabled(False)
        self._white_matter_cb.setToolTip("Open a study first")
        self._white_matter_cb.setStyleSheet(f"QCheckBox {{ color: {_BRAIN_TISSUE_COLORS['White Matter']}; }}")
        self._white_matter_cb.toggled.connect(lambda checked: self._on_brain_tissue_checkbox_toggled("White Matter", checked))
        layout.addWidget(self._white_matter_cb)

        self._gray_matter_cb = QCheckBox("Gray Matter")
        self._gray_matter_cb.setEnabled(False)
        self._gray_matter_cb.setToolTip("Open a study first")
        self._gray_matter_cb.setStyleSheet(f"QCheckBox {{ color: {_BRAIN_TISSUE_COLORS['Gray Matter']}; }}")
        self._gray_matter_cb.toggled.connect(lambda checked: self._on_brain_tissue_checkbox_toggled("Gray Matter", checked))
        layout.addWidget(self._gray_matter_cb)

        # The intermediate skull-strip result itself (extract_brain_mask),
        # not a tissue class - shown on request so the user can judge
        # skull-strip quality directly, since white/gray matter's own
        # accuracy is bounded by it.
        self._brain_mask_cb = QCheckBox("Brain Mask (skull-strip)")
        self._brain_mask_cb.setEnabled(False)
        self._brain_mask_cb.setToolTip("Open a study first")
        self._brain_mask_cb.setStyleSheet(f"QCheckBox {{ color: {_BRAIN_TISSUE_COLORS['Brain Mask']}; }}")
        self._brain_mask_cb.toggled.connect(lambda checked: self._on_brain_tissue_checkbox_toggled("Brain Mask", checked))
        layout.addWidget(self._brain_mask_cb)

        return box

    def _brain_mask_for_current_slice(self, slice_hu: np.ndarray) -> np.ndarray:
        """Prefers the CNN anatomy segmentation's merged-and-3D-volume-filtered
        brain mask (`segmentation_brain_mask.brain_mask_from_segmentation`)
        for the current slice when it has been segmented - more robust on
        irregular anatomy (e.g. a postop craniotomy/hardware scan) than the
        HU-threshold `extract_brain_mask`, which real data showed cannot
        reliably tell true brain apart from the connected scalp/orbit/sinus
        envelope on such scans (see `architecture/ml_segmentation_integration_notes.md`,
        2026-09-28/29 entries for the full investigation). Falls back to
        `extract_brain_mask` when this slice has no segmentation result yet
        (segmentation not run, weights/TensorFlow unavailable, or this
        specific slice hasn't been segmented) - the HU-threshold method is
        still the only option in that case.
        """
        if self._segmentation_cache:
            masks_by_slice = {index: result.masks for index, result in self._segmentation_cache.items()}
            segmentation_masks = brain_mask_from_segmentation(masks_by_slice, self._loaded_volume.num_slices)
            if self._current_slice_index in segmentation_masks:
                return segmentation_masks[self._current_slice_index]

        threshold_settings = self._brain_tissue_settings
        return extract_brain_mask(slice_hu, threshold_settings.air_hu_cutoff, threshold_settings.bone_hu_cutoff)

    def _ensure_brain_tissue_result_for_current_slice(self):
        """Returns this slice's cached `BrainTissueResult`, computing it (and
        caching it) on first request - `None` if no study is loaded.
        """
        if self._loaded_volume is None:
            return None
        cached = self._brain_tissue_cache.get(self._current_slice_index)
        if cached is not None:
            return cached
        slice_hu = self._loaded_volume.get_slice_hu(self._current_slice_index)
        result = self._tissue_classifier()(slice_hu, self._brain_mask_for_current_slice(slice_hu))
        self._brain_tissue_cache[self._current_slice_index] = result
        return result

    def _tissue_classifier(self):
        """`(slice_hu, brain_mask) -> BrainTissueResult` for the currently
        selected WM/GM method and settings, snapshotted - safe to run on a
        worker thread (see `_start_auto_tissue_worker`) while the user keeps
        changing things in the GUI.
        """
        method = self._brain_tissue_method
        threshold_settings = self._brain_tissue_settings
        fcm_settings = self._fcm_settings
        adaptive_settings = self._adaptive_settings

        def classify(slice_hu: np.ndarray, brain_mask: np.ndarray):
            if method == _BRAIN_TISSUE_METHOD_FCM:
                result = classify_brain_tissue_fcm(
                    slice_hu,
                    brain_mask,
                    fuzziness=fcm_settings.fuzziness,
                    max_iter=fcm_settings.max_iter,
                    spatial_weight=fcm_settings.spatial_weight,
                    min_component_size=fcm_settings.min_component_size,
                )
            elif method == _BRAIN_TISSUE_METHOD_ADAPTIVE:
                result = classify_brain_tissue_adaptive(
                    slice_hu,
                    brain_mask,
                    block_size=adaptive_settings.block_size,
                    plausible_range=(adaptive_settings.plausible_low, adaptive_settings.plausible_high),
                    min_component_size=adaptive_settings.min_component_size,
                )
            else:
                result = classify_brain_tissue(
                    slice_hu,
                    brain_mask,
                    csf_range=(threshold_settings.csf_low, threshold_settings.csf_high),
                    white_matter_range=(threshold_settings.white_matter_low, threshold_settings.white_matter_high),
                    gray_matter_range=(threshold_settings.gray_matter_low, threshold_settings.gray_matter_high),
                    smoothing_size=threshold_settings.smoothing_size,
                    min_component_size=threshold_settings.min_component_size,
                )
            # Not a tissue class from any of these methods, but the same
            # `masks: dict[str, np.ndarray]` shape happily carries it too - see
            # `_build_brain_tissue_box`'s "Brain Mask" checkbox.
            result.masks["Brain Mask"] = brain_mask
            return result

        return classify

    def _on_brain_tissue_method_selected(self, method: str) -> None:
        self._brain_tissue_method = method
        # Every already-computed result was made with the old method -
        # discard it so the next checkbox check/slice view recomputes fresh,
        # same as a settings-dialog change.
        self._brain_tissue_cache = {}
        self._refresh_brain_tissue_for_current_slice()
        self._restart_auto_roi2()

    def _on_fcm_settings_clicked(self) -> None:
        dialog = FcmSettingsDialog(self._fcm_settings, self._fcm_default_settings, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._fcm_settings = dialog.settings()
        self._brain_tissue_cache = {}
        self._refresh_brain_tissue_for_current_slice()
        self._restart_auto_roi2()

    def _on_adaptive_settings_clicked(self) -> None:
        dialog = AdaptiveSettingsDialog(self._adaptive_settings, self._adaptive_default_settings, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._adaptive_settings = dialog.settings()
        self._brain_tissue_cache = {}
        self._refresh_brain_tissue_for_current_slice()
        self._restart_auto_roi2()

    def _apply_brain_tissue_overlay(self, name: str) -> None:
        result = self._ensure_brain_tissue_result_for_current_slice()
        if result is None:
            return
        self._roi_viewport.set_overlay(f"tissue_{name}", result.masks[name], color=_BRAIN_TISSUE_COLORS[name], label=name)

    def _on_brain_tissue_checkbox_toggled(self, name: str, checked: bool) -> None:
        if not checked:
            self._roi_viewport.clear_overlay(f"tissue_{name}")
            return
        self._apply_brain_tissue_overlay(name)

    def _reset_brain_tissue_display(self) -> None:
        self._brain_tissue_cache = {}
        for checkbox in (self._csf_cb, self._white_matter_cb, self._gray_matter_cb, self._brain_mask_cb):
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
            checkbox.setToolTip("Open a study first")
            checkbox.blockSignals(False)
    def _update_brain_mask_checkbox_availability(self) -> None:
        """"Brain Mask" reflects `_brain_mask_for_current_slice`'s
        CNN-segmentation-derived result, which is only meaningful once real
        segmentation data exists for the slice being viewed - unlike White
        Matter/Gray Matter, which stay usable via the HU-threshold fallback
        regardless (see `_brain_mask_for_current_slice`'s docstring). Kept
        disabled, with an explanatory tooltip, until segmentation has
        actually produced a result for the current slice - never silently
        shows the inferior HU-threshold skull-strip under this checkbox.
        """
        if self._loaded_volume is None:
            self._brain_mask_cb.setEnabled(False)
            self._brain_mask_cb.setToolTip("Open a study first")
            return
        weights_path = self._segmentation_config.get("segmentation", {}).get("anatomy_weights_path")
        if not weights_path:
            self._brain_mask_cb.setEnabled(False)
            self._brain_mask_cb.setToolTip("Segmentation model weights not configured")
            return
        if self._current_slice_index not in self._segmentation_cache:
            self._brain_mask_cb.setEnabled(False)
            self._brain_mask_cb.setToolTip("Waiting for anatomy segmentation on this slice…")
            return
        self._brain_mask_cb.setEnabled(True)
        self._brain_mask_cb.setToolTip("")

    def _refresh_brain_tissue_for_current_slice(self) -> None:
        # Whole-volume-scoped checked state (like "Show Segments") - persists
        # across slice navigation; `RoiViewport.set_image` already cleared
        # every overlay by the time this runs (see `_show_slice`), so a
        # checked box's overlay just needs to be reapplied for the new slice.
        self._update_brain_mask_checkbox_availability()
        if self._csf_cb.isChecked():
            self._apply_brain_tissue_overlay("CSF")
        if self._white_matter_cb.isChecked():
            self._apply_brain_tissue_overlay("White Matter")
        if self._gray_matter_cb.isChecked():
            self._apply_brain_tissue_overlay("Gray Matter")
        if self._brain_mask_cb.isChecked() and self._brain_mask_cb.isEnabled():
            self._apply_brain_tissue_overlay("Brain Mask")

    def _build_stroke_box(self) -> QWidget:
        # Kept structurally simple and visually separate from Segmentation
        # (see `domain.segmentation.aisd_stroke_unet`'s module docstring for
        # why): one binary lesion-likelihood overlay, one explicit trigger
        # (not an always-on checkbox, given it's the more experimental of
        # the two models), and a permanent, visible disclaimer - not just a
        # tooltip - since a stray lesion overlay without that context could
        # be genuinely misread as a validated clinical finding.
        box = QGroupBox("Stroke Detection")
        layout = QVBoxLayout(box)

        disclaimer = QLabel("Investigational — not a diagnostic feature")
        disclaimer.setStyleSheet(f"color: {CRITICAL}; font-weight: 600;")
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)

        header = QHBoxLayout()
        self._stroke_status_label = QLabel("—")
        self._stroke_status_label.setProperty("role", "muted")
        header.addWidget(self._stroke_status_label, 1)
        self._check_stroke_btn = QPushButton("Check for Lesion")
        self._check_stroke_btn.setEnabled(False)
        self._check_stroke_btn.setToolTip("Open a study first")
        self._check_stroke_btn.clicked.connect(self._on_check_stroke_clicked)
        header.addWidget(self._check_stroke_btn)
        layout.addLayout(header)

        self._stroke_lesion_cb = QCheckBox("Show suspected lesion")
        self._stroke_lesion_cb.setEnabled(False)
        self._stroke_lesion_cb.setStyleSheet(f"QCheckBox {{ color: {CRITICAL}; }}")
        self._stroke_lesion_cb.toggled.connect(self._on_stroke_checkbox_toggled)
        layout.addWidget(self._stroke_lesion_cb)

        return box

    def _on_check_stroke_clicked(self) -> None:
        if self._loaded_volume is None:
            return
        cached = self._stroke_cache.get(self._current_slice_index)
        if cached is not None:
            self._apply_stroke_result(cached, from_cache=True)
            return

        weights_path = self._segmentation_config.get("segmentation", {}).get("stroke_weights_path")
        if not weights_path:
            QMessageBox.warning(
                self,
                "Stroke model not configured",
                "No stroke segmentation weights are configured. Set BICTOOL_STROKE_WEIGHTS "
                "or the segmentation.stroke_weights_path config key - see "
                "domain.segmentation.aisd_stroke_unet for where to obtain the weights file.",
            )
            return

        try:
            # Deferred to first actual use - see the TYPE_CHECKING import
            # block's comment for why (TensorFlow's import cost).
            from bictool.application.stroke_worker import StrokeWorker
            from bictool.domain.segmentation import aisd_stroke_unet
        except ImportError as exc:
            QMessageBox.critical(
                self, "TensorFlow not installed", f"Stroke detection requires TensorFlow (pip install bictool[ml]): {exc}"
            )
            return

        if self._stroke_model is None:
            self._stroke_model = aisd_stroke_unet.build_model()
        try:
            aisd_stroke_unet.load_weights(self._stroke_model, weights_path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Stroke model weights not found", str(exc))
            return

        logger.info("Checking slice %d for a stroke lesion...", self._current_slice_index)
        self._stroke_status_label.setText("Checking…")
        self._check_stroke_btn.setEnabled(False)
        slice_hu = self._loaded_volume.get_slice_hu(self._current_slice_index)
        worker = StrokeWorker(self._stroke_model, slice_hu)
        worker.signals.finished.connect(self._on_stroke_finished)
        worker.signals.error.connect(self._on_stroke_error)
        self._active_stroke_worker = worker  # keep a reference alive while it runs
        self._scheduler.submit(worker)

    def _on_stroke_finished(self, result) -> None:
        logger.info("Stroke check finished for slice %d.", self._current_slice_index)
        self._active_stroke_worker = None
        self._stroke_cache[self._current_slice_index] = result
        self._apply_stroke_result(result, from_cache=False)

    def _apply_stroke_result(self, result, *, from_cache: bool) -> None:
        self._check_stroke_btn.setEnabled(True)
        self._stroke_status_label.setText("Done (cached)" if from_cache else "Done")
        self._stroke_mask_for_current_slice = result.masks["Lesion"]
        self._stroke_lesion_cb.setEnabled(True)
        if self._stroke_lesion_cb.isChecked():
            self._roi_viewport.set_overlay("stroke_lesion", self._stroke_mask_for_current_slice, color=CRITICAL, label="Suspected Lesion")

    def _on_stroke_error(self, message: str) -> None:
        logger.error("Stroke check failed: %s", message)
        self._active_stroke_worker = None
        self._check_stroke_btn.setEnabled(True)
        self._stroke_status_label.setText("Error")
        self._stroke_status_label.setToolTip(message)

    def _on_stroke_checkbox_toggled(self, checked: bool) -> None:
        if not checked:
            self._roi_viewport.clear_overlay("stroke_lesion")
            return
        if self._stroke_mask_for_current_slice is not None:
            self._roi_viewport.set_overlay("stroke_lesion", self._stroke_mask_for_current_slice, color=CRITICAL, label="Suspected Lesion")

    # ---- segmentation/stroke lifecycle (volume load, slice change, close study) ----

    def _update_segmentation_availability(self) -> None:
        """Enables/disables the Segment Slice and Check for Lesion buttons
        based on whether their respective weights are configured (see
        `infrastructure.config`) - checked once per volume load rather than
        on every slice change, since configuration doesn't change mid-session.
        """
        segmentation_config = self._segmentation_config.get("segmentation", {})
        has_anatomy_weights = bool(segmentation_config.get("anatomy_weights_path"))
        self._segment_slice_btn.setEnabled(has_anatomy_weights)
        self._segment_slice_btn.setToolTip("" if has_anatomy_weights else "Segmentation model weights not configured")
        self._segment_volume_btn.setEnabled(has_anatomy_weights)
        self._segment_volume_btn.setToolTip("" if has_anatomy_weights else "Segmentation model weights not configured")
        self._select_segments_btn.setEnabled(has_anatomy_weights)
        self._select_segments_btn.setToolTip("" if has_anatomy_weights else "Segmentation model weights not configured")
        self._show_segments_cb.setEnabled(has_anatomy_weights)
        self._show_segments_cb.setToolTip("" if has_anatomy_weights else "Segmentation model weights not configured")

        has_stroke_weights = bool(segmentation_config.get("stroke_weights_path"))
        self._check_stroke_btn.setEnabled(has_stroke_weights)
        self._check_stroke_btn.setToolTip("" if has_stroke_weights else "Stroke model weights not configured")

        # No weights/config to gate on (HU-threshold only) - always available
        # once a study is loaded.
        self._csf_cb.setEnabled(True)
        self._csf_cb.setToolTip("")
        self._white_matter_cb.setEnabled(True)
        self._white_matter_cb.setToolTip("")
        self._gray_matter_cb.setEnabled(True)
        self._gray_matter_cb.setToolTip("")
        # "Brain Mask" has its own, segmentation-dependent availability - see
        # `_update_brain_mask_checkbox_availability`.
        self._update_brain_mask_checkbox_availability()

    def _reset_segmentation_display(self) -> None:
        """Full reset for a new/closed study - clears the per-volume cache
        too (unlike `_refresh_segmentation_for_current_slice`, which only
        resets the current slice's on-screen state without discarding
        other slices' already-computed results).

        A single-slice or whole-volume job may still be in flight (each
        slice is only ~1-2s now, but a multi-hundred-slice volume batch is
        still a multi-minute background job - see `_on_segment_volume_
        clicked`) when this fires - `cancel()` stops the volume worker from
        starting further slices, but the slice already running inside
        `model.predict()` can't be interrupted and will still emit its
        signal(s) later; `blockSignals(True)` on both workers' signal
        objects ensures that late emission is silently dropped instead of
        writing a stale result into what could by then
        be a different volume's cache (the same class of bug as the
        stale-cache-under-a-new-volume fix below, just for a job that
        outlives the reset instead of one that already finished).
        """
        if self._active_segmentation_worker is not None:
            self._active_segmentation_worker.signals.blockSignals(True)
            self._active_segmentation_worker = None
        if self._active_volume_segmentation_worker is not None:
            self._active_volume_segmentation_worker.cancel()
            self._active_volume_segmentation_worker.signals.blockSignals(True)
            self._active_volume_segmentation_worker = None
        if self._active_nifti_conversion_worker is not None:
            self._active_nifti_conversion_worker.signals.blockSignals(True)
            self._active_nifti_conversion_worker = None
        self._dicom_nifti_volume = None  # a different/closed study needs its own fresh conversion
        self._segmentation_cache = {}
        self._reset_auto_rois()  # derived from the segmentation cache
        self._segmentation_masks_for_current_slice = None
        # The persisted structure selection is whole-volume-scoped by
        # design (see `__init__`'s comment) - a new/closed study means a
        # different, not-yet-segmented volume, so the old selection no
        # longer applies and is reset back to its default (all 16
        # structures selected) along with everything else here.
        self._selected_segmentation_names = set(_SEGMENTATION_MAPPING)
        self._segment_slice_btn.setEnabled(False)
        self._segment_slice_btn.setToolTip("Open a study first")
        self._segment_volume_btn.setEnabled(False)
        self._segment_volume_btn.setToolTip("Open a study first")
        self._cancel_segment_volume_btn.setVisible(False)
        self._select_segments_btn.setEnabled(False)
        self._select_segments_btn.setToolTip("Open a study first")
        self._show_segments_cb.setEnabled(False)
        self._show_segments_cb.setToolTip("Open a study first")
        self._show_segments_cb.blockSignals(True)
        self._show_segments_cb.setChecked(False)  # back to its default (2026-09-24) - opt in, not on by default
        self._show_segments_cb.blockSignals(False)
        self._segmentation_status_label.setText("—")
        self._segmentation_status_label.setToolTip("")
        self._segmentation_opacity = 0.6  # matches the dialog's own slider reset, below
        if self._segment_selection_dialog is not None:
            self._segment_selection_dialog.reset()

    def _reset_stroke_display(self) -> None:
        if self._active_stroke_worker is not None:
            self._active_stroke_worker.signals.blockSignals(True)
            self._active_stroke_worker = None
        self._stroke_cache = {}
        self._stroke_mask_for_current_slice = None
        self._check_stroke_btn.setEnabled(False)
        self._check_stroke_btn.setToolTip("Open a study first")
        self._stroke_status_label.setText("—")
        self._stroke_status_label.setToolTip("")
        self._stroke_lesion_cb.blockSignals(True)
        self._stroke_lesion_cb.setChecked(False)
        self._stroke_lesion_cb.setEnabled(False)
        self._stroke_lesion_cb.blockSignals(False)

    def _refresh_segmentation_for_current_slice(self) -> None:
        """Per-slice-change refresh: re-applies the *persisted* structure
        selection (`_selected_segmentation_names`) using this slice's
        cached result, if one exists - the selection itself is
        whole-volume-scoped and is never touched here (only
        `_reset_segmentation_display` clears it, on close study/new volume
        load - see that method and `__init__`'s comment on why). The
        overlays themselves are already gone by the time this runs -
        `RoiViewport.set_image` (called just before this, in `_show_slice`)
        clears all overlays on every slice change - so there's nothing to
        explicitly clear when no cached result exists yet for the new
        slice: the selected structures simply reappear once this slice
        actually gets segmented.
        """
        self._segmentation_masks_for_current_slice = None
        cached = self._segmentation_cache.get(self._current_slice_index)
        if cached is not None:
            self._apply_segmentation_result(cached, from_cache=True)
        else:
            self._segmentation_status_label.setText("—")

    def _refresh_stroke_for_current_slice(self) -> None:
        self._stroke_mask_for_current_slice = None
        self._stroke_lesion_cb.blockSignals(True)
        self._stroke_lesion_cb.setChecked(False)
        self._stroke_lesion_cb.setEnabled(False)
        self._stroke_lesion_cb.blockSignals(False)
        cached = self._stroke_cache.get(self._current_slice_index)
        if cached is not None:
            self._apply_stroke_result(cached, from_cache=True)
        else:
            self._stroke_status_label.setText("—")

    def _build_roi_box(self) -> QWidget:
        # Structured table (ROI/Position/Size), not per-ROI QLabels: the old
        # labels' full coordinate text ("ROI1: 134-384 . 96-346 (250x250
        # px)") didn't fit this box's half of the sidebar row, forcing the
        # box wider - which grew the sidebar and shrank the viewer pane by
        # the same amount, visibly shifting the image left the moment an ROI
        # was drawn. A table's columns size to the sidebar's actual width
        # instead of forcing it wider, and it reads more like the Run
        # Metrics table already used elsewhere in this sidebar.
        box = QGroupBox("ROI Selection")
        layout = QVBoxLayout(box)

        # Set ROI1/2 live here (not the viewer toolbar) - they're an
        # ROI-selection action, grouped with the rest of that workflow, and
        # this box is already hidden on the Results page along with them.
        # Draw-then-assign: drawing a rectangle on the image (any drag, no
        # arming needed - see `RoiViewport`) enables these; clicking one
        # saves that rectangle as that slot (`_on_set_roi_clicked`).
        set_roi_row = QHBoxLayout()
        self._set_roi1_btn = QPushButton("Set ROI1")
        self._set_roi1_btn.setEnabled(False)
        self._set_roi1_btn.setToolTip("Draw a rectangle on the image first")
        self._set_roi1_btn.clicked.connect(lambda: self._on_set_roi_clicked("roi1"))
        self._set_roi2_btn = QPushButton("Set ROI2")
        self._set_roi2_btn.setEnabled(False)
        self._set_roi2_btn.setToolTip("Draw a rectangle on the image first")
        self._set_roi2_btn.clicked.connect(lambda: self._on_set_roi_clicked("roi2"))
        set_roi_row.addWidget(self._set_roi1_btn)
        set_roi_row.addWidget(self._set_roi2_btn)
        layout.addLayout(set_roi_row)

        # Auto ROI1 (domain.roi_selection.structured_roi): while checked, a
        # candidate is computed for every segmented slice and shown as the
        # draft on whichever slice is displayed - movable, then accepted with
        # Set ROI1 like a hand-drawn one. Two target structures so they can
        # be compared side by side.
        auto_row = QHBoxLayout()
        self._auto_rois_cb = QCheckBox("Auto ROIs")
        self._auto_rois_cb.setEnabled(False)
        self._auto_rois_cb.setToolTip("Needs anatomy segmentation first")
        self._auto_rois_cb.toggled.connect(self._on_auto_rois_toggled)
        self._auto_roi1_septum_rb = QRadioButton("Septum")
        self._auto_roi1_septum_rb.setToolTip(SEPTUM_PELLUCIDUM)
        self._auto_roi1_septum_rb.setChecked(True)
        self._auto_roi1_dural_rb = QRadioButton("Dural folds")
        self._auto_roi1_dural_rb.setToolTip(f"{DURAL_FOLDS} (falx + tentorium)")
        self._auto_roi1_structure_group = QButtonGroup(box)
        self._auto_roi1_structure_group.addButton(self._auto_roi1_septum_rb)
        self._auto_roi1_structure_group.addButton(self._auto_roi1_dural_rb)
        self._auto_roi1_structure_group.buttonToggled.connect(self._on_auto_roi1_structure_toggled)
        auto_row.addWidget(self._auto_rois_cb)
        auto_row.addWidget(self._auto_roi1_septum_rb)
        auto_row.addWidget(self._auto_roi1_dural_rb)
        auto_row.addStretch(1)
        layout.addLayout(auto_row)
        self._auto_rois_status_label = QLabel("")
        self._auto_rois_status_label.setProperty("role", "muted")
        self._auto_rois_status_label.setWordWrap(True)
        layout.addWidget(self._auto_rois_status_label)

        self._roi_table = QTableWidget(2, 3)
        self._roi_table.setHorizontalHeaderLabels(["ROI", "Position", "Size"])
        self._roi_table.verticalHeader().setVisible(False)
        self._roi_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._roi_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._roi_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._roi_table.setShowGrid(False)
        header_view = self._roi_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._roi_table.setStyleSheet(
            f"QTableWidget {{ background: {SURFACE}; border: none; }}"
            f"QHeaderView::section {{ background: {SURFACE}; color: {TEXT_MUTED}; border: none;"
            f" border-bottom: 1px solid {BORDER}; padding: 4px; font-weight: 600; font-size: 10px; }}"
            f"QTableWidget::item {{ padding: 4px; border-bottom: 1px solid {BORDER}; }}"
        )

        # Same colors `RoiViewport` draws the rectangles in (cyan ROI1, pink
        # ROI2), so the row and the rectangle it describes are visually tied
        # together without needing a legend.
        self._roi_position_items: dict[str, QTableWidgetItem] = {}
        self._roi_size_items: dict[str, QTableWidgetItem] = {}
        for row, (key, color) in enumerate((("roi1", ACCENT), ("roi2", ACCENT_2))):
            name_item = QTableWidgetItem(key.upper())
            name_item.setForeground(QColor(color))
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            position_item = QTableWidgetItem("not set")
            position_item.setForeground(QColor(TEXT_MUTED))
            position_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            size_item = QTableWidgetItem("—")
            size_item.setForeground(QColor(TEXT_MUTED))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._roi_table.setItem(row, 0, name_item)
            self._roi_table.setItem(row, 1, position_item)
            self._roi_table.setItem(row, 2, size_item)
            self._roi_position_items[key] = position_item
            self._roi_size_items[key] = size_item

        # Exactly 2 rows, always - a fixed height (not `QTableWidget`'s
        # default `Expanding` vertical size policy) keeps this box compact
        # and stops it silently soaking up sidebar space that should go to
        # Run Metrics (which actually needs it - see project memory: this is
        # the same "unbounded table" pattern already root-caused there once).
        self._roi_table.resizeRowsToContents()
        table_height = self._roi_table.horizontalHeader().sizeHint().height()
        table_height += sum(self._roi_table.rowHeight(r) for r in range(self._roi_table.rowCount()))
        self._roi_table.setFixedHeight(table_height + 2)  # +2 for the frame
        layout.addWidget(self._roi_table)

        self._confirm_rois_cb = QCheckBox("Confirmed")
        self._confirm_rois_cb.setEnabled(False)
        self._confirm_rois_cb.toggled.connect(self._on_confirm_rois_toggled)
        layout.addWidget(self._confirm_rois_cb)
        return box

    def _update_roi_table_row(self, key: str, rect: RoiRect) -> None:
        position_item = self._roi_position_items[key]
        position_item.setText(f"{rect.x0}–{rect.x1}, {rect.y0}–{rect.y1}")
        position_item.setForeground(QColor(TEXT))
        size_item = self._roi_size_items[key]
        size_item.setText(f"{rect.width}×{rect.height} px")
        size_item.setForeground(QColor(TEXT))

    def _reset_roi_table(self) -> None:
        for key in ("roi1", "roi2"):
            self._reset_roi_table_row(key)

    def _reset_roi_table_row(self, key: str) -> None:
        self._roi_position_items[key].setText("not set")
        self._roi_position_items[key].setForeground(QColor(TEXT_MUTED))
        self._roi_size_items[key].setText("—")
        self._roi_size_items[key].setForeground(QColor(TEXT_MUTED))

    def _on_confirm_rois_toggled(self, checked: bool) -> None:
        """Confirmed = the ROIs on every slice are final: locked on the image,
        Set ROI1/ROI2 and Auto ROIs disabled, Run All enabled. Unchecking
        makes them editable (draggable) again.
        """
        self._run_all_btn.setEnabled(checked)
        self._run_all_btn.setToolTip("" if checked else "Confirm ROIs to enable")
        self._roi_viewport.set_locked(checked)
        self._update_set_roi_buttons()
        self._update_auto_rois_availability()

    def _build_metrics_box(self) -> QWidget:
        box = QGroupBox("Run Metrics")
        layout = QVBoxLayout(box)

        header = QHBoxLayout()
        self._qa_progress_label = QLabel("")
        self._qa_progress_label.setProperty("role", "muted")
        header.addWidget(self._qa_progress_label)
        header.addStretch(1)
        self._cancel_run_btn = QPushButton("Cancel")
        self._cancel_run_btn.setVisible(False)
        self._cancel_run_btn.clicked.connect(self._on_cancel_run_clicked)
        header.addWidget(self._cancel_run_btn)
        self._run_all_btn = QPushButton("▶ Run All")
        self._run_all_btn.setObjectName("runAllBtn")
        self._run_all_btn.setEnabled(False)
        self._run_all_btn.setToolTip("Confirm ROIs to enable")
        self._run_all_btn.clicked.connect(self._on_run_all_clicked)
        header.addWidget(self._run_all_btn)
        layout.addLayout(header)

        # Table (Metric/Status/Value), not per-row QLabels: rows keep their
        # natural height and any leftover vertical space in a tall window
        # shows as plain table background below the last row, instead of
        # each row individually stretching to fill it.
        self._metrics_table = QTableWidget(len(_METRIC_ROWS), 3)
        self._metrics_table.setHorizontalHeaderLabels(["Metric", "Status", "Value"])
        self._metrics_table.verticalHeader().setVisible(False)
        self._metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._metrics_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._metrics_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._metrics_table.setShowGrid(False)
        header_view = self._metrics_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._metrics_table.setStyleSheet(
            f"QTableWidget {{ background: {SURFACE}; border: none; }}"
            f"QHeaderView::section {{ background: {SURFACE}; color: {TEXT_MUTED}; border: none;"
            f" border-bottom: 1px solid {BORDER}; padding: 4px; font-weight: 600; font-size: 10px; }}"
            f"QTableWidget::item {{ padding: 4px; border-bottom: 1px solid {BORDER}; }}"
        )

        self._metric_status_items: dict[str, QTableWidgetItem] = {}
        self._metric_value_items: dict[str, QTableWidgetItem] = {}
        for row, (key, name) in enumerate(_METRIC_ROWS):
            name_item = QTableWidgetItem(name)
            name_item.setForeground(QColor(TEXT))
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # "—" (not "queued") - nothing is actually queued until Run All
            # is clicked; "queued" as a permanent idle state was misleading.
            status_item = QTableWidgetItem("—")
            status_item.setForeground(QColor(TEXT_MUTED))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            value_item = QTableWidgetItem("")
            value_item.setForeground(QColor(TEXT))
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._metrics_table.setItem(row, 0, name_item)
            self._metrics_table.setItem(row, 1, status_item)
            self._metrics_table.setItem(row, 2, value_item)
            self._metric_status_items[key] = status_item
            self._metric_value_items[key] = value_item
        layout.addWidget(self._metrics_table, 1)

        self._result_strip = QWidget()
        self._result_strip.setVisible(False)
        strip_layout = QHBoxLayout(self._result_strip)
        strip_layout.setContentsMargins(0, 8, 0, 0)
        strip_layout.addStretch(1)
        export_btn = QPushButton("Export…")
        # No enabled/disabled dance needed here (unlike the menu action) -
        # `_result_strip` (this button's parent) is only ever shown once a
        # result exists (`_on_worker_finished`) and hidden again on Close
        # Study (`_reset_metrics_display`), so this button is only reachable
        # when export is actually possible.
        export_btn.clicked.connect(self._open_export_dialog)
        strip_layout.addWidget(export_btn)
        layout.addWidget(self._result_strip)

        return box

    def _set_metric_status(self, key: str, role: str, text: str, tooltip: str = "") -> None:
        item = self._metric_status_items[key]
        item.setText(text)
        item.setForeground(QColor(_STATUS_COLORS[role]))
        item.setToolTip(tooltip)

    def _on_run_all_clicked(self) -> None:
        """QA for every slice with both ROIs (the confirmed ones) - local QA
        per slice, then the study-level summary on the Results page.
        """
        slices = self._qa_ready_slices()
        if not slices:
            return
        self._qa_results = {}
        self._qa_errors = {}
        self._qa_jobs = list(slices)
        self._qa_total = len(slices)
        self._qa_cancel_requested = False
        logger.info("Run All: QA on %d slice(s): %s", len(slices), _format_slice_ranges(slices))
        for key, _ in _METRIC_ROWS:
            self._set_metric_status(key, "statusQueued", "running")
        self._run_all_btn.setEnabled(False)
        self._cancel_run_btn.setText("Cancel")
        self._cancel_run_btn.setEnabled(True)
        self._cancel_run_btn.setVisible(True)
        self._confirm_rois_cb.setEnabled(False)
        self._update_qa_progress()
        self._run_next_qa_slice()

    def _run_next_qa_slice(self) -> None:
        if not self._qa_jobs or self._qa_cancel_requested:
            self._finish_qa_batch()
            return
        index = self._qa_jobs.pop(0)
        rois = self._effective_rois(index)
        pixels = self._loaded_volume.get_slice_hu(index)
        img_roi1 = crop_roi(pixels, rois["roi1"])
        img_roi2 = crop_roi(pixels, rois["roi2"])
        pix_spc = self._loaded_volume.pixel_spacing_mm
        # calc_mtf's active contour must start inside the edge structure, so
        # seed it on the structure's centerline (segmentation as a hint where
        # this slice has one), with fallbacks if a seed yields no edges.
        seeds = mtf_seed_candidates(img_roi1, self._mtf_structure_mask(index, rois["roi1"]))
        logger.info(
            "Run All clicked: slice=%d, roi1 shape=%s, roi2 shape=%s, pix_spc=%.4f, mtf seeds=%s",
            index,
            img_roi1.shape,
            img_roi2.shape,
            pix_spc,
            seeds,
        )
        worker = Worker(img_roi1, img_roi2, pix_spc, mtf_seeds=seeds)
        # Bound methods (not lambdas) so Qt queues the call onto the GUI
        # thread; which slice it was is tracked in `_qa_active_slice`.
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        self._qa_active_slice = index
        self._active_worker = worker  # keep a reference alive while it runs
        self._scheduler.submit(worker)

    def _mtf_structure_mask(self, slice_index: int, rect: RoiRect) -> np.ndarray | None:
        """The ROI1 structures (septum + dural folds) cropped to `rect`, or None
        if the slice isn't segmented - only a hint for placing the MTF seed.
        """
        result = self._segmentation_cache.get(slice_index)
        if result is None:
            return None
        masks = [result.masks[name] for name in STRUCTURES if name in result.masks]
        if not masks:
            return None
        return crop_roi(np.logical_or.reduce(masks), rect)

    def _on_worker_finished(self, result: SliceQualityResult) -> None:
        index = self._qa_active_slice
        logger.info(
            "Run All succeeded: slice=%d, mtf_area=%.4f, predicted_label=%s, decision_value=%.4f",
            index,
            result.mtf_area,
            result.predicted_label,
            result.decision_value,
        )
        self._active_worker = None
        self._qa_results[index] = result
        self._refresh_metrics_box()
        self._update_qa_progress()
        self._run_next_qa_slice()

    def _on_worker_error(self, message: str) -> None:
        index = self._qa_active_slice
        logger.error("Run All failed on slice %d: %s", index, message)
        self._active_worker = None
        self._qa_errors[index] = message
        self._refresh_metrics_box()
        self._update_qa_progress()
        self._run_next_qa_slice()

    def _on_cancel_run_clicked(self) -> None:
        # The slice already running can't be interrupted; no further ones start.
        self._qa_cancel_requested = True
        self._cancel_run_btn.setText("Cancelling…")
        self._cancel_run_btn.setEnabled(False)

    def _finish_qa_batch(self) -> None:
        cancelled = self._qa_cancel_requested
        self._qa_jobs = None
        self._qa_active_slice = None
        self._qa_cancel_requested = False
        self._cancel_run_btn.setVisible(False)
        self._run_all_btn.setEnabled(self._confirm_rois_cb.isChecked())
        self._update_confirm_availability()
        self._update_qa_progress(cancelled=cancelled)
        self._refresh_metrics_box()
        if not self._qa_results:
            return
        self._results_tab_btn.setEnabled(True)
        self._results_tab_btn.setToolTip("")
        self._export_action.setEnabled(True)
        self._export_action.setToolTip("")
        self._result_strip.setVisible(True)
        self._refresh_results_page()
        shown = self._current_slice_index if self._current_slice_index in self._qa_results else min(self._qa_results)
        self._show_result_for_slice(shown)
        self._show_view_page(1)

    def _update_qa_progress(self, cancelled: bool = False) -> None:
        done = len(self._qa_results) + len(self._qa_errors)
        failed = f", {len(self._qa_errors)} failed" if self._qa_errors else ""
        if self._qa_jobs is not None:
            text = f"QA {done}/{self._qa_total}{failed}"
        elif self._qa_total == 0:
            text = ""
        elif cancelled:
            text = f"QA cancelled: {done}/{self._qa_total} slice(s){failed}"
        else:
            text = f"QA done: {done} slice(s){failed}"
        self._qa_progress_label.setText(text)

    def _on_results_tab_changed(self, _tab_index: int) -> None:
        self._refresh_metrics_box()

    def _showing_study_summary(self) -> bool:
        return self._view_stack.currentIndex() == 1 and self._results_tabs.currentIndex() == 1

    def _refresh_metrics_box(self) -> None:
        """Run Metrics box = study averages while the Study summary tab is
        shown, else the displayed slice's metrics (the Per slice tab keeps the
        viewer on the same slice).
        """
        if self._showing_study_summary() and self._qa_results:
            self._show_study_average_metrics()
        else:
            self._show_metrics_for_slice(self._current_slice_index)

    def _show_study_average_metrics(self) -> None:
        summary = summarize_metrics(self._qa_results.values())
        # Run Metrics row -> (summary metric, value format, unit), matching the per-slice rows.
        rows = {
            "mtf": ("mtf_area", "{:.3f}", " mm⁻¹"),
            "nps": ("nps_variance", "{:.5f}", " HU²mm²"),
            "ratio": ("ps_mtf_ratio", "{:.2f}", " LF/HF"),
            "contrast": ("contrast", "{:.1f}", " HU"),
        }
        for key, _ in _METRIC_ROWS:
            name, number, unit = rows[key]
            metric = summary.get(name)
            if metric is None:
                self._set_metric_status(key, "statusQueued", "—")
                self._metric_value_items[key].setText("")
                continue
            self._set_metric_status(
                key,
                "statusDone",
                "mean",
                tooltip=f"Mean over {metric.count} slice(s), SD {number.format(metric.std)}{unit}",
            )
            self._metric_value_items[key].setText(f"{number.format(metric.mean)}{unit}")

    def _show_metrics_for_slice(self, index: int) -> None:
        """Sidebar metrics table = the displayed slice's QA result (or error,
        or "—" if it has none).
        """
        result = self._qa_results.get(index)
        error = self._qa_errors.get(index)
        if result is not None:
            self._set_metric_status("mtf", "statusDone", "done")
            self._metric_value_items["mtf"].setText(f"{result.mtf_area:.3f} mm⁻¹")
            self._set_metric_status("nps", "statusDone", "done")
            # nps_roi2 (homogeneous ROI) is the true NPS - nps_roi1 (structured
            # ROI) is what WP2 calls "PS" (see the PS chart); this row was
            # previously showing nps_roi1, mislabeling PS as NPS.
            self._metric_value_items["nps"].setText(f"{result.nps_roi2.var_nps:.5f} HU²mm²")
            self._set_metric_status("ratio", "statusDone", "done")
            self._metric_value_items["ratio"].setText(f"{result.ratio.ratio_lf_hf:.2f} LF/HF")
            self._set_metric_status("contrast", "statusDone", "done")
            self._metric_value_items["contrast"].setText(f"{result.contrast_roi1:.1f} HU")
        elif error is not None:
            for key, _ in _METRIC_ROWS:
                self._set_metric_status(key, "statusError", "error", tooltip=error)
                self._metric_value_items[key].setText("")
        elif self._qa_jobs is not None and index in self._qa_jobs + [self._qa_active_slice]:
            for key, _ in _METRIC_ROWS:
                self._set_metric_status(key, "statusQueued", "running")
                self._metric_value_items[key].setText("")
        else:
            for key, _ in _METRIC_ROWS:
                self._set_metric_status(key, "statusQueued", "—")
                self._metric_value_items[key].setText("")

    def _reset_qa_results(self) -> None:
        """Drops every QA result (new/closed study) and stops a running batch."""
        if self._active_worker is not None:
            self._active_worker.signals.blockSignals(True)
            self._active_worker = None
        self._qa_jobs = None
        self._qa_active_slice = None
        self._qa_cancel_requested = False
        self._qa_total = 0
        self._qa_results = {}
        self._qa_errors = {}
        self._cancel_run_btn.setVisible(False)
        self._update_qa_progress()
        self._refresh_results_page()
        self._results_slice_label.setText("No results yet")
        self._reset_metrics_display()
        self._results_tab_btn.setEnabled(False)
        self._results_tab_btn.setToolTip("Run metrics to see results")
        self._export_action.setEnabled(False)
        self._export_action.setToolTip("Run metrics first")
        self._last_result = None
        self._last_export_context = None

    def _reset_metrics_display(self) -> None:
        """Back to the idle, nothing-has-run-yet state - `_build_metrics_box`'s
        own initial state, mirrored here for "Close Study" so nothing about
        a previous study's result lingers on screen once it's closed.
        """
        for key, _ in _METRIC_ROWS:
            self._set_metric_status(key, "statusQueued", "—")
            self._metric_value_items[key].setText("")
        self._result_verdict_label.setText("—")
        self._result_verdict_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px; font-weight: 700;")
        self._result_decision_label.setText("")
        for name in _FEATURE_NAMES:
            self._feature_labels[name].setText("—")
        for chart_view in (self._mtf_chart_view, self._nps_chart_view, self._ps_chart_view, self._ratio_chart_view):
            self._update_line_chart(chart_view, np.array([]), [])
        self._result_strip.setVisible(False)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindowView()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
