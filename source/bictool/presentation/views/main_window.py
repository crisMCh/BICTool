"""Main application window (View layer).

Structural port of `design/bictool_console_mockup.html`'s single fixed window:
a menu bar, a large viewer/results pane on the left, and a sidebar (Series /
Segmentation + ROI Selection / Run Metrics) on the right.

MVVM: the window holds no application state. `self.view_model`
(`viewmodels.main_viewmodel.MainViewModel`) owns one ViewModel per area and
the workflow between them; the window builds the widgets, binds them to the
ViewModels' signals, renders and forwards user input as commands.

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
  The MainViewModel auto-starts this same "Segment Volume" batch,
  with no user action at all, the moment the study's primary series (the
  one `discover_series` auto-selects and loads by default - see
  `StudyViewModel.select_series`) finishes loading, if anatomy weights are
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
  request). The selection itself (`SegmentationViewModel.selected_names`) is a
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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydicom
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QActionGroup, QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QScrollArea,
    QStackedWidget,
    QSplitter,
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
    QMenu,
    QMessageBox,
    QButtonGroup,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bictool.application.dicom_load_worker import DicomLoadWorker
from bictool.application.scheduler import Scheduler
from bictool.application.worker import Worker
from bictool.domain.results.study_quality import (
    RECIPE_OCTAVE,
    RECIPE_PHYSICAL,
    SliceQualityResult,
)
from bictool.domain.results.study_summary import summarize_metrics
from bictool.domain.roi_selection.homogeneous_roi import (
    FALLBACK,
    VENTRICLE,
    WHITE_MATTER,
)
from bictool.domain.roi_selection.structured_roi import (
    DURAL_FOLDS,
    SEPTUM_PELLUCIDUM,
)
from bictool.domain.segmentation.brain_tissue import (
    DEFAULT_AIR_HU_CUTOFF,
    DEFAULT_BONE_HU_CUTOFF,
    DEFAULT_CSF_RANGE,
    DEFAULT_GRAY_MATTER_RANGE,
    DEFAULT_MIN_COMPONENT_SIZE,
    DEFAULT_SMOOTHING_SIZE,
    DEFAULT_WHITE_MATTER_RANGE,
)
from bictool.domain.segmentation.brain_tissue_adaptive import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PLAUSIBLE_RANGE,
)
from bictool.domain.segmentation.brain_tissue_adaptive import (
    DEFAULT_MIN_COMPONENT_SIZE as ADAPTIVE_DEFAULT_MIN_COMPONENT_SIZE,
)
from bictool.domain.segmentation.brain_tissue_adaptive import (
    DEFAULT_SMOOTHING_MM as ADAPTIVE_DEFAULT_SMOOTHING_MM,
)
from bictool.domain.segmentation.brain_tissue_adaptive import (
    DEFAULT_USE_STUDY_RANGE as ADAPTIVE_DEFAULT_USE_STUDY_RANGE,
)
from bictool.domain.segmentation.brain_tissue_fcm import (
    DEFAULT_FUZZINESS,
    DEFAULT_MAX_ITER,
    DEFAULT_SPATIAL_WEIGHT,
)
from bictool.domain.segmentation.brain_tissue_fcm import (
    DEFAULT_MIN_COMPONENT_SIZE as FCM_DEFAULT_MIN_COMPONENT_SIZE,
)
from bictool.infrastructure.config import load_config
from bictool.infrastructure.logging_config import QtLogHandler, configure_logging
from bictool.infrastructure.ui_settings import (
    MAIN_SPLITTER,
    ROI_DETAILS_OPEN,
    SHOW_TRACE_LOG,
    TRACE_SPLITTER,
    VERDICT_DETAILS_OPEN,
    UiSettings,
    section_key,
)
from bictool.integration.dicom_file_loader import DicomVolume, SeriesCandidate, load_series
from bictool.integration.model_files import default_head_ct_model
from bictool.presentation.viewmodels.chart_data import mtf_display_curve
from bictool.presentation.viewmodels.display_text import format_dicom_date
from bictool.presentation.viewmodels.head_ct_verdict_text import (
    PROVISIONAL_NOTE,
    PROVISIONAL_SHORT,
    TONE_BAD,
    TONE_GOOD,
    TONE_NEUTRAL,
)
from bictool.presentation.viewmodels.main_viewmodel import MainViewModel
from bictool.presentation.viewmodels.help_text import STEP_HINTS, TAB_TOOLTIPS, TOOLTIPS
from bictool.presentation.viewmodels.roi_pairs import (
    FALX,
    PAIR_LABELS,
    PAIRS,
    ROI_KEYS,
    SEPTUM,
    ROLE_LABELS,
    ROLE_SHORT,
    key_label,
)
from bictool.presentation.viewmodels.brain_tissue_viewmodel import (
    METHOD_ADAPTIVE as _BRAIN_TISSUE_METHOD_ADAPTIVE,
    METHOD_FCM as _BRAIN_TISSUE_METHOD_FCM,
    METHOD_THRESHOLD as _BRAIN_TISSUE_METHOD_THRESHOLD,
    METHODS as _BRAIN_TISSUE_METHODS,
)
from bictool.presentation.viewmodels.roi_selection import RoiRect
from bictool.presentation.viewmodels.segmentation_overlay import decode_masks_for_checkboxes
from bictool.presentation.viewmodels.windowing import DEFAULT_WINDOW_PRESET, WINDOW_PRESETS
from bictool.presentation.views.app_icon import app_icon, register_windows_app_id
from bictool.presentation.views.adaptive_settings_dialog import AdaptiveSettings, AdaptiveSettingsDialog
from bictool.presentation.views.collapsible_section import CollapsibleSection
from bictool.presentation.views.slice_steps_panel import SliceStepsPanel
from bictool.presentation.views.status_bar import StatusPanel
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
    from bictool.application.volume_segmentation_worker import VolumeSegmentationWorker
    # `NiftiConversionWorker` doesn't import TensorFlow, but does import
    # `dicom2nifti` (a native/SWIG-backed dependency) - deferred for the
    # same "don't tax every launch for a feature not every user touches"
    # reason, not TF's cost specifically.

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
WARNING = "#d9a441"

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
QGroupBox#subBox {{ margin-top: 12px; padding: 4px 6px 6px 6px; border-radius: 6px; font-weight: 500; }}
QFrame#collapsibleSection {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QToolButton#sectionHeader {{
    background: transparent; border: none; padding: 6px 6px; color: {TEXT};
    font-weight: 700; font-size: 12px; text-align: left;
}}
QToolButton#sectionHeader:hover {{ color: {ACCENT}; }}
QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QProgressBar {{ background: {SURFACE_SUNKEN}; border: 1px solid {BORDER}; border-radius: 4px; }}
QStatusBar QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QPushButton#statusCancelBtn {{ padding: 1px 8px; font-size: 11px; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
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

# The three WM/GM classification methods, selected in the Brain Tissue box's
# Method dropdown (no single method works best on every scan, so it's
# switched often) - each is a separate domain module with its own module
# docstring explaining its approach/trade-offs, and its own settings dialog
# (the box's gear button, also under Tools). Adaptive Thresholding is the
# default per explicit request (2026-10-01; Fuzzy C-Means was before); GMM+MRF (a fourth
# candidate briefly implemented for comparison) was removed entirely -
# see brain_tissue_fcm.py/brain_tissue_adaptive.py's own docstrings for
# the surviving two, and brain_tissue.py for the original threshold method.
# WM/GM methods: see `viewmodels.brain_tissue_viewmodel` (imported above).


_ROI2_REGION_LABELS = {
    VENTRICLE: "ventricle",
    WHITE_MATTER: "white matter",
    FALLBACK: "fallback - no clean homogeneous area nearby",
}


_STUDY_TABLE_COLUMNS = [
    "Slice", "Structure", "MTF area [1/mm]", "NPS ROI2 [HU²mm²]", "PS/MTF LF/HF", "Decision", "Verdict",
]
# Physical recipe: the last two columns show the study model's per-slice
# view instead of the legacy chest-CT SVM (docs/qa_method/head_ct_classifier.md).
_STUDY_TABLE_COLUMNS_PHYSICAL = [*_STUDY_TABLE_COLUMNS[:5], "Resembles (hint)", "Note"]

# ROI pair -> the segmentation structure its ROI1 sits on.
_PAIR_STRUCTURES = {FALX: DURAL_FOLDS, SEPTUM: SEPTUM_PELLUCIDUM}
# Tone (from the ViewModels) -> colour.
_TONE_COLORS = {TONE_GOOD: GOOD, TONE_BAD: CRITICAL, TONE_NEUTRAL: TEXT_MUTED}
# Same colors `RoiViewport._ROI_STYLES` draws each slot in.
# The labels the ROIs carry on the image (`RoiViewport._ROI_STYLES`).
_ROI_SHORT_LABELS = {"falx.roi1": "F1", "falx.roi2": "F2", "septum.roi1": "S1", "septum.roi2": "S2"}
_ROI_TABLE_COLORS = {
    "falx.roi1": "#49c8de",
    "falx.roi2": "#d98bd8",
    "septum.roi1": "#6fcf97",
    "septum.roi2": "#f2994a",
}


def _default_tissue_settings() -> tuple[BrainTissueSettings, FcmSettings, AdaptiveSettings]:
    """The HU threshold, FCM and adaptive methods' default settings."""
    threshold = BrainTissueSettings(
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
    fcm = FcmSettings(
        fuzziness=DEFAULT_FUZZINESS,
        max_iter=DEFAULT_MAX_ITER,
        spatial_weight=DEFAULT_SPATIAL_WEIGHT,
        min_component_size=FCM_DEFAULT_MIN_COMPONENT_SIZE,
    )
    adaptive = AdaptiveSettings(
        block_size=DEFAULT_BLOCK_SIZE,
        plausible_low=DEFAULT_PLAUSIBLE_RANGE[0],
        plausible_high=DEFAULT_PLAUSIBLE_RANGE[1],
        min_component_size=ADAPTIVE_DEFAULT_MIN_COMPONENT_SIZE,
        smoothing_mm=ADAPTIVE_DEFAULT_SMOOTHING_MM,
        use_study_range=ADAPTIVE_DEFAULT_USE_STUDY_RANGE,
    )
    return threshold, fcm, adaptive


class MainWindowView(QMainWindow):
    """Application shell: menu bar + viewer/results pane + sidebar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BICTool")
        self.resize(1180, 880)  # taller than before to fit the always-visible trace bar
        self.setStyleSheet(STYLESHEET)

        self._scheduler = Scheduler()
        # The application state and workflow (MVVM): one ViewModel per area,
        # owned by the MainViewModel; the window binds to them and renders.
        self.view_model = MainViewModel(
            self._scheduler.submit,
            load_config(),
            _SEGMENTATION_MAPPING,
            default_head_ct_model,
            _default_tissue_settings(),
            self,
        )
        self._study = self.view_model.study
        self._segmentation = self.view_model.segmentation
        self._stroke = self.view_model.stroke
        self._brain_tissue = self.view_model.brain_tissue
        self._roi = self.view_model.roi
        self._qa = self.view_model.qa
        self._results = self.view_model.results
        self._export = self.view_model.export
        # ROI pairs per slice (falx + septum, `viewmodels.roi_pairs`): automatic
        # ROIs, manual ones (saved drawings, moved ROIs - they win) and
        # deletions. `_roi_rects` is the displayed slice's effective ROIs.
        self._roi_rects: dict[str, RoiRect] = {}
        # Transient line shown in the image's bottom-right corner overlay -
        # live cursor/pixel-value while hovering, or the in-progress
        # rectangle's size while drawing (see `_on_cursor_moved`/
        # `_on_draft_changed`); empty when neither is happening.
        self._interaction_line: str = ""
        self._has_hand_drawn_draft = False  # see `_on_draft_changed`
        # The "Select Segments…" dialog, created on first use (the structure
        # choice and opacity live in the SegmentationViewModel).
        self._segment_selection_dialog: SegmentationSelectionDialog | None = None

        # Remembered view choices (trace log, ROI details, verdict details).
        self._ui_settings = UiSettings()
        self._build_status_bar_items()
        self._build_menu_bar()

        central = QWidget()
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Viewer | sidebar and content | trace log are split by draggable
        # handles (positions remembered), so the user decides how much room
        # the image and the controls get.
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(5)
        self._main_splitter.addWidget(self._build_viewer_pane())
        # The sidebar scrolls vertically when its boxes don't fit the window
        # (e.g. both ROI pairs + a multi-line status) instead of Qt squeezing
        # every box below its size - which clipped table rows.
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(self._build_sidebar())
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setMinimumWidth(320)  # sidebar minimum + room for the scrollbar
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; }")
        self._main_splitter.addWidget(sidebar_scroll)
        # Extra width (full screen, bigger window) goes 3:1 to viewer and sidebar.
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([1100, 400])

        self._trace_box = self._build_trace_bar()
        self._trace_splitter = QSplitter(Qt.Orientation.Vertical)
        self._trace_splitter.setChildrenCollapsible(False)
        self._trace_splitter.setHandleWidth(5)
        self._trace_splitter.addWidget(self._main_splitter)
        self._trace_splitter.addWidget(self._trace_box)
        self._trace_splitter.setStretchFactor(0, 1)
        self._trace_splitter.setStretchFactor(1, 0)
        outer_layout.addWidget(self._trace_splitter, 1)
        for splitter, key in ((self._main_splitter, MAIN_SPLITTER), (self._trace_splitter, TRACE_SPLITTER)):
            state = self._ui_settings.get_bytes(key)
            if state is not None:
                splitter.restoreState(state)
            splitter.splitterMoved.connect(lambda *_a, sp=splitter, k=key: self._ui_settings.set_bytes(k, sp.saveState()))

        self.setCentralWidget(central)
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(True)
        # Permanent: Qt's temporary status messages (status tips) never hide it.
        status_bar.addPermanentWidget(self._status_panel, 1)

        # The trace panel only receives log records while it is shown
        # (Settings > Show Trace Log); the console log is unaffected.
        self._log_handler = QtLogHandler()
        self._log_handler.message_logged.connect(self._append_trace_line)
        configure_logging()
        self._set_trace_visible(self._ui_settings.get_bool(SHOW_TRACE_LOG))
        self._bind_main_viewmodel()
        self._bind_study_viewmodel()
        self._bind_segmentation_viewmodel()
        self._brain_tissue.settings_changed.connect(self._on_brain_tissue_settings_changed)
        self._bind_roi_viewmodel()
        self._bind_qa_viewmodel()
        self._bind_stroke_viewmodel()
        self._apply_help_tooltips()
        logger.info("BICTool window ready.")

    # ---- ViewModel bindings ----

    def _bind_segmentation_viewmodel(self) -> None:
        seg = self._segmentation
        seg.status_changed.connect(self._segmentation_status_label.setText)
        seg.status_detail_changed.connect(lambda detail: self._set_tip(self._segmentation_status_label, detail))
        seg.busy_changed.connect(self._on_segmentation_busy_changed)
        seg.cancel_available_changed.connect(self._cancel_segment_volume_btn.setVisible)
        seg.message.connect(self._show_message)

    def _show_message(self, level: str, title: str, text: str) -> None:
        """A ViewModel's message as a dialog (ViewModels show no dialogs themselves)."""
        show = {"warning": QMessageBox.warning, "information": QMessageBox.information}.get(level, QMessageBox.critical)
        show(self, title, text)

    def _on_segmentation_busy_changed(self, busy: bool) -> None:
        self._segment_slice_action.setEnabled(not busy)
        self._segment_volume_action.setEnabled(not busy)

    def _on_slice_segmented(self, index: int) -> None:
        """A slice's segmentation landed (single slice, volume job or cache hit);
        the MainViewModel already dropped its stale WM/GM result."""
        if index == self._study.current_slice:
            self._apply_segmentation_masks(self._segmentation.cache[index])
            self._refresh_brain_tissue_for_current_slice()
        self._update_auto_rois_availability()

    def _bind_stroke_viewmodel(self) -> None:
        stroke = self._stroke
        stroke.status_changed.connect(self._stroke_status_label.setText)
        stroke.status_detail_changed.connect(lambda detail: self._set_tip(self._stroke_status_label, detail))
        stroke.busy_changed.connect(lambda busy: self._check_stroke_btn.setEnabled(not busy))
        stroke.slice_checked.connect(self._on_stroke_checked)
        stroke.message.connect(self._show_message)

    def _bind_qa_viewmodel(self) -> None:
        self._export.message.connect(self._show_message)
        qa = self._qa
        qa.progress_changed.connect(self._qa_progress_label.setText)
        qa.job_done.connect(lambda _job: self._refresh_metrics_box())
        qa.finished.connect(self._on_qa_finished)
        self._results.changed.connect(self._refresh_results_page)

    def _bind_roi_viewmodel(self) -> None:
        roi = self._roi
        roi.rois_changed.connect(self._refresh_rois_for_current_slice)
        roi.auto_status_changed.connect(lambda _text: self._refresh_auto_rois_tip())
        roi.tissue_progress_changed.connect(self._roi_status_label.setText)

    def _on_brain_tissue_settings_changed(self) -> None:
        """Method or settings changed (results were dropped, the MainViewModel
        re-places the white-matter ROI2s): redraw this slice's overlays."""
        self._refresh_brain_tissue_for_current_slice()

    def _bind_main_viewmodel(self) -> None:
        vm = self.view_model
        vm.series_opened.connect(self._on_series_opened)
        vm.study_closed.connect(self._on_study_closed)
        vm.slice_segmented.connect(self._on_slice_segmented)

    def _bind_study_viewmodel(self) -> None:
        study = self._study
        study.status_changed.connect(self._study_status_label.setText)
        study.loading_changed.connect(self._on_study_loading_changed)
        study.series_list_changed.connect(self._on_series_list_changed)
        study.load_failed.connect(lambda message: QMessageBox.critical(self, "Failed to open DICOM data", message))
        study.slice_changed.connect(self._render_slice)

    def _on_study_loading_changed(self, loading: bool) -> None:
        self._open_folder_action.setEnabled(not loading)
        self._open_file_action.setEnabled(not loading)

    def _on_series_list_changed(self, labels: list) -> None:
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        self._series_combo.addItems(labels)
        self._series_combo.blockSignals(False)
        # Only meaningful to interact with when there's an actual choice.
        self._series_combo.setEnabled(len(labels) > 1)

    # ---- status bar: what is running / just happened ----

    def _build_status_bar_items(self) -> None:
        """The bottom bar (`views.status_bar`): the study message, replaced by a
        full-width progress bar while loading, segmentation, ROI placement or
        the QA run is going on (Cancel where it can be stopped); each task's
        last state small on the right. The workflow code writes plain status
        text into the tasks' labels."""
        self._status_panel = StatusPanel()
        self._study_activity = self._status_panel.message
        self._study_status_label = self._study_activity.label
        self._roi_activity = self._status_panel.add_item("ROIs:", busy=r"^Placing")
        self._roi_status_label = self._roi_activity.label
        self._segmentation_activity = self._status_panel.add_item(
            "Segmentation:", busy=r"^(Segmenting|Converting|Cancelling)"
        )
        self._segmentation_status_label = self._segmentation_activity.label
        self._cancel_segment_volume_btn = self._segmentation_activity.add_cancel(self._on_cancel_segment_volume_clicked)
        self._qa_activity = self._status_panel.add_item("", busy=r"^QA \d+/\d+")
        self._qa_progress_label = self._qa_activity.label
        self._cancel_run_btn = self._qa_activity.add_cancel(self._on_cancel_run_clicked)

    # ---- hover help (presentation.viewmodels.help_text) ----

    def _help_target(self, name: str):
        attr, _, key = name.partition(":")
        target = getattr(self, attr, None)
        return target.get(key) if key and isinstance(target, dict) else target

    def _apply_help_tooltips(self) -> None:
        """Gives every control its hover help; a reason set earlier (e.g. why
        it is disabled) is kept below the help."""
        for name, text in TOOLTIPS.items():
            target = self._help_target(name)
            if target is None:
                continue
            reason = target.property("tipReason")
            if reason is None:
                reason = target.toolTip()
            target.setProperty("helpText", text)
            self._set_tip(target, reason or "")

    @staticmethod
    def _set_tip(target, reason: str = "") -> None:
        """Tooltip = the control's help text, then the current reason (why it is
        disabled, what it is waiting for) - so neither overwrites the other."""
        target.setProperty("tipReason", reason)
        help_text = target.property("helpText") or ""
        target.setToolTip("\n\n".join(text for text in (help_text, reason) if text))

    def _section(self, title: str, name: str) -> CollapsibleSection:
        """A collapsible sidebar step whose open/closed state is remembered."""
        section = CollapsibleSection(title)
        section.set_expanded(self._ui_settings.get_bool(section_key(name), default=True))
        section.toggled.connect(lambda expanded: self._ui_settings.set_bool(section_key(name), expanded))
        return section

    @staticmethod
    def _hint_label(text: str) -> QLabel:
        """A one-line instruction under a step's title."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        return label

    def _set_trace_visible(self, visible: bool) -> None:
        """Settings > Show Trace Log: shows the panel and feeds it log records,
        or hides it and detaches it from logging (remembered)."""
        self._trace_box.setVisible(visible)
        bictool_logger = logging.getLogger("bictool")
        if visible and self._log_handler not in bictool_logger.handlers:
            self._log_handler.setFormatter(bictool_logger.handlers[0].formatter if bictool_logger.handlers else None)
            bictool_logger.addHandler(self._log_handler)
        elif not visible and self._log_handler in bictool_logger.handlers:
            bictool_logger.removeHandler(self._log_handler)
        if self._show_trace_action.isChecked() != visible:
            self._show_trace_action.setChecked(visible)
        self._ui_settings.set_bool(SHOW_TRACE_LOG, visible)

    # ---- trace / log panel ----

    def _build_trace_bar(self) -> QWidget:
        box = QGroupBox("Trace / Log")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        self._trace_log = QPlainTextEdit()
        self._trace_log.setReadOnly(True)
        self._trace_log.setMaximumBlockCount(5000)  # cap memory growth on a long session
        self._trace_log.setMinimumHeight(60)
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
        self._set_tip(self._close_study_action, "Open a study first")
        self._reload_series_action = study_menu.addAction("Reload Series", self._on_reload_series_clicked)
        self._reload_series_action.setEnabled(False)
        self._set_tip(self._reload_series_action, "Open a study first")

        view_menu = menu_bar.addMenu("&View")
        self._view_tags_action = view_menu.addAction("DICOM Tags…", self._on_view_dicom_tags_clicked)
        self._view_tags_action.setEnabled(False)
        self._set_tip(self._view_tags_action, "Open a study first")

        # Configuration/adjustment dialogs - distinct from View's display
        # toggles and File's I/O. None of these actions need a study-loaded
        # gate except Custom Window, which is disabled/re-enabled the same
        # way the viewer toolbar's own preset dropdown already is (see
        # `_show_view_page`) - the settings dialogs and the method choice
        # below are plain numeric-parameters forms/preferences, usable
        # (and worth setting up) before any study is even opened.
        tools_menu = menu_bar.addMenu("&Tools")

        # Segmentation runs automatically for a new volume (see
        # the MainViewModel), so its manual triggers live here
        # instead of taking sidebar space; tooltips explain a disabled entry.
        segmentation_menu = tools_menu.addMenu("Segmentation")
        segmentation_menu.setToolTipsVisible(True)
        self._segment_slice_action = segmentation_menu.addAction("Segment Current Slice", self._on_segment_slice_clicked)
        self._segment_slice_action.setEnabled(False)
        self._set_tip(self._segment_slice_action, "Open a study first")
        self._segment_volume_action = segmentation_menu.addAction("Segment Volume", self._on_segment_volume_clicked)
        self._segment_volume_action.setEnabled(False)
        self._set_tip(self._segment_volume_action, "Open a study first")
        tools_menu.addSeparator()

        # The WM/GM method itself is chosen in the Brain Tissue box (no single
        # method works best on every scan, so it's switched often); each
        # method's tunable parameters stay reachable here as well.
        tools_menu.addAction("HU Thresholding Settings…", self._on_brain_tissue_settings_clicked)
        tools_menu.addAction("Fuzzy C-Means Settings…", self._on_fcm_settings_clicked)
        tools_menu.addAction("Adaptive Threshold Settings…", self._on_adaptive_settings_clicked)
        tools_menu.addSeparator()
        self._custom_window_action = tools_menu.addAction("Custom Window…", self._on_custom_window_clicked)

        # Deliberately tucked away: the legacy Octave recipe only exists to
        # reproduce Octave-era numbers (its frequency axes are mis-scaled -
        # docs/qa_method/physical_qa_recipe.md). While it is active, the Run
        # Metrics box says so.
        tools_menu.addSeparator()
        advanced_menu = tools_menu.addMenu("Advanced")
        recipe_menu = advanced_menu.addMenu("QA Recipe")
        recipe_menu.setToolTipsVisible(True)
        recipe_group = QActionGroup(self)
        recipe_group.setExclusive(True)
        self._qa_recipe_actions = {}
        for recipe, text, tooltip in (
            (RECIPE_PHYSICAL, "Physical (default)", "True frequency axes, fixed 0.1-0.8 1/mm band"),
            (RECIPE_OCTAVE, "Octave (legacy)", "Original Octave/thesis code - reproduces old results only"),
        ):
            action = recipe_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(recipe == self._qa.recipe)
            action.setToolTip(tooltip)
            action.triggered.connect(lambda checked, r=recipe: self._on_qa_recipe_selected(r))
            recipe_group.addAction(action)
            self._qa_recipe_actions[recipe] = action
        self._custom_window_action.setEnabled(False)
        self._set_tip(self._custom_window_action, "Open a study first")

        settings_menu = menu_bar.addMenu("S&ettings")
        settings_menu.setToolTipsVisible(True)
        self._show_trace_action = settings_menu.addAction("Show Trace Log")
        self._show_trace_action.setCheckable(True)
        self._show_trace_action.toggled.connect(self._set_trace_visible)

        export_menu = menu_bar.addMenu("&Export")
        self._export_action = export_menu.addAction("Export Results…", self._open_export_dialog)
        # Disabled rather than a silent no-op click - a clickable-looking
        # control that does nothing when clicked is an error-prevention/
        # visibility-of-system-status problem (NFR-07) - stays disabled
        # until Run All has actually produced a result to export.
        self._export_action.setEnabled(False)
        self._set_tip(self._export_action, "Run metrics first")

    def _on_close_study_clicked(self) -> None:
        self._study.close()

    def _on_study_closed(self) -> None:
        """The study was closed (every ViewModel reset): back to the no-study state."""
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        self._series_combo.setEnabled(False)
        self._series_combo.blockSignals(False)

        self._roi_viewport.clear()  # also clears every seg_*/stroke_lesion overlay - see RoiViewport.clear_all_overlays
        self._roi_rects = {}
        self._reset_roi_table()
        self._confirm_rois_cb.setChecked(False)
        self._confirm_rois_cb.setEnabled(False)
        self._has_hand_drawn_draft = False
        self._update_roi_edit_buttons()

        self._reset_segmentation_display()
        self._reset_stroke_display()
        self._reset_brain_tissue_display()

        self._slice_slider.setEnabled(False)
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(0)
        self._slice_slider.blockSignals(False)
        self._slice_label.setText("No slices loaded")

        self._reset_qa_results()

        self._close_study_action.setEnabled(False)
        self._set_tip(self._close_study_action, "Open a study first")
        self._reload_series_action.setEnabled(False)
        self._set_tip(self._reload_series_action, "Open a study first")
        self._view_tags_action.setEnabled(False)
        self._set_tip(self._view_tags_action, "Open a study first")
        self._custom_window_action.setEnabled(False)
        self._set_tip(self._custom_window_action, "Open a study first")

        self._show_view_page(0)

    def _on_reload_series_clicked(self) -> None:
        self._study.reload_series()

    def _on_view_dicom_tags_clicked(self) -> None:
        if self._study.volume is None:
            return
        path = self._study.volume.slice_paths[self._study.current_slice]
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            logger.exception("Failed to read DICOM tags from %s", path)
            QMessageBox.critical(self, "Failed to read DICOM tags", str(exc))
            return
        title = f"DICOM Tags — Slice {self._study.current_slice + 1} / {self._study.volume.num_slices}"
        DicomTagsDialog(dataset, title=title, parent=self).exec()

    def _on_brain_tissue_settings_clicked(self) -> None:
        dialog = BrainTissueSettingsDialog(self._brain_tissue.settings[_BRAIN_TISSUE_METHOD_THRESHOLD], self._brain_tissue.default_settings[_BRAIN_TISSUE_METHOD_THRESHOLD], parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._brain_tissue.set_settings(_BRAIN_TISSUE_METHOD_THRESHOLD, dialog.settings())

    def _open_export_dialog(self) -> None:
        if not self._export.can_export_slice():
            return
        # Opened from the Study summary tab -> whole study preselected.
        dialog = ExportDialog(
            study_available=bool(self._qa.results),
            default_scope=SCOPE_STUDY if self._showing_study_summary() else "slice",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.scope() == SCOPE_STUDY:
            self._perform_study_export(dialog.selected_items(), dialog.destination())
        else:
            self._perform_export(dialog.selected_items(), dialog.destination())

    def _perform_study_export(self, items: set[str], destination: Path) -> None:
        self._export.export_study(items, destination, self._series_combo.currentText())

    def _perform_export(self, items: set[str], destination: Path) -> None:
        self._export.export_slice(items, destination, self._save_chart_images)

    def _save_chart_images(self, destination: Path, stem: str) -> list[Path]:
        """The four result charts of the shown slice as JPG (needs the widgets)."""
        written = []
        for name, chart_view in (
            ("mtf", self._mtf_chart_view), ("nps", self._nps_chart_view),
            ("ps", self._ps_chart_view), ("ratio", self._ratio_chart_view),
        ):
            path = destination / f"{stem}_{name}.jpg"
            chart_view.grab().save(str(path), "JPG")
            written.append(path)
        return written

    def _on_open_dicom_folder_clicked(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open DICOM series folder")
        if path:
            self._load_volume_from(path)

    def _on_open_dicom_file_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM file", "", "DICOM files (*.dcm);;All files (*)")
        if path:
            self._load_volume_from(path)

    def _load_volume_from(self, path: str) -> None:
        self._study.open_path(path)

    def _on_series_discovered(self, candidates: list[SeriesCandidate]) -> None:
        self._study.on_series_discovered(candidates)

    def _on_series_combo_changed(self, index: int) -> None:
        if index >= 0:  # -1 while the combo is cleared/repopulated
            self._study.select_series(index)

    def _load_selected_series(self, index: int, *, force_reload: bool = False) -> None:
        self._study.select_series(index, force_reload=force_reload)

    def _on_volume_loaded(
        self, volume: DicomVolume, *, is_primary_series: bool = True, force_reload: bool = False
    ) -> None:
        """A volume loaded by other means (e.g. tests) becomes the shown series."""
        self._study.adopt_volume(volume, is_primary_series=is_primary_series, force_reload=force_reload)

    def _on_series_opened(self, volume: DicomVolume, _is_primary: bool) -> None:
        """A series was opened (every ViewModel reset for it): reset the widgets
        and show its first slice. Automatic segmentation of the primary series
        starts right after (MainViewModel)."""
        self._open_folder_action.setEnabled(True)
        self._open_file_action.setEnabled(True)
        self._close_study_action.setEnabled(True)
        self._set_tip(self._close_study_action, "")
        self._reload_series_action.setEnabled(True)
        self._set_tip(self._reload_series_action, "Re-reads the current series from disk")
        self._view_tags_action.setEnabled(True)
        self._set_tip(self._view_tags_action, "")
        self._custom_window_action.setEnabled(True)
        self._set_tip(self._custom_window_action, "")

        self._reset_segmentation_display()
        self._reset_stroke_display()
        self._reset_brain_tissue_display()
        self._confirm_rois_cb.setChecked(False)
        self._reset_qa_results()
        self._update_segmentation_availability()

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

    def _show_slice(self, index: int) -> None:
        self._study.set_slice(index)

    def _render_slice(self, index: int) -> None:
        """The StudyViewModel moved to slice `index`: show it and everything on it."""
        self._slice_label.setText(f"Slice {index + 1} / {self._study.volume.num_slices}")
        self._roi_viewport.set_image(self._study.volume.get_slice_hu(index))  # also clears every overlay
        self._refresh_rois_for_current_slice()
        self._refresh_segmentation_for_current_slice()
        self._refresh_stroke_for_current_slice()
        self._refresh_brain_tissue_for_current_slice()
        self._show_auto_rois_for_current_slice()
        self._refresh_metrics_box()
        if index in self._results.slices:
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
        if self._study.volume is None:
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
        self._set_tip(self._results_tab_btn, "Run metrics to see results")
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
        self._series_box.setVisible(is_viewer_page)
        self._run_all_btn.setVisible(is_viewer_page)
        self._metrics_hint.setVisible(is_viewer_page)
        self._update_metrics_box_title()
        self._refresh_metrics_box()

    def _update_metrics_box_title(self) -> None:
        """Step 3 on the viewer; on the Results page the box shows the shown
        slice's measurements, or the study averages on the summary tab."""
        if self._view_stack.currentIndex() == 0:
            title = "3. Run QA metrics"
        elif self._results_tabs.currentIndex() == 1:
            title = f"Study averages ({PAIR_LABELS[self._results.pair]})"
        else:
            title = "Slice results"
        self._metrics_box.setTitle(title)

    def _build_viewer_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self._roi_viewport = RoiViewport()
        self._roi_viewport.setMinimumHeight(420)
        self._roi_viewport.draft_changed.connect(self._on_draft_changed)
        self._roi_viewport.draft_moved.connect(self._on_draft_moved)
        self._roi_viewport.roi_selected.connect(self._on_roi_selected)
        self._roi_viewport.roi_delete_requested.connect(self._on_roi_delete_requested)
        self._roi_viewport.context_menu_requested.connect(self._on_roi_context_menu)
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

        self._study_status_label.setText("No study loaded — File → Open DICOM Folder/File…")

        return page

    def _on_draft_changed(self, rect: RoiRect | None) -> None:
        """A rectangle was just drawn (or cleared, e.g. by a slice change) -
        drawing doesn't directly produce an ROI (see `RoiViewport`'s
        draw-then-assign model); it makes the "Save drawn rectangle as"
        buttons available. Also drives the live "Selecting: WxH px" line in
        the image's bottom-right corner overlay while dragging.
        """
        self._has_hand_drawn_draft = rect is not None
        self._update_roi_edit_buttons()
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

    def _save_draft_as(self, key: str) -> None:
        """Saves the hand-drawn rectangle into slot `key` (e.g. "falx.roi1") on
        this slice - a manual ROI that wins over the automatic one (right-click
        > Save as). Saving a ROI1 re-places that pair's automatic ROI2 around
        it (unless that ROI2 is manual too).
        """
        rect = self._roi_viewport.commit_current_as(key)
        if rect is None:
            return  # the menu only offers this on a drawn rectangle - defensive only
        self._roi_viewport.clear_draft()
        index = self._study.current_slice
        logger.info(
            "%s set on slice %d: x=%d-%d, y=%d-%d (%dx%d)",
            key_label(key), index + 1, rect.x0, rect.x1, rect.y0, rect.y1, rect.width, rect.height,
        )
        # Redrawing/re-saving an ROI after confirmation forces a fresh review.
        self._confirm_rois_cb.setChecked(False)
        self._roi.save_manual(index, key, rect)
        self._update_confirm_availability()

    def _reassign_roi(self, source: str, target: str) -> None:
        """An existing ROI saved as another role on this slice (right-click > Save as)."""
        self._roi_viewport.select_roi(None)
        self._roi.reassign(self._study.current_slice, source, target)
        self._update_confirm_availability()

    def _roi_context_menu(self, key: str) -> QMenu:
        """Right-click menu of an ROI (`key`) or of the drawn rectangle (""):
        Delete, and Save as one of the four roles."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        title = menu.addAction(
            "Drawn rectangle" if not key else f"{_ROI_SHORT_LABELS[key]} · {ROLE_LABELS[key]}"
        )
        title.setEnabled(False)
        menu.addSeparator()
        locked = self._confirm_rois_cb.isChecked()
        delete = menu.addAction("Delete")
        delete.triggered.connect(lambda _checked=False: self._on_roi_delete_requested(key))
        save_menu = menu.addMenu("Save as")
        save_menu.setToolTipsVisible(True)
        for target in ROI_KEYS:
            action = save_menu.addAction(ROLE_LABELS[target])
            action.setEnabled(target != key and not locked)
            action.triggered.connect(lambda _checked=False, t=target: self._save_roi_as(key, t))
        if locked:
            delete.setEnabled(False)
            save_menu.setEnabled(False)
            menu.addSeparator()
            note = menu.addAction("ROIs are confirmed - untick 'ROIs checked' to edit")
            note.setEnabled(False)
        return menu

    def _on_roi_context_menu(self, key: str, global_pos) -> None:
        self._roi_context_menu(key).exec(global_pos)

    def _save_roi_as(self, source: str, target: str) -> None:
        if source:
            self._reassign_roi(source, target)
        else:
            self._save_draft_as(target)

    def _update_confirm_availability(self) -> None:
        # One confirmation covers every slice: available as soon as any slice
        # has a complete ROI pair (and no QA batch is running).
        ready = self._study.volume is not None and bool(self._qa_ready_jobs()) and self._qa.jobs is None
        self._confirm_rois_cb.setEnabled(ready or self._confirm_rois_cb.isChecked())

    def _effective_rois(self, slice_index: int) -> dict[str, RoiRect]:
        return self._roi.effective(slice_index)

    def _qa_ready_jobs(self) -> list[tuple[int, str]]:
        return self._roi.ready_jobs()

    def _refresh_rois_for_current_slice(self) -> None:
        """Shows the displayed slice's ROIs (both pairs) on the image and in the table."""
        rois = self._effective_rois(self._study.current_slice) if self._study.volume is not None else {}
        self._roi_rects = rois
        selected = self._roi_viewport.selected_roi()
        self._roi_viewport.clear_rois()
        for key in ROI_KEYS:
            if key in rois:
                self._roi_viewport.set_roi(key, rois[key])
                self._update_roi_table_row(key, rois[key])
            else:
                self._reset_roi_table_row(key)
        if selected is not None and selected not in rois:
            self._roi_viewport.select_roi(None)
        self._update_confirm_availability()
        self._update_roi_edit_buttons()

    # ---- auto ROIs (domain.roi_selection: structured_roi -> ROI1, homogeneous_roi -> ROI2), per pair ----

    def _auto_pairs(self) -> list[str]:
        """Auto ROIs always place both pairs (falx and septum)."""
        return list(PAIRS)

    def _update_auto_rois_availability(self) -> None:
        confirmed = self._confirm_rois_cb.isChecked()
        available = self._study.volume is not None and bool(self._segmentation.cache) and not confirmed
        self._auto_rois_btn.setEnabled(available)
        self._refresh_auto_rois_tip()

    def _refresh_auto_rois_tip(self) -> None:
        """The auto-ROI button's tooltip: why it is disabled, else where the
        automatic ROIs are (this slice, all slices, skipped slices)."""
        if self._confirm_rois_cb.isChecked():
            reason = "ROIs are confirmed - untick 'ROIs checked' to change them"
        elif not self._auto_rois_btn.isEnabled():
            reason = "Needs anatomy segmentation first"
        else:
            reason = self._roi.status_text if self._auto_rois_btn.isChecked() else ""
        self._set_tip(self._auto_rois_btn, reason)

    def _auto_roi1_slices(self, pair: str) -> list[int]:
        return self._roi.auto_roi1_slices(pair)

    def _show_auto_rois_for_current_slice(self) -> None:
        if self._study.volume is None:
            return
        self._refresh_rois_for_current_slice()
        self._roi.update_status()

    def _update_auto_rois_status(self) -> None:
        self._roi.update_status()

    def _on_auto_rois_toggled(self, checked: bool) -> None:
        self._auto_rois_btn.setText("Remove auto ROIs" if checked else "Start auto ROIs")
        self._roi.set_auto(checked)
        self._refresh_auto_rois_tip()

    def _on_draft_moved(self, key: str, rect: RoiRect) -> None:
        """The user dragged an ROI (only possible while not confirmed): it
        becomes that slot's manual ROI on this slice - the automatic one is
        kept for "Restore automatic". A hand-drawn draft ("") isn't an ROI yet."""
        if key:
            self._roi.save_manual(self._study.current_slice, key, rect)

    def _on_roi_selected(self, key: str | None) -> None:
        """An ROI was clicked on the image (or the selection cleared): mirror it in the table."""
        self._roi_table.blockSignals(True)
        if key is None:
            self._roi_table.clearSelection()
        else:
            self._roi_table.selectRow(ROI_KEYS.index(key))
        self._roi_table.blockSignals(False)
        self._update_roi_edit_buttons()

    def _on_roi_table_clicked(self, row: int, _column: int) -> None:
        key = ROI_KEYS[row]
        self._roi_viewport.select_roi(key if key in self._roi_rects else None)
        if key not in self._roi_rects:
            self._roi_table.clearSelection()

    def _on_roi_delete_requested(self, key: str) -> None:
        """Delete key / "Delete ROI" / right-click > Delete: the slot shows
        nothing on this slice any more (manual and automatic) until a new ROI
        is saved into it or the slice is restored. Key "" = the hand-drawn
        rectangle, which is simply discarded."""
        if self._confirm_rois_cb.isChecked():
            return  # confirmed ROIs are locked
        if not key:
            logger.info("Drawn rectangle discarded")
            self._roi_viewport.clear_draft()
            self._update_roi_edit_buttons()
            return
        self._roi_viewport.select_roi(None)
        self._roi.delete(self._study.current_slice, key)

    def _on_delete_roi_clicked(self) -> None:
        key = self._roi_viewport.selected_roi()
        if key is not None:
            self._on_roi_delete_requested(key)
        elif self._roi_viewport.has_draft():
            self._on_roi_delete_requested("")

    def _on_restore_auto_clicked(self) -> None:
        """Back to the automatic ROIs on this slice: manual ROIs and deletions dropped."""
        self._roi.restore_slice(self._study.current_slice)

    def _reset_auto_rois_button(self) -> None:
        self._auto_rois_btn.blockSignals(True)
        self._auto_rois_btn.setChecked(False)
        self._auto_rois_btn.setText("Start auto ROIs")
        self._auto_rois_btn.blockSignals(False)
        self._update_auto_rois_availability()

    def _update_roi_edit_buttons(self) -> None:
        """"Delete ROI" while an ROI is selected or a drawn rectangle is shown,
        "Restore automatic" while this slice has manual ROIs or deletions -
        neither once the ROIs are confirmed.
        """
        confirmed = self._confirm_rois_cb.isChecked()
        locked_tip = "ROIs are confirmed - untick 'ROIs checked' to change them"
        selected = self._roi_viewport.selected_roi()
        draft = self._roi_viewport.has_draft()
        self._delete_roi_btn.setEnabled((selected is not None or draft) and not confirmed)
        if confirmed:
            self._set_tip(self._delete_roi_btn, locked_tip)
        elif selected is not None:
            self._set_tip(self._delete_roi_btn, f"Delete {ROLE_LABELS[selected]} on this slice (or press Del)")
        elif draft:
            self._set_tip(self._delete_roi_btn, "Delete the drawn rectangle (or press Del)")
        else:
            self._set_tip(self._delete_roi_btn, "Select an ROI on the image or in the table first")
        has_edits = self._study.volume is not None and self._roi.has_edits(self._study.current_slice)
        self._restore_auto_btn.setEnabled(has_edits and not confirmed)
        if confirmed:
            self._set_tip(self._restore_auto_btn, locked_tip)
        else:
            self._set_tip(self._restore_auto_btn, 
                "Drop this slice's manual ROIs and deletions - back to the automatic ones"
                if has_edits
                else "Nothing changed by hand on this slice"
            )

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
        # Falx / Septum: which ROI pair's result the curves (and the Run
        # Metrics box) show; a pair without a result on the slice is greyed out.
        self._results_pair_buttons: dict[str, QPushButton] = {}
        pair_group = QButtonGroup(page)
        pair_group.setExclusive(True)
        for pair in PAIRS:
            button = QPushButton(PAIR_LABELS[pair])
            button.setCheckable(True)
            button.setChecked(pair == FALX)
            button.clicked.connect(lambda _checked=False, p=pair: self._on_results_pair_clicked(p))
            pair_group.addButton(button)
            nav_row.addWidget(button)
            self._results_pair_buttons[pair] = button
        layout.addLayout(nav_row)

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
        curves_page = QWidget()
        curves_page.setLayout(charts_grid)
        # "Intermediate steps" swaps the four curves for how they were measured
        # (same space - the page can't scroll: the wheel steps through slices).
        self._slice_steps_panel = SliceStepsPanel()
        for widget in self._slice_steps_panel.findChildren(QWidget):
            widget.installEventFilter(self)
        self._results_curves_stack = QStackedWidget()
        self._results_curves_stack.addWidget(curves_page)
        self._results_curves_stack.addWidget(self._slice_steps_panel)
        # Which of the two is shown: an explicit two-way switch (the active
        # one highlighted); always starts on the curves.
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Show:"))
        view_group = QButtonGroup(page)
        view_group.setExclusive(True)
        self._curves_view_btn = QPushButton("Curves")
        self._steps_view_btn = QPushButton("Intermediate steps (advanced)")
        for button, steps in ((self._curves_view_btn, False), (self._steps_view_btn, True)):
            button.setObjectName("navTabBtn")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, s=steps: self._show_intermediate_steps(s))
            view_group.addButton(button)
            view_row.addWidget(button)
        self._curves_view_btn.setChecked(True)
        view_row.addStretch(1)
        layout.addLayout(view_row)
        # Stretch factor 1 so the charts actually grow to fill available
        # vertical space - previously a trailing addStretch(1) after the
        # feature box soaked up all of it instead, leaving the charts
        # pinned to their minimum size no matter how tall the window was.
        layout.addWidget(self._results_curves_stack, 1)

        feature_box = QGroupBox("Feature vector (per slice)")
        feature_form = QFormLayout(feature_box)
        self._feature_labels: dict[str, QLabel] = {}
        for name in _FEATURE_NAMES:
            value_label = QLabel("—")
            value_label.setProperty("role", "mono")
            feature_form.addRow(name, value_label)
            self._feature_labels[name] = value_label
        feature_box.setVisible(False)  # shown with the intermediate steps
        self._feature_box = feature_box
        self._slice_steps_panel.set_feature_widget(feature_box)  # Intermediate steps > PS/MTF & features
        self._results_tabs.addTab(per_slice, "Per slice")

        summary = QWidget()
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(0, 8, 0, 0)
        self._study_verdict_label = QLabel("—")
        self._study_verdict_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px; font-weight: 700;")
        self._study_verdict_label.setWordWrap(True)
        summary_layout.addWidget(self._study_verdict_label)
        # What a user needs (reasons in plain words, artefacts, unusual
        # slices); the numbers behind it are under "Details".
        self._study_verdict_basics_label = QLabel("")
        self._study_verdict_basics_label.setWordWrap(True)
        self._study_verdict_basics_label.setStyleSheet(f"color: {TEXT};")
        self._study_verdict_basics_label.setVisible(False)
        summary_layout.addWidget(self._study_verdict_basics_label)
        self._verdict_details_btn = QPushButton("▸ Details")
        self._verdict_details_btn.setCheckable(True)
        self._verdict_details_btn.setFlat(True)
        self._verdict_details_btn.setStyleSheet("text-align: left;")
        self._verdict_details_btn.setVisible(False)
        self._verdict_details_btn.toggled.connect(self._on_verdict_details_toggled)
        summary_layout.addWidget(self._verdict_details_btn)
        self._study_verdict_details_label = QLabel("")
        self._study_verdict_details_label.setWordWrap(True)
        self._study_verdict_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._study_verdict_details_label.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        self._study_verdict_details_label.setVisible(False)
        summary_layout.addWidget(self._study_verdict_details_label)
        self._verdict_details_btn.setChecked(self._ui_settings.get_bool(VERDICT_DETAILS_OPEN))
        self._study_provisional_label = QLabel(PROVISIONAL_SHORT)
        self._study_provisional_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self._study_provisional_label.setVisible(False)
        summary_layout.addWidget(self._study_provisional_label)
        self._study_table = QTableWidget(0, len(_STUDY_TABLE_COLUMNS))
        self._study_table.setHorizontalHeaderLabels(_STUDY_TABLE_COLUMNS)
        self._study_table.verticalHeader().setVisible(False)
        self._study_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._study_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._study_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._study_table.cellClicked.connect(self._on_study_table_clicked)
        summary_layout.addWidget(self._study_table, 1)
        self._results_tabs.addTab(summary, "Study summary")
        self._results_tabs.setTabToolTip(0, TAB_TOOLTIPS["per_slice"])
        self._results_tabs.setTabToolTip(1, TAB_TOOLTIPS["summary"])
        # Connected only now - adding the first tab emits currentChanged too.
        self._results_tabs.currentChanged.connect(self._on_results_tab_changed)
        return page

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override signature
        # Scroll through per-slice results with the mouse wheel, like the image viewer.
        if (
            event.type() == QEvent.Type.Wheel
            and self._results.slices
            and (
                watched is self._results_per_slice_page
                or watched in self._results_chart_views()
                or self._slice_steps_panel.isAncestorOf(watched)
            )
        ):
            self._step_results_slice(-1 if event.angleDelta().y() > 0 else 1)
            return True
        return super().eventFilter(watched, event)

    def _results_chart_views(self) -> tuple:
        return (self._mtf_chart_view, self._nps_chart_view, self._ps_chart_view, self._ratio_chart_view)

    def _step_results_slice(self, step: int) -> None:
        if self._results.slices:
            self._results_slider.setValue(self._results_slider.value() + step)

    def _on_results_slider_changed(self, position: int) -> None:
        if not 0 <= position < len(self._results.slices):
            return
        index = self._results.slices[position]
        self._show_result_for_slice(index)
        # Keep the image viewer on the same slice, so switching back shows its ROIs.
        if self._study.volume is not None and self._slice_slider.value() != index:
            self._slice_slider.setValue(index)

    def _on_study_table_clicked(self, row: int, _column: int) -> None:
        if not 0 <= row < len(self._results.table_jobs):
            return
        index, pair = self._results.table_jobs[row]
        self._results.select_pair(pair)
        self._results_tabs.setCurrentIndex(0)
        position = self._results.slices.index(index)
        if self._results_slider.value() != position:
            self._results_slider.setValue(position)
        else:
            self._show_result_for_slice(index)

    def _on_results_pair_clicked(self, pair: str) -> None:
        """The Falx/Septum switch above the per-slice curves (also drives the Run Metrics box)."""
        self._results.select_pair(pair)
        if self._results.slices:
            self._show_result_for_slice(self._results.slices[self._results_slider.value()])
        self._refresh_metrics_box()

    def _results_by_pair(self, pair: str) -> dict[int, SliceQualityResult]:
        return self._qa.results_by_pair(pair)

    def _refresh_results_page(self) -> None:
        """Rebuilds the slice slider and the study summary from the
        ResultsViewModel (failed jobs are only counted in the verdict)."""
        slices = self._results.slices
        self._results_slider.blockSignals(True)
        self._results_slider.setRange(0, max(len(slices) - 1, 0))
        self._results_slider.blockSignals(False)
        self._results_slider.setEnabled(len(slices) > 1)

        physical = self._results.is_physical()
        self._study_table.setHorizontalHeaderLabels(self._results.table_columns())
        rows = self._results.table_rows()
        self._study_table.setRowCount(len(rows))
        for row, table_row in enumerate(rows):
            cells = table_row.cells
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == len(cells) - 1:
                    item.setForeground(QColor(_TONE_COLORS[table_row.last_tone]))
                    if text:
                        item.setToolTip(text)  # full note when the column is too narrow
                elif physical and column == len(cells) - 2:
                    item.setToolTip(table_row.hint_tooltip)
                self._study_table.setItem(row, column, item)

        artefact_lines = self._results.artefact_lines() + self._results.basal_ganglia_lines()
        self._study_verdict_basics_label.setVisible(physical or bool(artefact_lines))
        self._verdict_details_btn.setVisible(physical)
        self._study_verdict_details_label.setVisible(physical and self._verdict_details_btn.isChecked())
        self._study_provisional_label.setVisible(physical)
        if physical:
            self._show_head_ct_verdict()
            return

        # Legacy recipe: one majority-vote line per pair (chest-CT SVM), and
        # the possible-artefact hints.
        self._study_verdict_basics_label.setText("\n".join(artefact_lines))
        lines, tones = self._results.legacy_verdict()
        if not lines:
            self._study_verdict_label.setText("—")
            self._study_verdict_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px; font-weight: 700;")
            return
        color = _TONE_COLORS[tones[0]] if len(set(tones)) == 1 else TEXT
        self._study_verdict_label.setText("\n".join(lines))
        self._study_verdict_label.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: 700;")

    def _results_are_physical(self) -> bool:
        return self._results.is_physical()

    def _show_head_ct_verdict(self) -> None:
        """Study summary of the physical recipe: the study-level head-CT verdict
        (basics, and the details behind "Details")."""
        text, details = self._results.head_ct_text(model_available=default_head_ct_model() is not None)
        color = {TONE_GOOD: GOOD, TONE_BAD: CRITICAL}.get(text.tone, WARNING if self._results.verdict else TEXT_MUTED)
        self._study_verdict_label.setText(text.headline)
        self._study_verdict_label.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: 700;")
        self._study_verdict_basics_label.setText("\n".join(text.basics))
        self._study_verdict_details_label.setText(details)
        self._set_tip(self._study_provisional_label, PROVISIONAL_NOTE)

    def _on_verdict_details_toggled(self, open_: bool) -> None:
        self._study_verdict_details_label.setVisible(open_ and not self._verdict_details_btn.isHidden())
        self._verdict_details_btn.setText(("▾" if open_ else "▸") + " Details")
        self._ui_settings.set_bool(VERDICT_DETAILS_OPEN, open_)

    def _shown_pair(self, index: int) -> str | None:
        return self._results.shown_pair(index)

    def _show_result_for_slice(self, index: int) -> None:
        """The Per-slice tab's content for `index` (a slice with at least one result)."""
        view = self._results.slice_view(index)
        if view is None:
            return
        pair, position = view.pair, view.position
        self._results_slider.blockSignals(True)
        self._results_slider.setValue(position)
        self._results_slider.blockSignals(False)
        self._results_prev_btn.setEnabled(position > 0)
        self._results_next_btn.setEnabled(position < view.count - 1)
        for p, button in self._results_pair_buttons.items():
            button.setEnabled(p in view.available_pairs)
            button.setChecked(p == pair)
        self._results_slice_label.setText(
            f"Slice {index + 1} / {self._study.volume.num_slices} — {PAIR_LABELS[pair]}  "
            f"({position + 1} of {view.count})"
        )
        self._result_verdict_label.setText(view.verdict_text)
        self._set_tip(self._result_verdict_label, view.verdict_tooltip)
        self._result_verdict_label.setStyleSheet(
            f"color: {_TONE_COLORS[view.verdict_tone]}; font-size: 15px; font-weight: 700;"
        )
        self._result_decision_label.setText(view.note_text)

        result = self._qa.results[(index, pair)]

        for name, value in zip(_FEATURE_NAMES, result.feature_vector):
            self._feature_labels[name].setText(f"{value:.4g}")

        # Plot the mean across edges, truncated to the first half (see
        # `mtf_display_curve`'s docstring for why) - not one series per
        # edge, which for a busy ROI producing dozens of detected edges made
        # the chart an unreadable tangle.
        mtf_x, mtf_y = mtf_display_curve(result.mtf_xaxis, result.mtf)
        self._update_line_chart(self._mtf_chart_view, mtf_x, [mtf_y] if mtf_y.size else [])
        # NPS = ROI2 (homogeneous, true noise spectrum); PS = ROI1
        # (structured ROI, WP2's "PS") - the same estimator on different ROIs.
        self._update_line_chart(self._nps_chart_view, result.nps_roi2.f_avg, [result.nps_roi2.nps_measured_radagv])
        self._update_line_chart(self._ps_chart_view, result.nps_roi1.f_avg, [result.nps_roi1.nps_measured_radagv])
        self._update_line_chart(self._ratio_chart_view, result.ratio.f_int, [result.ratio.ratio_nps_mtf])
        if self._steps_view_btn.isChecked():
            self._update_slice_steps(index)

        # Export covers the slice and pair shown here.
        self._export.show_slice(index, pair, result, self._series_combo.currentText())

    def _show_intermediate_steps(self, steps: bool) -> None:
        """Per slice: the slice's intermediate steps, or the four curves."""
        (self._steps_view_btn if steps else self._curves_view_btn).setChecked(True)
        self._feature_box.setVisible(steps)
        self._results_curves_stack.setCurrentIndex(1 if steps else 0)
        if steps and self._results.slices:
            self._update_slice_steps(self._results.slices[self._results_slider.value()])

    def _update_slice_steps(self, index: int) -> None:
        self._slice_steps_panel.show_steps(self._results.slice_steps(index), "No intermediate steps for this slice.")

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
        # Minimum only: the sidebar grows with the window and with the
        # viewer | sidebar handle (no maximum), and its boxes grow with it.
        sidebar.setMinimumWidth(310)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._series_box = self._build_study_info_box()
        layout.addWidget(self._series_box)

        # Wrapped in one widget (not just a bare layout) so `_show_view_page`
        # can hide Segmentation + ROI Selection together on the Results page
        # - Run Metrics below stays visible on both pages. Stacked full-width
        # (not side by side) - ROI Selection's table needs its own row's
        # worth of width; splitting the sidebar in half is what caused the
        # coordinate-text overflow bug in the first place (see project memory).
        self._segmentation_roi_widget = QWidget()
        column = QVBoxLayout(self._segmentation_roi_widget)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        # Step 1: anatomy (CNN) and WM/GM separation (HU-based, no ML) as two
        # sub-boxes of one step - different methods, one purpose: finding where
        # the ROIs go.
        self._step1_section = self._section("1. Segmentation", "segmentation")
        self._step1_section.content_layout.addWidget(self._hint_label(STEP_HINTS["segmentation"]))
        self._step1_section.content_layout.addWidget(self._build_segmentation_box())
        self._step1_section.content_layout.addWidget(self._build_brain_tissue_box())
        column.addWidget(self._step1_section)
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
        box = self._section("Series", "series")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        box.content_layout.addLayout(form)

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
        volume = self._study.volume
        if volume is None:
            for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
                self._roi_viewport.set_overlay_text(corner, [])
            return

        series_label = self._series_combo.currentText() or "Series"
        slice_index = self._study.current_slice

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
                f"Acquired: {format_dicom_date(volume.acquisition_date)} {volume.acquisition_time or ''}".strip(),
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
        # see `SegmentationViewModel.selected_names`'s own comment), and a vertical
        # slider for their shared opacity; the choice itself lives here
        # (`SegmentationViewModel.selected_names`), not in that dialog, and persists
        # across slice navigation for the whole loaded volume (also per
        # explicit request - see `_refresh_segmentation_for_current_slice`).
        # The sidebar's own "Show Segments" checkbox (per a later request)
        # is a quick on/off switch for whatever's currently selected,
        # without needing to open that dialog just to hide/show it.
        box = QGroupBox("Anatomical")
        box.setObjectName("subBox")
        self._anatomical_box = box
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # Select Segments / Show Segments. Progress (and Cancel) of the
        # segmentation runs is in the status bar; the manual Segment
        # Slice/Volume triggers are under Tools > Segmentation.
        segments_row = QHBoxLayout()
        self._select_segments_btn = QPushButton("Select Segments…")
        self._select_segments_btn.setEnabled(False)
        self._set_tip(self._select_segments_btn, "Open a study first")
        self._select_segments_btn.clicked.connect(self._on_select_segments_clicked)
        segments_row.addWidget(self._select_segments_btn)
        # A quick on/off switch for the *currently selected* structures'
        # overlays (per explicit request) - separate from the selection
        # itself (chosen in the "Select Segments…" dialog, all 16 selected
        # by default): toggling this never changes
        # `SegmentationViewModel.selected_names`, just whether those
        # already-chosen structures are actually drawn right now, so
        # flipping it back on shows exactly what was showing before.
        self._show_segments_cb = QCheckBox("Show Segments")
        self._show_segments_cb.setChecked(False)  # per explicit request (2026-09-24) - opt in, not on by default
        self._show_segments_cb.setEnabled(False)
        self._set_tip(self._show_segments_cb, "Open a study first")
        self._show_segments_cb.toggled.connect(self._on_show_segments_toggled)
        segments_row.addWidget(self._show_segments_cb)
        segments_row.addStretch(1)
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
        self._segment_selection_dialog.set_enabled_all(bool(self._segmentation.cache))
        self._segment_selection_dialog.show()
        self._segment_selection_dialog.raise_()
        self._segment_selection_dialog.activateWindow()

    def _on_segmentation_selection_changed(self, display_name: str, checked: bool) -> None:
        if checked:
            self._segmentation.selected_names.add(display_name)
            self._apply_segment_overlay(display_name)
        else:
            self._segmentation.selected_names.discard(display_name)
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
        masks = self._segmentation.current_masks
        if masks is None or display_name not in masks:
            return
        self._roi_viewport.set_overlay(
            f"seg_{display_name}",
            masks[display_name],
            color=_SEGMENTATION_COLORS[display_name],
            opacity=self._segmentation.opacity,
            label=display_name,
        )

    def _on_show_segments_toggled(self, checked: bool) -> None:
        """The main sidebar's quick on/off switch - shows or hides every
        currently *selected* structure's overlay without touching the
        selection itself (`SegmentationViewModel.selected_names`), so turning it
        back on reproduces exactly what was showing before.
        """
        for display_name in self._segmentation.selected_names:
            if checked:
                self._apply_segment_overlay(display_name)
            else:
                self._roi_viewport.clear_overlay(f"seg_{display_name}")

    def _on_segmentation_opacity_changed(self, opacity: float) -> None:
        self._segmentation.opacity = opacity
        for display_name in self._segmentation.selected_names:
            self._apply_segment_overlay(display_name)

    # ---- segmentation commands (SegmentationViewModel) ----

    def _ensure_nifti_volume_then(self, on_ready) -> None:
        self._segmentation.ensure_nifti_then(on_ready)

    def _on_segment_slice_clicked(self) -> None:
        self._segmentation.segment_slice(self._study.current_slice)

    def _on_segment_volume_clicked(self) -> None:
        self._segmentation.segment_volume()

    def _on_cancel_segment_volume_clicked(self) -> None:
        self._segmentation.cancel_volume()

    def _on_volume_segmentation_cancelled(self, completed: int, total: int) -> None:
        """What the volume job reports when cancelled (also used by tests)."""
        self._segmentation.on_volume_cancelled(completed, total)

    def _apply_segmentation_masks(self, result) -> None:
        """Decodes `result`'s raw model-class masks into checkbox display
        names and (re-)applies overlays for whichever ones are currently
        selected (`SegmentationViewModel.selected_names` - a persisted, whole-volume
        choice, not tied to this particular slice) - shared by the
        single-slice path (`_apply_segmentation_result`, which also updates
        the status label/buttons) and the mid-volume-batch live-update path
        (`_on_volume_segmentation_slice_finished`), which must leave the
        in-progress batch's own status text/button state alone. Also
        enables the selection dialog's checkboxes, if it's open - it starts
        disabled until at least one slice has real data to show.
        """
        decoded = decode_masks_for_checkboxes(result, _SEGMENTATION_MAPPING)
        self._segmentation.current_masks = decoded
        if self._segment_selection_dialog is not None:
            self._segment_selection_dialog.set_enabled_all(True)
        for display_name in self._segmentation.selected_names:
            self._apply_segment_overlay(display_name)

    # ---- anatomy segmentation: whole-volume batch (background, cancellable) ----

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
        box = QGroupBox("WM/GM separation")
        box.setObjectName("subBox")
        self._wmgm_box = box
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # Method row: which WM/GM classifier runs (no single method works
        # best on every scan) + its settings dialog. A plain preference - no
        # study needed to set it up.
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self._brain_tissue_method_combo = QComboBox()
        self._brain_tissue_method_combo.addItems(_BRAIN_TISSUE_METHODS)
        self._brain_tissue_method_combo.setCurrentText(self._brain_tissue.method)
        self._brain_tissue_method_combo.currentTextChanged.connect(self._on_brain_tissue_method_selected)
        method_row.addWidget(self._brain_tissue_method_combo, 1)
        self._brain_tissue_settings_btn = QPushButton("⚙")
        self._set_tip(self._brain_tissue_settings_btn, "Settings of the selected method")
        self._brain_tissue_settings_btn.setFixedWidth(32)
        self._brain_tissue_settings_btn.clicked.connect(self._on_brain_tissue_method_settings_clicked)
        method_row.addWidget(self._brain_tissue_settings_btn)
        self._update_brain_tissue_settings_button()
        layout.addLayout(method_row)

        # Overlays in a 2x2 grid - half the height of a single column. The
        # two tissue classes share the left column, CSF and the skull-strip
        # mask the right one.
        self._csf_cb = QCheckBox("CSF")
        self._white_matter_cb = QCheckBox("White Matter")
        self._gray_matter_cb = QCheckBox("Gray Matter")
        # The intermediate skull-strip result itself, not a tissue class -
        # shown on request so the user can judge skull-strip quality
        # directly, since white/gray matter's own accuracy is bounded by it.
        self._brain_mask_cb = QCheckBox("Brain Mask")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        # One row: short names, colored like their overlays (hover for the full name).
        self._white_matter_cb.setText("WM")
        self._gray_matter_cb.setText("GM")
        overlays = [
            (self._white_matter_cb, "White Matter", 0, 0),
            (self._gray_matter_cb, "Gray Matter", 0, 1),
            (self._csf_cb, "CSF", 0, 2),
            (self._brain_mask_cb, "Brain Mask", 0, 3),
        ]
        for checkbox, name, row, column in overlays:
            checkbox.setEnabled(False)
            checkbox.setToolTip("Open a study first")
            checkbox.setStyleSheet(f"QCheckBox {{ color: {_BRAIN_TISSUE_COLORS[name]}; }}")
            checkbox.toggled.connect(lambda checked, n=name: self._on_brain_tissue_checkbox_toggled(n, checked))
            grid.addWidget(checkbox, row, column)
        layout.addLayout(grid)

        return box

    def _brain_mask_for_current_slice(self, slice_hu: np.ndarray) -> np.ndarray:
        return self._brain_tissue.brain_mask(self._study.current_slice, slice_hu)

    def _ensure_brain_tissue_result_for_current_slice(self):
        return self._brain_tissue.result(self._study.current_slice)

    def _on_brain_tissue_method_selected(self, method: str) -> None:
        self._brain_tissue.set_method(method)
        self._update_brain_tissue_settings_button()

    def _brain_tissue_settings_handlers(self) -> dict:
        # The anchored prototype has no settings yet.
        return {
            _BRAIN_TISSUE_METHOD_FCM: self._on_fcm_settings_clicked,
            _BRAIN_TISSUE_METHOD_ADAPTIVE: self._on_adaptive_settings_clicked,
            _BRAIN_TISSUE_METHOD_THRESHOLD: self._on_brain_tissue_settings_clicked,
        }

    def _on_brain_tissue_method_settings_clicked(self) -> None:
        handler = self._brain_tissue_settings_handlers().get(self._brain_tissue.method)
        if handler is not None:
            handler()

    def _update_brain_tissue_settings_button(self) -> None:
        has_settings = self._brain_tissue.method in self._brain_tissue_settings_handlers()
        self._brain_tissue_settings_btn.setEnabled(has_settings)
        self._set_tip(self._brain_tissue_settings_btn, "" if has_settings else "The prototype has no settings yet")

    def _on_fcm_settings_clicked(self) -> None:
        dialog = FcmSettingsDialog(self._brain_tissue.settings[_BRAIN_TISSUE_METHOD_FCM], self._brain_tissue.default_settings[_BRAIN_TISSUE_METHOD_FCM], parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._brain_tissue.set_settings(_BRAIN_TISSUE_METHOD_FCM, dialog.settings())

    def _on_adaptive_settings_clicked(self) -> None:
        dialog = AdaptiveSettingsDialog(self._brain_tissue.settings[_BRAIN_TISSUE_METHOD_ADAPTIVE], self._brain_tissue.default_settings[_BRAIN_TISSUE_METHOD_ADAPTIVE], parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._brain_tissue.set_settings(_BRAIN_TISSUE_METHOD_ADAPTIVE, dialog.settings())

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
        for checkbox in (self._csf_cb, self._white_matter_cb, self._gray_matter_cb, self._brain_mask_cb):
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
            self._set_tip(checkbox, "Open a study first")
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
        if self._study.volume is None:
            self._brain_mask_cb.setEnabled(False)
            self._set_tip(self._brain_mask_cb, "Open a study first")
            return
        weights_path = self._segmentation.config.get("segmentation", {}).get("anatomy_weights_path")
        if not weights_path:
            self._brain_mask_cb.setEnabled(False)
            self._set_tip(self._brain_mask_cb, "Segmentation model weights not configured")
            return
        if self._study.current_slice not in self._segmentation.cache:
            self._brain_mask_cb.setEnabled(False)
            self._set_tip(self._brain_mask_cb, "Waiting for anatomy segmentation on this slice…")
            return
        self._brain_mask_cb.setEnabled(True)
        self._set_tip(self._brain_mask_cb, "")

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
        self._set_tip(self._check_stroke_btn, "Open a study first")
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
        self._stroke.check_slice(self._study.current_slice)

    def _on_stroke_checked(self, index: int, _from_cache: bool) -> None:
        """A slice's stroke result is available: the lesion overlay may be shown."""
        self._check_stroke_btn.setEnabled(True)
        if index != self._study.current_slice:
            return
        self._stroke_lesion_cb.setEnabled(True)
        if self._stroke_lesion_cb.isChecked():
            self._roi_viewport.set_overlay("stroke_lesion", self._stroke.lesion_mask(self._study.current_slice), color=CRITICAL, label="Suspected Lesion")

    def _on_stroke_checkbox_toggled(self, checked: bool) -> None:
        if not checked:
            self._roi_viewport.clear_overlay("stroke_lesion")
            return
        if self._stroke.lesion_mask(self._study.current_slice) is not None:
            self._roi_viewport.set_overlay("stroke_lesion", self._stroke.lesion_mask(self._study.current_slice), color=CRITICAL, label="Suspected Lesion")

    # ---- segmentation/stroke lifecycle (volume load, slice change, close study) ----

    def _update_segmentation_availability(self) -> None:
        """Enables/disables the Segment Slice and Check for Lesion buttons
        based on whether their respective weights are configured (see
        `infrastructure.config`) - checked once per volume load rather than
        on every slice change, since configuration doesn't change mid-session.
        """
        segmentation_config = self._segmentation.config.get("segmentation", {})
        has_anatomy_weights = bool(segmentation_config.get("anatomy_weights_path"))
        self._segment_slice_action.setEnabled(has_anatomy_weights)
        self._set_tip(self._segment_slice_action, "" if has_anatomy_weights else "Segmentation model weights not configured")
        self._segment_volume_action.setEnabled(has_anatomy_weights)
        self._set_tip(self._segment_volume_action, "" if has_anatomy_weights else "Segmentation model weights not configured")
        self._select_segments_btn.setEnabled(has_anatomy_weights)
        self._set_tip(self._select_segments_btn, "" if has_anatomy_weights else "Segmentation model weights not configured")
        self._show_segments_cb.setEnabled(has_anatomy_weights)
        self._set_tip(self._show_segments_cb, "" if has_anatomy_weights else "Segmentation model weights not configured")

        has_stroke_weights = bool(segmentation_config.get("stroke_weights_path"))
        self._check_stroke_btn.setEnabled(has_stroke_weights)
        self._set_tip(self._check_stroke_btn, "" if has_stroke_weights else "Stroke model weights not configured")

        # No weights/config to gate on (HU-threshold only) - always available
        # once a study is loaded.
        self._csf_cb.setEnabled(True)
        self._set_tip(self._csf_cb, "")
        self._white_matter_cb.setEnabled(True)
        self._set_tip(self._white_matter_cb, "")
        self._gray_matter_cb.setEnabled(True)
        self._set_tip(self._gray_matter_cb, "")
        # "Brain Mask" has its own, segmentation-dependent availability - see
        # `_update_brain_mask_checkbox_availability`.
        self._update_brain_mask_checkbox_availability()

    def _reset_segmentation_display(self) -> None:
        """Segmentation widgets back to their no-results state (new/closed series)."""
        self._reset_auto_rois_button()
        self._segment_slice_action.setEnabled(False)
        self._set_tip(self._segment_slice_action, "Open a study first")
        self._segment_volume_action.setEnabled(False)
        self._set_tip(self._segment_volume_action, "Open a study first")
        self._cancel_segment_volume_btn.setVisible(False)
        self._select_segments_btn.setEnabled(False)
        self._set_tip(self._select_segments_btn, "Open a study first")
        self._show_segments_cb.setEnabled(False)
        self._set_tip(self._show_segments_cb, "Open a study first")
        self._show_segments_cb.blockSignals(True)
        self._show_segments_cb.setChecked(False)  # back to its default (2026-09-24) - opt in, not on by default
        self._show_segments_cb.blockSignals(False)
        if self._segment_selection_dialog is not None:
            self._segment_selection_dialog.reset()

    def _reset_stroke_display(self) -> None:
        self._check_stroke_btn.setEnabled(False)
        self._set_tip(self._check_stroke_btn, "Open a study first")
        self._stroke_lesion_cb.blockSignals(True)
        self._stroke_lesion_cb.setChecked(False)
        self._stroke_lesion_cb.setEnabled(False)
        self._stroke_lesion_cb.blockSignals(False)

    def _refresh_segmentation_for_current_slice(self) -> None:
        """Per-slice-change refresh: re-applies the *persisted* structure
        selection (`SegmentationViewModel.selected_names`) using this slice's
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
        self._segmentation.current_masks = None
        cached = self._segmentation.cache.get(self._study.current_slice)
        # Browsing only redraws this slice's cached overlay: the status bar
        # and the Segment actions belong to the segmentation *task* (e.g. a
        # running volume segmentation) and must not change on a slice change.
        if cached is not None:
            self._apply_segmentation_masks(cached)

    def _refresh_stroke_for_current_slice(self) -> None:
        self._stroke_lesion_cb.blockSignals(True)
        self._stroke_lesion_cb.setChecked(False)
        self._stroke_lesion_cb.setEnabled(False)
        self._stroke_lesion_cb.blockSignals(False)
        if self._study.current_slice in self._stroke.cache:
            self._stroke_status_label.setText("Done (cached)")
            self._on_stroke_checked(self._study.current_slice, True)
        else:
            self._stroke_status_label.setText("—")

    def _build_roi_box(self) -> QWidget:
        # Falx and septum ROI pairs side by side (presentation.viewmodels.
        # roi_pairs; docs/qa_method/head_ct_classifier.md, D11). Structured
        # table (ROI/Position/Size), not per-ROI QLabels: the old labels' full
        # coordinate text didn't fit this box and forced the sidebar wider.
        box = self._section("2. ROI Selection", "roi")
        self._roi_section = box
        layout = box.content_layout
        layout.addWidget(self._hint_label(STEP_HINTS["roi"]))

        # Auto ROIs (domain.roi_selection): pressed = both pairs are placed on
        # every eligible segmented slice, each ROI draggable until confirmed;
        # pressed again = the automatic ROIs are removed (manual ones stay).
        self._auto_rois_btn = QPushButton("Start auto ROIs")
        self._auto_rois_btn.setCheckable(True)
        self._auto_rois_btn.setEnabled(False)
        self._set_tip(self._auto_rois_btn, "Needs anatomy segmentation first")
        self._auto_rois_btn.toggled.connect(self._on_auto_rois_toggled)
        layout.addWidget(self._auto_rois_btn)
        # This slice's auto-ROI state and where ROIs exist is shown in the
        # button's tooltip (`RoiViewModel.status_text`).

        # Draw-then-assign: a rectangle drawn on the image (any drag on empty
        # image - see `RoiViewport`) is saved into a slot with right-click >
        # Save as (`_roi_context_menu`).
        # The ROI table is optional detail - "ROI details" shows/hides it (remembered).
        self._roi_details_btn = QPushButton("▸ ROI details")
        self._roi_details_btn.setCheckable(True)
        self._roi_details_btn.setFlat(True)
        self._roi_details_btn.setStyleSheet("text-align: left;")
        self._roi_details_btn.toggled.connect(self._on_roi_details_toggled)

        self._roi_table = QTableWidget(len(ROI_KEYS), 3)
        self._roi_table.setHorizontalHeaderLabels(["ROI", "Position", "Size"])
        self._roi_table.verticalHeader().setVisible(False)
        self._roi_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        # Clicking a row selects that ROI on the image (Delete then removes it).
        self._roi_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._roi_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._roi_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._roi_table.setShowGrid(False)
        self._roi_table.cellClicked.connect(self._on_roi_table_clicked)
        self._roi_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._roi_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header_view = self._roi_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._roi_table.setStyleSheet(
            f"QTableWidget {{ background: {SURFACE}; border: none; }}"
            f"QHeaderView::section {{ background: {SURFACE}; color: {TEXT_MUTED}; border: none;"
            f" border-bottom: 1px solid {BORDER}; padding: 4px; font-weight: 600; font-size: 10px; }}"
            f"QTableWidget::item {{ padding: 4px; border-bottom: 1px solid {BORDER}; }}"
            f"QTableWidget::item:selected {{ background: {SURFACE_RAISED}; }}"
        )

        # Same colors `RoiViewport` draws the rectangles in (`_ROI_STYLES`),
        # so the row and the rectangle it describes are visually tied together.
        self._roi_position_items: dict[str, QTableWidgetItem] = {}
        self._roi_size_items: dict[str, QTableWidgetItem] = {}
        for row, key in enumerate(ROI_KEYS):
            name_item = QTableWidgetItem(f"{_ROI_SHORT_LABELS[key]} · {ROLE_SHORT[key]}")
            name_item.setToolTip(ROLE_LABELS[key])
            name_item.setForeground(QColor(_ROI_TABLE_COLORS[key]))
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

        # A fixed height for the fixed rows (not `QTableWidget`'s default
        # `Expanding` vertical size policy) keeps this box compact instead of
        # soaking up sidebar space Run Metrics needs.
        self._roi_table.resizeRowsToContents()
        table_height = self._roi_table.horizontalHeader().sizeHint().height()
        table_height += sum(self._roi_table.rowHeight(r) for r in range(self._roi_table.rowCount()))
        self._roi_table.setFixedHeight(table_height + 2)  # +2 for the frame
        layout.addWidget(self._roi_table)

        edit_row = QHBoxLayout()
        edit_row.addWidget(self._roi_details_btn)
        edit_row.addStretch(1)
        self._delete_roi_btn = QPushButton("Delete ROI")
        self._delete_roi_btn.setEnabled(False)
        self._set_tip(self._delete_roi_btn, "Select an ROI on the image or in the table first")
        self._delete_roi_btn.clicked.connect(self._on_delete_roi_clicked)
        self._restore_auto_btn = QPushButton("Restore automatic")
        self._restore_auto_btn.setEnabled(False)
        self._set_tip(self._restore_auto_btn, "Nothing changed by hand on this slice")
        self._restore_auto_btn.clicked.connect(self._on_restore_auto_clicked)
        edit_row.addWidget(self._delete_roi_btn)
        edit_row.addWidget(self._restore_auto_btn)
        # Above the table, so opening the details doesn't move the buttons.
        layout.insertLayout(layout.indexOf(self._roi_table), edit_row)

        # Confirming locks the ROIs and enables the QA run - its own sub-box,
        # so this step is not overlooked.
        confirm_box = QGroupBox("Confirm ROIs")
        confirm_box.setObjectName("subBox")
        self._confirm_box = confirm_box
        confirm_layout = QVBoxLayout(confirm_box)
        confirm_layout.addWidget(self._hint_label(STEP_HINTS["confirm"]))
        self._confirm_rois_cb = QCheckBox("ROIs checked - lock them for QA")
        self._confirm_rois_cb.setEnabled(False)
        self._confirm_rois_cb.setStyleSheet("QCheckBox { font-weight: 700; }")
        self._confirm_rois_cb.toggled.connect(self._on_confirm_rois_toggled)
        confirm_layout.addWidget(self._confirm_rois_cb)
        layout.addWidget(confirm_box)

        self._roi_table.setVisible(False)
        self._roi_details_btn.setChecked(self._ui_settings.get_bool(ROI_DETAILS_OPEN))
        return box

    def _on_roi_details_toggled(self, open_: bool) -> None:
        self._roi_table.setVisible(open_)
        self._roi_details_btn.setText(("▾" if open_ else "▸") + " ROI details")
        self._ui_settings.set_bool(ROI_DETAILS_OPEN, open_)

    def _update_roi_table_row(self, key: str, rect: RoiRect) -> None:
        position_item = self._roi_position_items[key]
        position_item.setText(f"{rect.x0}–{rect.x1}, {rect.y0}–{rect.y1}")
        position_item.setForeground(QColor(TEXT))
        size_item = self._roi_size_items[key]
        size_item.setText(f"{rect.width}×{rect.height} px")
        size_item.setForeground(QColor(TEXT))

    def _reset_roi_table(self) -> None:
        for key in ROI_KEYS:
            self._reset_roi_table_row(key)

    def _reset_roi_table_row(self, key: str) -> None:
        self._roi_position_items[key].setText("not set")
        self._roi_position_items[key].setForeground(QColor(TEXT_MUTED))
        self._roi_size_items[key].setText("—")
        self._roi_size_items[key].setForeground(QColor(TEXT_MUTED))

    def _on_confirm_rois_toggled(self, checked: bool) -> None:
        """Confirmed = the ROIs on every slice are final: locked on the image
        (no moving, drawing or deleting), the edit buttons and Auto ROIs
        disabled, Run All enabled. Unchecking makes them editable again.
        """
        self._roi.set_confirmed(checked)
        self._run_all_btn.setEnabled(checked)
        self._set_tip(self._run_all_btn, "" if checked else "Confirm ROIs to enable")
        self._roi_viewport.set_locked(checked)
        self._update_roi_edit_buttons()
        self._update_auto_rois_availability()

    def _build_metrics_box(self) -> QWidget:
        box = self._section("3. Run QA metrics", "qa")
        self._metrics_box = box
        layout = box.content_layout
        self._metrics_hint = self._hint_label(STEP_HINTS["qa"])
        layout.addWidget(self._metrics_hint)

        # The run button first and large - it is the step's action.
        self._run_all_btn = QPushButton("▶  Run QA")
        self._run_all_btn.setObjectName("runAllBtn")
        self._run_all_btn.setMinimumHeight(36)
        self._run_all_btn.setStyleSheet(
            f"QPushButton#runAllBtn {{ font-size: 14px; font-weight: 700; padding: 6px 18px; }}"
            f"QPushButton#runAllBtn:enabled {{ background: {ACCENT}; color: {BG}; border: none; border-radius: 4px; }}"
        )
        self._run_all_btn.setEnabled(False)
        self._set_tip(self._run_all_btn, "Confirm ROIs to enable")
        self._run_all_btn.clicked.connect(self._on_run_all_clicked)

        # Progress and Cancel of the run are in the status bar.
        header = QHBoxLayout()
        header.addWidget(self._run_all_btn)
        self._qa_recipe_label = QLabel("Legacy QA recipe (Octave)")
        self._qa_recipe_label.setStyleSheet(f"color: {ACCENT_2}; font-weight: 600;")
        self._set_tip(self._qa_recipe_label, "Tools > Advanced > QA Recipe - switch back to Physical (default)")
        self._qa_recipe_label.setVisible(self._qa.recipe == RECIPE_OCTAVE)
        header.addWidget(self._qa_recipe_label)
        header.addStretch(1)
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
        self._metrics_table.resizeRowsToContents()
        self._metrics_table.setMinimumHeight(
            self._metrics_table.horizontalHeader().sizeHint().height()
            + sum(self._metrics_table.rowHeight(r) for r in range(self._metrics_table.rowCount()))
            + 2
        )
        layout.addWidget(self._metrics_table, 1)

        return box

    def _set_metric_status(self, key: str, role: str, text: str, tooltip: str = "") -> None:
        item = self._metric_status_items[key]
        item.setText(text)
        item.setForeground(QColor(_STATUS_COLORS[role]))
        item.setToolTip(tooltip)

    def _on_qa_recipe_selected(self, recipe: str) -> None:
        dropped = self._qa.set_recipe(recipe)
        self._qa_recipe_label.setVisible(recipe == RECIPE_OCTAVE)
        # Results of the two recipes are on different scales - never shown
        # side by side; the confirmed ROIs stay, Run QA recomputes.
        if dropped:
            self._reset_qa_results()
            self._show_view_page(0)

    def _update_qa_recipe_actions(self) -> None:
        # No switching mid-batch - that would mix recipes within one study.
        for action in self._qa_recipe_actions.values():
            action.setEnabled(not self._qa.is_running())

    def _on_run_all_clicked(self) -> None:
        """QA for every complete ROI pair on every slice (the confirmed ones) -
        local QA per slice and pair, then the study-level summary on the
        Results page."""
        jobs = self._qa_ready_jobs()
        if not jobs:
            return
        for key, _ in _METRIC_ROWS:
            self._set_metric_status(key, "statusQueued", "running")
        self._run_all_btn.setEnabled(False)
        self._cancel_run_btn.setText("Cancel")
        self._cancel_run_btn.setEnabled(True)
        self._cancel_run_btn.setVisible(True)
        self._confirm_rois_cb.setEnabled(False)
        self._qa.run(jobs)
        self._update_qa_recipe_actions()

    def _on_cancel_run_clicked(self) -> None:
        self._qa.cancel()
        self._cancel_run_btn.setText("Cancelling…")
        self._cancel_run_btn.setEnabled(False)

    def _on_qa_finished(self, _cancelled: bool) -> None:
        """The QA batch is over (done or cancelled): controls back, and - if
        anything was measured - the Results page."""
        self._update_qa_recipe_actions()
        self._cancel_run_btn.setVisible(False)
        self._run_all_btn.setEnabled(self._confirm_rois_cb.isChecked())
        self._update_confirm_availability()
        self._refresh_metrics_box()
        if not self._qa.results:
            return
        self._results_tab_btn.setEnabled(True)
        self._set_tip(self._results_tab_btn, "")
        self._export_action.setEnabled(True)
        self._set_tip(self._export_action, "")
        shown = self._study.current_slice if self._study.current_slice in self._results.slices else self._results.slices[0]
        self._show_result_for_slice(shown)
        self._show_view_page(1)

    def _on_results_tab_changed(self, _tab_index: int) -> None:
        self._update_metrics_box_title()
        self._refresh_metrics_box()

    def _showing_study_summary(self) -> bool:
        return self._view_stack.currentIndex() == 1 and self._results_tabs.currentIndex() == 1

    def _refresh_metrics_box(self) -> None:
        """Run Metrics box = the selected pair's study averages while the Study
        summary tab is shown, else the displayed slice's metrics (the Per
        slice tab keeps the viewer on the same slice).
        """
        if self._showing_study_summary() and self._qa.results:
            self._show_study_average_metrics()
        else:
            self._show_metrics_for_slice(self._study.current_slice)

    def _show_study_average_metrics(self) -> None:
        pair = self._results.pair if self._results_by_pair(self._results.pair) else next(
            pair for pair in PAIRS if self._results_by_pair(pair)
        )
        summary = summarize_metrics(self._results_by_pair(pair).values())
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
                tooltip=f"{PAIR_LABELS[pair]}: mean over {metric.count} slice(s), SD {number.format(metric.std)}{unit}",
            )
            self._metric_value_items[key].setText(f"{number.format(metric.mean)}{unit}")

    def _show_metrics_for_slice(self, index: int) -> None:
        """Sidebar metrics table = the displayed slice's QA result for the
        selected pair (or the other one), its error, or "—" if it has none.
        """
        pair = self._shown_pair(index)
        if pair is None:
            pair = next((p for p in [self._results.pair, *PAIRS] if (index, p) in self._qa.errors), self._results.pair)
        result = self._qa.results.get((index, pair))
        error = self._qa.errors.get((index, pair))
        if result is not None:
            self._set_metric_status("mtf", "statusDone", "done")
            self._metric_value_items["mtf"].setText(f"{result.mtf_area:.3f} mm⁻¹")
            self._set_metric_status("nps", "statusDone", "done")
            # nps_roi2 (homogeneous ROI) is the true NPS - nps_roi1 (structured
            # ROI) is what WP2 calls "PS" (see the PS chart).
            self._metric_value_items["nps"].setText(f"{result.nps_roi2.var_nps:.5f} HU²mm²")
            self._set_metric_status("ratio", "statusDone", "done")
            self._metric_value_items["ratio"].setText(f"{result.ratio.ratio_lf_hf:.2f} LF/HF")
            self._set_metric_status("contrast", "statusDone", "done")
            self._metric_value_items["contrast"].setText(f"{result.contrast_roi1:.1f} HU")
        elif error is not None:
            for key, _ in _METRIC_ROWS:
                self._set_metric_status(key, "statusError", "error", tooltip=error)
                self._metric_value_items[key].setText("")
        elif self._qa.jobs is not None and any(
            job[0] == index for job in self._qa.jobs + [self._qa.active_job] if job is not None
        ):
            for key, _ in _METRIC_ROWS:
                self._set_metric_status(key, "statusQueued", "running")
                self._metric_value_items[key].setText("")
        else:
            for key, _ in _METRIC_ROWS:
                self._set_metric_status(key, "statusQueued", "—")
                self._metric_value_items[key].setText("")

    def _reset_qa_results(self) -> None:
        """QA widgets back to their no-results state (the results were dropped:
        new/closed series or recipe change)."""
        self._cancel_run_btn.setVisible(False)
        self._results_slice_label.setText("No results yet")
        self._reset_metrics_display()
        self._results_tab_btn.setEnabled(False)
        self._set_tip(self._results_tab_btn, "Run metrics to see results")
        self._export_action.setEnabled(False)
        self._set_tip(self._export_action, "Run metrics first")

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


def main() -> None:
    register_windows_app_id()  # before any window, so the taskbar shows BICTool's icon, not Python's
    app = QApplication(sys.argv)
    app.setApplicationName("BICTool")
    app.setWindowIcon(app_icon())
    window = MainWindowView()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
