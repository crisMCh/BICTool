"""MainViewModel: the application's ViewModels and the workflow between them.

Owns one ViewModel per area (study, segmentation, WM/GM, ROIs, QA run,
results, export, stroke) and what ties them together:

- a series is opened: everything that belonged to the previous series is
  dropped (the primary series' segmentation is kept while another series is
  shown, and comes back with it), then the primary series is segmented
  automatically when a model is configured;
- the study is closed or another one opened: everything is dropped, the
  kept primary too;
- a slice got segmented: its WM/GM result is recomputed and the automatic
  ROIs follow;
- the WM/GM method or settings changed: the white-matter ROI2s are re-placed.

The view binds to the child ViewModels for their own area and to the three
signals below for the workflow; it only resets and redraws widgets. QtCore only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from bictool.presentation.viewmodels.brain_tissue_viewmodel import BrainTissueViewModel
from bictool.presentation.viewmodels.export_viewmodel import ExportViewModel
from bictool.presentation.viewmodels.qa_run_viewmodel import QaRunViewModel
from bictool.presentation.viewmodels.results_viewmodel import ResultsViewModel
from bictool.presentation.viewmodels.roi_viewmodel import RoiViewModel
from bictool.presentation.viewmodels.segmentation_viewmodel import SegmentationViewModel
from bictool.presentation.viewmodels.stroke_viewmodel import StrokeViewModel
from bictool.presentation.viewmodels.study_viewmodel import StudyViewModel

logger = logging.getLogger(__name__)


class MainViewModel(QObject):
    series_opened = Signal(object, bool)  # (volume, is primary) - every area was reset; show slice 0
    study_closed = Signal()               # every area was reset; back to the no-study state
    slice_segmented = Signal(int)         # a slice's segmentation landed; WM/GM and auto ROIs updated

    def __init__(
        self,
        submit: Callable[[object], None],
        config: dict,
        segmentation_mapping: dict[str, list[str]],
        head_ct_model: Callable[[], object],
        tissue_defaults: tuple,
        parent: QObject | None = None,
    ) -> None:
        """`submit` runs a background job; `config` = the app configuration
        (model weight paths); `segmentation_mapping` = shown structure name ->
        model classes; `head_ct_model()` = the study-verdict model or None;
        `tissue_defaults` = the HU threshold, FCM and adaptive methods' default
        settings (the settings dialogs' dataclasses)."""
        super().__init__(parent)
        # Off in tests that load a volume without wanting a segmentation job.
        self.auto_segment_primary = True
        self.study = StudyViewModel(submit, self)
        self.segmentation = SegmentationViewModel(
            submit, config, lambda: self.study.volume, segmentation_mapping, self
        )
        self.stroke = StrokeViewModel(submit, lambda: self.segmentation.config, lambda: self.study.volume, self)
        self.brain_tissue = BrainTissueViewModel(
            lambda: self.study.volume, lambda: self.segmentation.cache, *tissue_defaults, self
        )
        self.roi = RoiViewModel(
            submit,
            lambda: self.study.volume,
            lambda: self.study.current_slice,
            lambda: self.segmentation.cache,
            lambda: self.brain_tissue.cache,
            lambda: self.brain_tissue.classifier(),
            self,
        )
        self.qa = QaRunViewModel(
            submit, lambda: self.study.volume, self.roi.effective, lambda: self.segmentation.cache, self
        )
        self.results = ResultsViewModel(
            self.qa, head_ct_model, self,
            volume=lambda: self.study.volume, segmentation_cache=lambda: self.segmentation.cache,
        )
        self.export = ExportViewModel(self.qa, self.results, lambda: self.study.volume, self)

        self.study.volume_loaded.connect(self._on_volume_loaded)
        self.study.closed.connect(self._on_study_closed)
        # A fresh study: the kept primary segmentation belongs to the old one.
        self.study.series_list_changed.connect(lambda _labels: self.segmentation.forget_primary())
        self.segmentation.slice_segmented.connect(self._on_slice_segmented)
        self.brain_tissue.settings_changed.connect(self._on_tissue_settings_changed)
        self.qa.reset_done.connect(self.export.clear)

    def has_anatomy_weights(self) -> bool:
        return bool(self.segmentation.config.get("segmentation", {}).get("anatomy_weights_path"))

    # ---- workflow ----

    def _reset_series_state(self) -> None:
        """Everything that belongs to the shown series (results are keyed by
        slice index - a stale one must never reappear under a new series)."""
        self.segmentation.reset()
        self.stroke.reset()
        self.brain_tissue.reset()
        self.roi.reset()
        self.qa.reset()

    def _on_volume_loaded(self, volume, is_primary: bool, force_reload: bool, previous_was_primary: bool) -> None:
        # Leaving the primary series: keep its segmentation (only the primary's -
        # keeping every visited series' NIfTI volume and masks costs too much memory).
        if previous_was_primary:
            self.segmentation.backup_primary()
        self._reset_series_state()
        if is_primary and force_reload:
            # Its files may have changed - the kept results are not trustworthy.
            self.segmentation.forget_primary()
        elif is_primary:
            # Back on the primary series: its results return (unfinished
            # slices are segmented by the automatic job below).
            self.segmentation.restore_primary()
        self.series_opened.emit(volume, is_primary)
        if is_primary:
            self._maybe_auto_segment()

    def _maybe_auto_segment(self) -> None:
        """The primary series is segmented without a click when a model is
        configured. Silently nothing when none is set up or TensorFlow is
        missing - an automatic start never shows an error dialog (a broken
        configured weights path still does, via `segment_volume`)."""
        if not self.auto_segment_primary or not self.has_anatomy_weights():
            return
        try:
            # Deferred to first actual use (TensorFlow's import cost).
            from bictool.domain.segmentation import head_ct_unet  # noqa: F401
        except ImportError:
            return
        self.segmentation.segment_volume()

    def _on_study_closed(self) -> None:
        # A future study's primary series is unrelated to this one.
        self.segmentation.forget_primary()
        self._reset_series_state()
        self.study_closed.emit()

    def _on_tissue_settings_changed(self) -> None:
        # Every WM/GM result was dropped: the white-matter ROI2s are re-placed.
        self.roi.restart_roi2()

    def _on_slice_segmented(self, index: int) -> None:
        if index == self.study.current_slice:
            # Real CNN data now exists for the shown slice, which the brain
            # mask prefers over the HU-threshold fallback - recompute its WM/GM.
            self.brain_tissue.invalidate(index)
        self.slice_segmented.emit(index)
        self.roi.on_slice_segmented(index)
