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

from bictool.api import results as R
from bictool.application import pipeline, plugin_manager
from bictool.application.qa_settings import Settings
from bictool.application.roi_store import PAIRS, roi_key
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
            submit, lambda: self.study.volume, self.roi.effective, lambda: self.segmentation.cache, self,
            tissue_settings=lambda: self.brain_tissue.tissue_settings(),
        )
        self.results = ResultsViewModel(
            self.qa, head_ct_model, self,
            volume=lambda: self.study.volume, segmentation_cache=lambda: self.segmentation.cache,
            segmentation_plausibility=lambda: self.segmentation.plausibility,
        )
        # A late segmentation check updates the summary of results already shown.
        self.segmentation.plausibility_changed.connect(lambda: self.results.refresh() if self.qa.results else None)
        self.export = ExportViewModel(self.qa, self.results, lambda: self.study.volume, self, study_result=self.study_result)

        self.study.volume_loaded.connect(self._on_volume_loaded)
        self.study.closed.connect(self._on_study_closed)
        # A fresh study: the kept primary segmentation belongs to the old one.
        self.study.series_list_changed.connect(lambda _labels: self.segmentation.forget_primary())
        self.segmentation.slice_segmented.connect(self._on_slice_segmented)
        self.brain_tissue.settings_changed.connect(self._on_tissue_settings_changed)
        self.qa.reset_done.connect(self.export.clear)

    def study_result(self) -> R.StudyResult | None:
        """The shown QA run as the API's `StudyResult` (what `api.Study.measure` returns
        for the same ROIs) - export-format plug-ins write it. None without results."""
        volume, qa, store = self.study.volume, self.qa, self.roi.store
        if volume is None or not qa.results:
            return None
        series = R.SeriesInfo.from_volume(volume)
        measurements = []
        for (index, pair), quality in qa.results.items():
            key1, key2 = roi_key(pair, "roi1"), roi_key(pair, "roi2")
            effective = store.effective(index)
            measurements.append(R.make_measurement(
                R.make_slice_ref(volume, index), pair, effective[key1], effective[key2],
                store.is_manual(index, key1), store.is_manual(index, key2), store.auto_region(index, key2),
                quality, False, qa.plugin_values.get((index, pair), ()),
            ))
        failed = [R.Failure(R.make_slice_ref(volume, index), pair, message) for (index, pair), message in qa.errors.items()]
        for index, pair in sorted(store.unplaced):
            if pair not in store.complete_pairs(index) and roi_key(pair, "roi1") in store.effective(index):
                failed.append(R.Failure(R.make_slice_ref(volume, index), pair, "No ROI2 found"))
        results = self.results
        return R.StudyResult(
            label=f"study - {series.description or 'series'}",
            series=series,
            settings=Settings(recipe=qa.recipe, tissue=self.brain_tissue.tissue_settings()),
            rois_reviewed=True,  # the program measures confirmed ROIs only
            rois_edited=store.is_edited(),
            measurements=tuple(measurements),
            failed=tuple(sorted(failed, key=lambda f: (f.slice.index, PAIRS.index(f.pair)))),
            skipped=self._skipped_slices(),
            segmented_slices=len(self.segmentation.cache),
            verdict=R.make_verdict(results.verdict),
            artefacts=R.make_artefacts(results.artefacts),
            basal_ganglia=R.make_basal_ganglia(results.basal_ganglia),
            segmentation_check=R.make_segmentation_check(results.plausibility),
            plugins=dict(plugin_manager.current().versions),
            plugin_values=qa.study_plugin_values,
            slice_plugin_values=dict(qa.slice_plugin_values),
            plugin_hints=qa.plugin_hints,
            _raw=R._Raw(results.verdict, results.artefacts, results.basal_ganglia, results.plausibility,
                        qa.plugin_tissue_method),
        )

    def _skipped_slices(self) -> dict[str, tuple[int, ...]]:
        """Per pair: the segmented slices where the structure was found but there is too
        little brain for automatic ROIs (as `api.Study.place_rois` records them)."""
        volume, cache, roi = self.study.volume, self.segmentation.cache, self.roi
        skipped: dict[str, list[int]] = {pair: [] for pair in PAIRS}
        if not roi.auto_on:
            return {}
        for index in sorted(set(cache) - roi.eligible):
            for pair in PAIRS:
                image = volume.get_slice_hu(index)
                if pipeline.auto_roi1(cache[index].masks, pair, volume.pixel_spacing_mm, roi.midline, image) is not None:
                    skipped[pair].append(index)
        return {pair: tuple(slices) for pair, slices in skipped.items()}

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
