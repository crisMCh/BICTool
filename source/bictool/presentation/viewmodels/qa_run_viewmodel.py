"""QaRunViewModel: the image-quality measurement over the confirmed ROI pairs.

Owns the QA recipe (physical by default; the legacy Octave one under
Tools > Advanced), the job queue - one job per (slice, ROI pair) - the job
running in the background (`application.worker.Worker`, one at a time),
the results and errors per job, and cancelling (the running job finishes,
no further one starts). Each job is `pipeline.measure_pair` (the segmentation
seeds the MTF contour on the structure) followed by the plug-ins' values for
that pair; with plug-ins that report slice or series values or artefact hints,
one more job at the end runs them (`pipeline.plugin_study_phase`, as the API:
`plugin_values`, `slice_plugin_values`, `study_plugin_values`, `plugin_hints` -
descriptive only; `plugin_markers(slice)` = where plug-ins measured).
QtCore only - no widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from bictool.application import pipeline, plugin_manager
from bictool.application.qa_settings import TissueSettings
from bictool.application.worker import Worker
from bictool.domain.results.study_quality import DEFAULT_RECIPE, RECIPE_PHYSICAL
from bictool.presentation.viewmodels.display_text import format_slice_ranges
from bictool.presentation.viewmodels.roi_pairs import PAIR_LABELS, roi_key

logger = logging.getLogger(__name__)

Job = tuple[int, str]  # (slice index, ROI pair)


class QaRunViewModel(QObject):
    progress_changed = Signal(str)   # status-bar text ("QA 5/33", "QA done: 33 ROI pair(s)")
    running_changed = Signal(bool)
    job_done = Signal(object)        # the (slice, pair) whose result or error just arrived
    finished = Signal(bool)          # batch over; True = it was cancelled
    reset_done = Signal()            # every result dropped (new/closed study, recipe change)

    def __init__(
        self,
        submit: Callable[[object], None],
        volume: Callable[[], object],
        rois: Callable[[int], dict],
        segmentation_cache: Callable[[], dict],
        parent: QObject | None = None,
        tissue_settings: Callable[[], TissueSettings] | None = None,
    ) -> None:
        """`rois(slice)` = the slice's effective ROIs per slot; `segmentation_cache()`
        = the anatomy results (MTF seed hint); `tissue_settings()` = the WM/GM method
        whose masks plug-ins get (default: the default method)."""
        super().__init__(parent)
        self._submit = submit
        self._volume = volume
        self._rois = rois
        self._segmentation_cache = segmentation_cache
        self._tissue_settings = tissue_settings or TissueSettings
        self.recipe = DEFAULT_RECIPE
        self.jobs: list[Job] | None = None  # pending jobs while a batch runs, else None
        self.total = 0
        self.active_job: Job | None = None
        self.cancel_requested = False
        self.active_worker: Worker | None = None
        self.results: dict[Job, object] = {}  # (slice, pair) -> SliceQualityResult
        self.errors: dict[Job, str] = {}
        self.plugin_values: dict[Job, tuple] = {}  # (slice, pair) -> PluginValues of each plug-in
        self.study_plugin_values: tuple = ()      # PluginValues for the whole series
        self.slice_plugin_values: dict[int, tuple] = {}  # slice -> PluginValues of each plug-in
        self.plugin_hints: tuple = ()             # PluginHints: possible artefacts plug-ins noticed
        self.plugin_tissue_method: str | None = None  # the WM/GM method whose masks plug-ins used
        self._contexts: dict[Job, object] = {}     # what the plug-ins saw of each pair
        self._study_job_started = False

    # ---- queries ----

    def is_running(self) -> bool:
        return self.jobs is not None

    def plugin_markers(self, index: int) -> list:
        """Where plug-ins measured on slice `index` (`PluginMarker`s)."""
        groups = [values for (i, _pair), values in self.plugin_values.items() if i == index]
        groups += [self.slice_plugin_values.get(index, ()), self.study_plugin_values]
        return [m for group in groups for values in group for m in values.markers if m.slice_index == index]

    def has_plugin_markers(self) -> bool:
        groups = [*self.plugin_values.values(), *self.slice_plugin_values.values(), self.study_plugin_values]
        return any(values.markers for group in groups for values in group)

    def results_by_pair(self, pair: str) -> dict[int, object]:
        return {index: result for (index, p), result in self.results.items() if p == pair}

    def is_physical(self) -> bool:
        """All results come from the physical recipe (the head-CT verdict applies)."""
        return bool(self.results) and all(
            getattr(result, "recipe", None) == RECIPE_PHYSICAL for result in self.results.values()
        )

    def progress_text(self, cancelled: bool = False) -> str:
        done = len(self.results) + len(self.errors)
        failed = f", {len(self.errors)} failed" if self.errors else ""
        if self.jobs is not None:
            return f"QA {done}/{self.total}{failed}"
        if self.total == 0:
            return ""
        if cancelled:
            return f"QA cancelled: {done}/{self.total} ROI pair(s){failed}"
        return f"QA done: {done} ROI pair(s){failed}"

    # ---- commands ----

    def run(self, jobs: list[Job]) -> None:
        """QA for every job (slice, pair) - one after another in the background."""
        if not jobs:
            return
        self.results, self.errors = {}, {}
        self._drop_plugin_values()
        self.jobs = list(jobs)
        self.total = len(jobs)
        self.cancel_requested = False
        logger.info(
            "Run All: QA on %d ROI pair(s) over slice(s) %s",
            len(jobs), format_slice_ranges(sorted({index for index, _pair in jobs})),
        )
        self.running_changed.emit(True)
        self.progress_changed.emit(self.progress_text())
        self._run_next()

    def cancel(self) -> None:
        """The running job can't be interrupted; no further one starts."""
        self.cancel_requested = True

    def set_recipe(self, recipe: str) -> bool:
        """Switches the recipe; True if earlier results were dropped (the two
        recipes' numbers are on different scales - never mixed)."""
        if recipe == self.recipe:
            return False
        self.recipe = recipe
        logger.info("QA recipe set to %s", recipe)
        if self.results or self.errors:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        """Drops every result and stops a running batch (a late result of the
        running job is ignored)."""
        if self.active_worker is not None:
            self.active_worker.signals.blockSignals(True)
            self.active_worker = None
        self.jobs = None
        self.active_job = None
        self.cancel_requested = False
        self.total = 0
        self.results, self.errors = {}, {}
        self._drop_plugin_values()
        self.progress_changed.emit("")
        self.reset_done.emit()

    def _drop_plugin_values(self) -> None:
        self.plugin_values, self.study_plugin_values, self._contexts = {}, (), {}
        self.slice_plugin_values, self.plugin_hints, self.plugin_tissue_method = {}, (), None
        self._study_job_started = False

    # ---- the batch ----

    def _run_next(self) -> None:
        if not self.jobs or self.cancel_requested:
            if not self.cancel_requested and self._needs_study_job():
                self._run_study_job()
            else:
                self._finish()
            return
        index, pair = self.jobs.pop(0)
        rois = self._rois(index)
        roi1, roi2 = rois[roi_key(pair, "roi1")], rois[roi_key(pair, "roi2")]
        volume = self._volume()
        pixels = volume.get_slice_hu(index)
        result = self._segmentation_cache().get(index)
        masks = None if result is None else result.masks
        spacing, recipe = volume.pixel_spacing_mm, self.recipe
        logger.info(
            "Run All: slice=%d (%s), roi1=%s, roi2=%s, pix_spc=%.4f",
            index, PAIR_LABELS[pair], roi1, roi2, spacing,
        )
        with_plugins = plugin_manager.current().has_measurement_hooks

        def job():
            # Intermediate steps kept for Results > Advanced (a few MB per study, no extra time).
            quality = pipeline.measure_pair(pixels, masks, roi1, roi2, spacing, recipe, keep_steps=True)
            if not with_plugins:
                return quality, None, ()
            context = pipeline.pair_context(index, pair, pixels, masks, roi1, roi2, spacing, quality)
            return quality, context, pipeline.plugin_pair_metrics(context)

        worker = Worker(job)
        # Bound methods (not lambdas) so Qt queues the call onto the GUI thread.
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.active_job = (index, pair)
        self.active_worker = worker  # keep a reference alive while it runs
        self._submit(worker)

    def _on_worker_finished(self, outcome) -> None:
        result, context, plugin_values = outcome
        index, pair = self.active_job
        logger.info(
            "Run All succeeded: slice=%d (%s), mtf_area=%.4f, predicted_label=%s, decision_value=%.4f",
            index, PAIR_LABELS[pair], result.mtf_area, result.predicted_label, result.decision_value,
        )
        self.active_worker = None
        self.results[(index, pair)] = result
        if context is not None:
            self._contexts[(index, pair)] = context
        if plugin_values:
            self.plugin_values[(index, pair)] = plugin_values
        self._job_ended((index, pair))

    def _on_worker_error(self, message: str) -> None:
        index, pair = self.active_job
        logger.error("Run All failed on slice %d (%s): %s", index, PAIR_LABELS[pair], message)
        self.active_worker = None
        self.errors[(index, pair)] = message
        self._job_ended((index, pair))

    def _job_ended(self, job: Job) -> None:
        self.job_done.emit(job)
        self.progress_changed.emit(self.progress_text())
        self._run_next()

    def _needs_study_job(self) -> bool:
        if self._study_job_started or not self._contexts:
            return False
        plugins = plugin_manager.current()
        return plugins.has_slice_metrics or plugins.has_study_metrics or plugins.has_artefact_hints

    def _run_study_job(self) -> None:
        """The plug-ins' slice and series values and artefact hints, in the background."""
        self._study_job_started = True
        volume = self._volume()
        masks = {index: result.masks for index, result in self._segmentation_cache().items()}
        contexts = tuple(self._contexts[job] for job in sorted(self._contexts))
        tissue = self._tissue_settings()  # snapshot: the user may change the method meanwhile
        worker = Worker(lambda: pipeline.plugin_study_phase(volume, masks, contexts, tissue))
        worker.signals.finished.connect(self._on_study_job_finished)
        worker.signals.error.connect(self._on_study_job_error)
        self.active_worker = worker
        self.progress_changed.emit("Plug-ins…")
        self._submit(worker)

    def _on_study_job_finished(self, outputs) -> None:
        self.active_worker = None
        self.slice_plugin_values = dict(outputs.slice_values)
        self.study_plugin_values = tuple(outputs.study_values)
        self.plugin_hints = tuple(outputs.hints)
        self.plugin_tissue_method = outputs.tissue_method
        self._finish()

    def _on_study_job_error(self, message: str) -> None:
        logger.error("Plug-in series values failed: %s", message)
        self.active_worker = None
        self._finish()

    def _finish(self) -> None:
        cancelled = self.cancel_requested
        self.jobs = None
        self.active_job = None
        self.cancel_requested = False
        self.progress_changed.emit(self.progress_text(cancelled=cancelled))
        self.running_changed.emit(False)
        self.finished.emit(cancelled)
