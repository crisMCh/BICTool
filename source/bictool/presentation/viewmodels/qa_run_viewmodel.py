"""QaRunViewModel: the image-quality measurement over the confirmed ROI pairs.

Owns the QA recipe (physical by default; the legacy Octave one under
Tools > Advanced), the job queue - one job per (slice, ROI pair) - the job
running in the background (`application.worker.Worker`, one at a time),
the results and errors per job, and cancelling (the running job finishes,
no further one starts). For each job it crops the two ROIs from the slice
and seeds the MTF contour on the structure (segmentation as a hint).
QtCore only - no widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from bictool.application.worker import Worker
from bictool.domain.metrics.mtf_seed import mtf_seed_candidates
from bictool.domain.results.study_quality import DEFAULT_RECIPE, RECIPE_PHYSICAL
from bictool.domain.roi_selection.structured_roi import STRUCTURES
from bictool.presentation.viewmodels.display_text import format_slice_ranges
from bictool.presentation.viewmodels.roi_pairs import PAIR_LABELS, roi_key
from bictool.presentation.viewmodels.roi_selection import RoiRect, crop_roi

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
    ) -> None:
        """`rois(slice)` = the slice's effective ROIs per slot; `segmentation_cache()`
        = the anatomy results (MTF seed hint)."""
        super().__init__(parent)
        self._submit = submit
        self._volume = volume
        self._rois = rois
        self._segmentation_cache = segmentation_cache
        self.recipe = DEFAULT_RECIPE
        self.jobs: list[Job] | None = None  # pending jobs while a batch runs, else None
        self.total = 0
        self.active_job: Job | None = None
        self.cancel_requested = False
        self.active_worker: Worker | None = None
        self.results: dict[Job, object] = {}  # (slice, pair) -> SliceQualityResult
        self.errors: dict[Job, str] = {}

    # ---- queries ----

    def is_running(self) -> bool:
        return self.jobs is not None

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
        self.progress_changed.emit("")
        self.reset_done.emit()

    # ---- the batch ----

    def mtf_structure_mask(self, index: int, rect: RoiRect) -> np.ndarray | None:
        """The ROI1 structures (septum + dural folds) cropped to `rect`, or None
        if the slice isn't segmented - places the MTF seed and checks that the
        measured edge lies on the structure."""
        result = self._segmentation_cache().get(index)
        if result is None:
            return None
        masks = [result.masks[name] for name in STRUCTURES if name in result.masks]
        return crop_roi(np.logical_or.reduce(masks), rect) if masks else None

    def _run_next(self) -> None:
        if not self.jobs or self.cancel_requested:
            self._finish()
            return
        index, pair = self.jobs.pop(0)
        rois = self._rois(index)
        roi1, roi2 = rois[roi_key(pair, "roi1")], rois[roi_key(pair, "roi2")]
        volume = self._volume()
        pixels = volume.get_slice_hu(index)
        img_roi1, img_roi2 = crop_roi(pixels, roi1), crop_roi(pixels, roi2)
        # calc_mtf's active contour must start inside the edge structure: seed
        # it on the structure's centerline, with fallbacks if a seed finds no edges.
        structure = self.mtf_structure_mask(index, roi1)
        seeds = mtf_seed_candidates(img_roi1, structure)
        logger.info(
            "Run All: slice=%d (%s), roi1 shape=%s, roi2 shape=%s, pix_spc=%.4f, mtf seeds=%s",
            index, PAIR_LABELS[pair], img_roi1.shape, img_roi2.shape, volume.pixel_spacing_mm, seeds,
        )
        # Intermediate steps kept for Results > Advanced (a few MB per study, no extra time).
        worker = Worker(
            img_roi1, img_roi2, volume.pixel_spacing_mm, mtf_seeds=seeds, recipe=self.recipe, keep_steps=True,
            mtf_structure=structure,
        )
        # Bound methods (not lambdas) so Qt queues the call onto the GUI thread.
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.active_job = (index, pair)
        self.active_worker = worker  # keep a reference alive while it runs
        self._submit(worker)

    def _on_worker_finished(self, result) -> None:
        index, pair = self.active_job
        logger.info(
            "Run All succeeded: slice=%d (%s), mtf_area=%.4f, predicted_label=%s, decision_value=%.4f",
            index, PAIR_LABELS[pair], result.mtf_area, result.predicted_label, result.decision_value,
        )
        self.active_worker = None
        self.results[(index, pair)] = result
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

    def _finish(self) -> None:
        cancelled = self.cancel_requested
        self.jobs = None
        self.active_job = None
        self.cancel_requested = False
        self.progress_changed.emit(self.progress_text(cancelled=cancelled))
        self.running_changed.emit(False)
        self.finished.emit(cancelled)
