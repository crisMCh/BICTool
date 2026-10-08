"""RoiViewModel: the falx and septum ROI pairs of every slice.

Owns the ROIs (`RoiPairStore`: automatic ROIs, manual ones that win over
them, deletions), whether automatic ROIs are on, whether the ROIs are
confirmed (then nothing can be edited), and the automatic placement itself:
ROI1 on the structure (`structured_roi`) for every eligible segmented slice
(`slice_eligibility`), ROI2 next to it (`homogeneous_roi`) - for the falx in
white matter, which a background WM/GM classification provides.

The view draws `effective(slice)` and binds to the signals. QtCore only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from bictool.application.brain_tissue_volume_worker import BrainTissueVolumeWorker
from bictool.domain.roi_selection.homogeneous_roi import FALLBACK, VENTRICLE, WHITE_MATTER, find_roi2_candidate
from bictool.domain.roi_selection.slice_eligibility import MIN_BRAIN_FRACTION, eligible_slices
from bictool.domain.roi_selection.structured_roi import (
    DURAL_FOLDS,
    SEPTUM_PELLUCIDUM,
    RoiCandidate,
    estimate_midline_direction,
    find_roi1_candidate,
)
from bictool.domain.segmentation.segmentation_brain_mask import brain_mask_from_segmentation
from bictool.presentation.viewmodels.display_text import format_slice_ranges
from bictool.presentation.viewmodels.roi_pairs import FALX, PAIR_LABELS, PAIRS, ROLE_LABELS, SEPTUM, RoiPairStore, roi_key, split_key
from bictool.presentation.viewmodels.roi_selection import RoiRect, candidate_to_rect

logger = logging.getLogger(__name__)

# ROI pair -> the segmentation structure its ROI1 sits on.
PAIR_STRUCTURES = {FALX: DURAL_FOLDS, SEPTUM: SEPTUM_PELLUCIDUM}
ROI2_REGION_LABELS = {
    VENTRICLE: "ventricle",
    WHITE_MATTER: "white matter",
    FALLBACK: "fallback - no clean homogeneous area nearby",
}


class RoiViewModel(QObject):
    rois_changed = Signal()             # some slice's ROIs changed - redraw the shown slice
    auto_status_changed = Signal(str)   # where the automatic ROIs are (for a tooltip)
    tissue_progress_changed = Signal(str)  # status-bar text of the background WM/GM job
    confirmed_changed = Signal(bool)

    def __init__(
        self,
        submit: Callable[[object], None],
        volume: Callable[[], object],
        current_slice: Callable[[], int],
        segmentation_cache: Callable[[], dict],
        tissue_cache: Callable[[], dict],
        tissue_classifier: Callable[[], Callable],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._volume = volume
        self._current_slice = current_slice
        self._segmentation_cache = segmentation_cache
        self._tissue_cache = tissue_cache
        self._tissue_classifier = tissue_classifier
        self.store = RoiPairStore()
        self.auto_on = False
        self.confirmed = False
        self.midline = None  # volume-wide midline direction (structured_roi.estimate_midline_direction)
        self.eligible: set[int] = set()  # segmented slices with enough brain (slice_eligibility)
        self.roi2_regions: dict[str, dict[int, str]] = {pair: {} for pair in PAIRS}
        self.active_tissue_worker: BrainTissueVolumeWorker | None = None
        self.status_text = ""
        self._tissue_done = self._tissue_total = 0

    # ---- queries ----

    def effective(self, index: int) -> dict[str, RoiRect]:
        """A slice's ROIs per slot: manual over automatic, deleted ones left out."""
        return self.store.effective(index)

    def ready_jobs(self) -> list[tuple[int, str]]:
        """Every (slice, pair) with both ROIs - what one confirmation + QA run covers."""
        return self.store.ready_jobs()

    def has_edits(self, index: int) -> bool:
        return self.store.has_edits(index)

    def auto_roi1_slices(self, pair: str) -> list[int]:
        key1 = roi_key(pair, "roi1")
        return [index for index in self.store.slices() if self.store.auto(index, key1) is not None]

    # ---- automatic ROIs ----

    def set_auto(self, on: bool) -> None:
        """Start (place both pairs on every eligible segmented slice) or remove
        the automatic ROIs (manual ones stay)."""
        self.auto_on = on
        if on:
            self.recompute()
        else:
            self.clear_auto()
        self.rois_changed.emit()

    def recompute(self) -> None:
        """ROI1 (and ROI2 where possible right away) of both pairs on every
        segmented slice; white-matter ROI2s follow from the background WM/GM job."""
        self._stop_tissue_worker()
        self.store.clear_auto()
        self.roi2_regions = {pair: {} for pair in PAIRS}
        self.midline = estimate_midline_direction(self._masks_by_slice())
        self._refresh_eligible()
        for pair in PAIRS:
            for index in self._segmentation_cache():
                self._compute_roi1(index, pair)
        logger.info(
            "Auto ROIs: %s (of %d segmented slice(s))",
            "; ".join(f"{PAIR_LABELS[pair]} ROI1 on {len(self.auto_roi1_slices(pair))}" for pair in PAIRS),
            len(self._segmentation_cache()),
        )
        self._start_tissue_worker()
        self.update_status()

    def on_slice_segmented(self, index: int) -> None:
        """A slice got segmented: while auto ROIs are on, it gets its ROIs now
        (and others may change eligibility - a larger slice can make them too small)."""
        if not self.auto_on:
            return
        self.midline = estimate_midline_direction(self._masks_by_slice())
        changed = self._refresh_eligible() | {index}
        for slice_index in sorted(changed & set(self._segmentation_cache())):
            for pair in PAIRS:
                self._compute_roi1(slice_index, pair)
        self._start_tissue_worker()
        self.update_status()
        self.rois_changed.emit()

    def restart_roi2(self) -> None:
        """The WM/GM method or its settings changed: white-matter ROI2s are
        stale. Ventricle ROI2s and manual ones stay; confirmed ROIs are locked."""
        if not self.auto_on or self.confirmed:
            return
        self._stop_tissue_worker()
        for index in self.store.slices():
            for pair in PAIRS:
                self._compute_roi2(index, pair)
        self._start_tissue_worker()
        self.update_status()
        self.rois_changed.emit()

    def clear_auto(self) -> None:
        self._stop_tissue_worker()
        self.store.clear_auto()
        self.eligible = set()
        self.roi2_regions = {pair: {} for pair in PAIRS}
        self._set_status("")

    # ---- manual edits (refused while confirmed) ----

    def save_manual(self, index: int, key: str, rect: RoiRect) -> None:
        """`rect` becomes slot `key`'s ROI on slice `index` (a drawn rectangle
        saved as, or an ROI moved). Saving/moving a ROI1 re-places that pair's
        automatic ROI2 around it (unless that ROI2 is manual too)."""
        if self.confirmed:
            return
        self.store.set_manual(index, key, rect)
        pair, slot = split_key(key)
        if slot == "roi1":
            self._compute_roi2(index, pair)
        self.update_status()
        self.rois_changed.emit()

    def reassign(self, index: int, source: str, target: str) -> None:
        """An existing ROI saved as another role: it moves to slot `target`
        (replacing what was there); slot `source` is left empty."""
        rect = self.store.effective(index).get(source)
        if self.confirmed or rect is None or source == target:
            return
        logger.info("Slice %d: %s saved as %s", index + 1, ROLE_LABELS[source], ROLE_LABELS[target])
        self.store.set_manual(index, target, rect)
        self.store.delete(index, source)
        pair, slot = split_key(target)
        if slot == "roi1":
            self._compute_roi2(index, pair)
        self.update_status()
        self.rois_changed.emit()

    def delete(self, index: int, key: str) -> None:
        """The slot shows nothing on this slice any more (manual and automatic)
        until a new ROI is saved into it or the slice is restored."""
        if self.confirmed:
            return
        logger.info("%s deleted on slice %d", ROLE_LABELS[key], index + 1)
        self.store.delete(index, key)
        self.update_status()
        self.rois_changed.emit()

    def restore_slice(self, index: int) -> None:
        """Back to the automatic ROIs on this slice: manual ROIs and deletions dropped."""
        if self.confirmed:
            return
        logger.info("Slice %d: automatic ROIs restored", index + 1)
        self.store.restore_slice(index)
        if self.auto_on:
            for pair in PAIRS:
                self._compute_roi1(index, pair)
            self._start_tissue_worker()
        self.update_status()
        self.rois_changed.emit()

    def set_confirmed(self, confirmed: bool) -> None:
        """Confirmed = the ROIs on every slice are final (locked) and QA may run."""
        self.confirmed = confirmed
        self.confirmed_changed.emit(confirmed)

    def reset(self) -> None:
        """A new/closed series: no ROIs, auto off, not confirmed."""
        self.clear_auto()
        self.store.clear()
        self.auto_on = False
        self.confirmed = False

    # ---- status (where the automatic ROIs are) ----

    def update_status(self) -> None:
        """One line per pair for the shown slice, plus the skipped slices."""
        if not self.auto_on:
            return
        index = self._current_slice()
        lines = []
        for pair in PAIRS:
            name = PAIR_LABELS[pair]
            slices = self.auto_roi1_slices(pair)
            if not slices:
                lines.append(f"{name}: no ROI1 on any segmented slice.")
                continue
            if self.store.auto(index, roi_key(pair, "roi1")) is None:
                here = "not on this slice"
            elif self.store.auto(index, roi_key(pair, "roi2")) is None:
                here = "ROI2 waiting for WM/GM…"
            else:
                here = f"ROI2 in {ROI2_REGION_LABELS[self.roi2_regions[pair].get(index, VENTRICLE)]}"
            waiting = sum(1 for i in slices if self.store.auto(i, roi_key(pair, "roi2")) is None)
            waiting_text = f", {waiting} waiting for WM/GM" if waiting else ""
            lines.append(f"{name}: {here}. ROI1 on {len(slices)} slice(s): {format_slice_ranges(slices)}{waiting_text}.")
        skipped = sorted(set(self._segmentation_cache()) - self.eligible)
        if skipped:
            lines.append(
                f"No automatic ROIs on {len(skipped)} slice(s) with < {MIN_BRAIN_FRACTION:.0%} of the largest "
                f"brain area: {format_slice_ranges(skipped)}."
            )
        self._set_status("\n".join(lines))

    def _set_status(self, text: str) -> None:
        self.status_text = text
        self.auto_status_changed.emit(text)

    # ---- placement ----

    def _masks_by_slice(self) -> dict:
        return {index: result.masks for index, result in self._segmentation_cache().items()}

    def _refresh_eligible(self) -> set[int]:
        """Recomputes the eligible slices; returns those whose eligibility changed."""
        eligible = eligible_slices(self._masks_by_slice())
        changed = eligible ^ self.eligible
        self.eligible = eligible
        return changed

    def _compute_roi1(self, index: int, pair: str) -> None:
        result = self._segmentation_cache().get(index)
        if result is None:
            return
        candidate = None
        if index in self.eligible:  # too little brain: no automatic ROI (manual ones allowed)
            volume = self._volume()
            candidate = find_roi1_candidate(
                result.masks,
                PAIR_STRUCTURES[pair],
                volume.pixel_spacing_mm,
                midline_direction=self.midline,
                image=volume.get_slice_hu(index),
            )
        self.store.set_auto(index, roi_key(pair, "roi1"), None if candidate is None else candidate_to_rect(candidate))
        self._compute_roi2(index, pair)

    def _compute_roi2(self, index: int, pair: str) -> None:
        """The pair's automatic ROI2 next to its ROI1 as shown (manual or
        automatic). Pending (absent) while it needs white matter that the
        background WM/GM job hasn't delivered yet. A manual ROI2 is left alone."""
        key2 = roi_key(pair, "roi2")
        segmentation = self._segmentation_cache()
        if not self.auto_on or index not in segmentation or self.store.is_manual(index, key2):
            return
        roi1 = self.store.effective(index).get(roi_key(pair, "roi1"))
        candidate = None
        if roi1 is not None:
            tissue = self._tissue_cache().get(index)
            candidate = find_roi2_candidate(
                RoiCandidate(roi1.y0, roi1.x0, roi1.width, PAIR_STRUCTURES[pair]),
                segmentation[index].masks,
                self._volume().pixel_spacing_mm,
                None if tissue is None else tissue.masks,
            )
        if candidate is None:
            self.store.set_auto(index, key2, None)
            self.roi2_regions[pair].pop(index, None)
        else:
            self.store.set_auto(index, key2, candidate_to_rect(candidate))
            self.roi2_regions[pair][index] = candidate.structure

    def _waiting_for_tissue(self, index: int) -> bool:
        """Some pair on this slice has its ROI1 but its automatic ROI2 is still pending."""
        rois = self.store.effective(index)
        return any(
            roi_key(pair, "roi1") in rois
            and self.store.auto(index, roi_key(pair, "roi2")) is None
            and not self.store.is_manual(index, roi_key(pair, "roi2"))
            for pair in PAIRS
        )

    # ---- background WM/GM classification for white-matter ROI2s ----

    def _start_tissue_worker(self) -> None:
        """Classifies WM/GM for every slice whose ROI2 waits on white matter. If
        one job already runs, its `finished` starts the next for slices that
        became pending meanwhile."""
        volume = self._volume()
        if self.active_tissue_worker is not None or volume is None:
            return
        tissue = self._tissue_cache()
        pending = [i for i in sorted(self._segmentation_cache()) if i not in tissue and self._waiting_for_tissue(i)]
        if not pending:
            return
        self._tissue_done, self._tissue_total = 0, len(pending)
        self.tissue_progress_changed.emit(f"Placing ROIs (WM/GM) 0/{len(pending)}")
        classify = self._tissue_classifier()
        brain_masks = brain_mask_from_segmentation(self._masks_by_slice(), volume.num_slices)
        get_slice_hu = volume.get_slice_hu
        worker = BrainTissueVolumeWorker(lambda i: classify(get_slice_hu(i), brain_masks[i]), pending)
        worker.signals.slice_finished.connect(self._on_tissue_slice_finished)
        worker.signals.finished.connect(self._on_tissue_finished)
        worker.signals.error.connect(self._on_tissue_error)
        self.active_tissue_worker = worker  # keep a reference alive while it runs
        logger.info("Auto ROIs: classifying WM/GM for %d slice(s) in the background", len(pending))
        self._submit(worker)

    def _on_tissue_slice_finished(self, index: int, result) -> None:
        self._tissue_cache().setdefault(index, result)
        self._tissue_done += 1
        self.tissue_progress_changed.emit(f"Placing ROIs (WM/GM) {self._tissue_done}/{self._tissue_total}")
        if not self.auto_on:
            return
        for pair in PAIRS:
            self._compute_roi2(index, pair)
        self.update_status()
        self.rois_changed.emit()

    def _on_tissue_finished(self) -> None:
        self.active_tissue_worker = None
        self.tissue_progress_changed.emit("placed" if self.auto_on else "")
        if self.auto_on:
            self._start_tissue_worker()

    def _on_tissue_error(self, index: int, message: str) -> None:
        logger.error("Auto ROIs: WM/GM classification failed on slice %d: %s", index + 1, message)
        self.active_tissue_worker = None
        self.tissue_progress_changed.emit(f"WM/GM failed on slice {index + 1}")
        self._set_status(f"WM/GM classification failed on slice {index + 1}: {message}")

    def _stop_tissue_worker(self) -> None:
        # cancel() only stops future slices; blockSignals silences the one running.
        self.tissue_progress_changed.emit("")
        worker = self.active_tissue_worker
        if worker is not None:
            worker.cancel()
            worker.signals.blockSignals(True)
            self.active_tissue_worker = None
