"""Runs a per-slice WM/GM (Brain Tissue) classification across many slices
off the GUI thread.

The Brain Tissue overlay classifies only the displayed slice, synchronously
(milliseconds for HU/adaptive thresholding). Auto ROI2 needs white matter
on every slice with an ROI1, and Fuzzy C-Means - the default method - costs
about a second per slice, so a whole volume would freeze the GUI. Same shape
as `VolumeSegmentationWorker`: one `slice_finished` per slice so results can
be used as they land, cooperative cancellation between slices.

`classify(slice_index)` does the actual work and must be thread-safe - the
caller snapshots the method/settings/brain masks into it before submitting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class BrainTissueVolumeWorkerSignals(QObject):
    slice_finished = Signal(int, object)  # slice_index, BrainTissueResult
    finished = Signal()
    cancelled = Signal()
    error = Signal(int, str)  # slice_index, message


class BrainTissueVolumeWorker(QRunnable):
    def __init__(self, classify: Callable[[int], object], slice_indices: list[int]) -> None:
        super().__init__()
        self._classify = classify
        self._slice_indices = slice_indices
        self._cancel_requested = False
        self.signals = BrainTissueVolumeWorkerSignals()

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        logger.info("BrainTissueVolumeWorker starting: %d slice(s)", len(self._slice_indices))
        for slice_index in self._slice_indices:
            if self._cancel_requested:
                logger.info("BrainTissueVolumeWorker cancelled")
                self.signals.cancelled.emit()
                return
            try:
                result = self._classify(slice_index)
            except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
                logger.exception("BrainTissueVolumeWorker failed on slice %d", slice_index)
                self.signals.error.emit(slice_index, str(exc))
                return
            self.signals.slice_finished.emit(slice_index, result)
        logger.info("BrainTissueVolumeWorker finished")
        self.signals.finished.emit()
