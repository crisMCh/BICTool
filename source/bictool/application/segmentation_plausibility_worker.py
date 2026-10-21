"""Runs `segmentation_plausibility.assess_plausibility` off the GUI thread (about 1-4 s per study).

`assess()` must be a closure over a snapshot of the inputs (the label masks, the pixel spacing) - the
caller builds it before submitting, so later changes in the GUI can't race with the job.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class SegmentationPlausibilityWorkerSignals(QObject):
    finished = Signal(object)  # SegmentationPlausibility
    error = Signal(str)


class SegmentationPlausibilityWorker(QRunnable):
    def __init__(self, assess: Callable[[], object]) -> None:
        super().__init__()
        self._assess = assess
        self.signals = SegmentationPlausibilityWorkerSignals()

    def run(self) -> None:
        try:
            result = self._assess()
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("SegmentationPlausibilityWorker failed")
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)
