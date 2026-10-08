"""Runs a single stroke `segment_slice_for_stroke` call off the GUI thread.

Same reasoning as `Worker`/`SegmentationWorker`: a Keras `model.predict()`
call is real CPU time and must not run on the Qt GUI thread.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from bictool.domain.segmentation.aisd_stroke_unet import segment_slice_for_stroke

logger = logging.getLogger(__name__)


class StrokeWorkerSignals(QObject):
    """Signals emitted by a `StrokeWorker`.

    A separate `QObject` is needed because `QRunnable` itself cannot emit signals.
    """

    finished = Signal(object)  # StrokeSegmentationResult
    error = Signal(str)


class StrokeWorker(QRunnable):
    """Runs `segment_slice_for_stroke` on a `QThreadPool` thread.

    Wraps the one Domain-layer call this does; not a general job-execution
    abstraction (see `Scheduler` for dispatch). Connect to `signals` before
    handing this to `Scheduler.submit` - the job may start running, and
    emitting, on another thread immediately. `model` must already have
    weights loaded (see `integration.model_files.load_stroke_weights`).
    """

    def __init__(self, model, slice_hu: np.ndarray) -> None:
        super().__init__()
        self._model = model
        self._slice_hu = slice_hu
        self.signals = StrokeWorkerSignals()

    def run(self) -> None:
        logger.info("StrokeWorker starting: slice shape=%s", self._slice_hu.shape)
        try:
            result = segment_slice_for_stroke(self._model, self._slice_hu)
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("StrokeWorker failed: segment_slice_for_stroke raised %s", type(exc).__name__)
            self.signals.error.emit(str(exc))
            return
        logger.info("StrokeWorker finished successfully.")
        self.signals.finished.emit(result)
