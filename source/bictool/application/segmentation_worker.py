"""Runs a single anatomy `segment_slice` call off the GUI thread.

A Keras `model.predict()` call on a 512x512 U-Net is real, non-trivial CPU
time - must not run on the Qt GUI thread, same reasoning as `Worker` for
`evaluate_slice_quality`.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from bictool.domain.segmentation.head_ct_unet import segment_slice

logger = logging.getLogger(__name__)


class SegmentationWorkerSignals(QObject):
    """Signals emitted by a `SegmentationWorker`.

    A separate `QObject` is needed because `QRunnable` itself cannot emit signals.
    """

    finished = Signal(object)  # SegmentationResult
    error = Signal(str)


class SegmentationWorker(QRunnable):
    """Runs `segment_slice` on a `QThreadPool` thread.

    Wraps the one Domain-layer call this does; not a general job-execution
    abstraction (see `Scheduler` for dispatch). Connect to `signals` before
    handing this to `Scheduler.submit` - the job may start running, and
    emitting, on another thread immediately. `model` must already have
    weights loaded (see `head_ct_unet.load_weights`) - loading isn't this
    worker's job, so the same loaded model can be reused across slices
    without reloading weights each time.
    """

    def __init__(self, model, slice_hu: np.ndarray) -> None:
        super().__init__()
        self._model = model
        self._slice_hu = slice_hu
        self.signals = SegmentationWorkerSignals()

    def run(self) -> None:
        logger.info("SegmentationWorker starting: slice shape=%s", self._slice_hu.shape)
        try:
            result = segment_slice(self._model, self._slice_hu)
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("SegmentationWorker failed: segment_slice raised %s", type(exc).__name__)
            self.signals.error.emit(str(exc))
            return
        logger.info("SegmentationWorker finished successfully.")
        self.signals.finished.emit(result)
