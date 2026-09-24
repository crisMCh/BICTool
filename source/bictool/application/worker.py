"""Runs a single `evaluate_slice_quality` call off the GUI thread."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from bictool.domain.core_api import evaluate_slice_quality

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals emitted by a `Worker`.

    A separate `QObject` is needed because `QRunnable` itself cannot emit signals.
    """

    finished = Signal(object)  # SliceQualityResult
    error = Signal(str)


class Worker(QRunnable):
    """Runs `evaluate_slice_quality` on a `QThreadPool` thread.

    Wraps the one Core API call the Presentation layer currently has to make;
    not a general job-execution abstraction (see `Scheduler` for dispatch).
    Connect to `signals` before handing this to `Scheduler.submit` - the job
    may start running, and emitting, on another thread immediately.
    """

    def __init__(
        self,
        img_roi1: np.ndarray,
        img_roi2: np.ndarray,
        pix_spc: float,
        act_cont_mask_x: int = 10,
        act_cont_mask_y: int = 10,
    ) -> None:
        super().__init__()
        self._img_roi1 = img_roi1
        self._img_roi2 = img_roi2
        self._pix_spc = pix_spc
        self._act_cont_mask_x = act_cont_mask_x
        self._act_cont_mask_y = act_cont_mask_y
        self.signals = WorkerSignals()

    def run(self) -> None:
        logger.info(
            "Worker starting: roi1 shape=%s, roi2 shape=%s, pix_spc=%s, act_cont_mask=(%s, %s)",
            self._img_roi1.shape,
            self._img_roi2.shape,
            self._pix_spc,
            self._act_cont_mask_x,
            self._act_cont_mask_y,
        )
        try:
            result = evaluate_slice_quality(
                self._img_roi1,
                self._img_roi2,
                self._pix_spc,
                self._act_cont_mask_x,
                self._act_cont_mask_y,
            )
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("Worker failed: evaluate_slice_quality raised %s", type(exc).__name__)
            self.signals.error.emit(str(exc))
            return
        logger.info("Worker finished successfully.")
        self.signals.finished.emit(result)
