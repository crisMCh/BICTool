"""Runs a single `evaluate_slice_quality` call off the GUI thread."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from bictool.domain.core_api import evaluate_slice_quality
from bictool.integration.model_files import default_chest_svm_model

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
        mtf_seeds: Sequence[tuple[int, int]] | None = None,
        recipe: str | None = None,
        svm_model=None,
    ) -> None:
        super().__init__()
        self._img_roi1 = img_roi1
        self._img_roi2 = img_roi2
        self._pix_spc = pix_spc
        self._act_cont_mask_x = act_cont_mask_x
        self._act_cont_mask_y = act_cont_mask_y
        self._mtf_seeds = mtf_seeds
        self._recipe = recipe
        self._svm_model = svm_model  # None: the shipped chest-CT SVM (legacy per-slice verdict)
        self.signals = WorkerSignals()

    def run(self) -> None:
        logger.info(
            "Worker starting: roi1 shape=%s, roi2 shape=%s, pix_spc=%s, act_cont_mask=(%s, %s), mtf_seeds=%s",
            self._img_roi1.shape,
            self._img_roi2.shape,
            self._pix_spc,
            self._act_cont_mask_x,
            self._act_cont_mask_y,
            self._mtf_seeds,
        )
        try:
            result = evaluate_slice_quality(
                self._img_roi1,
                self._img_roi2,
                self._pix_spc,
                self._act_cont_mask_x,
                self._act_cont_mask_y,
                svm_model=self._svm_model if self._svm_model is not None else default_chest_svm_model(),
                mtf_seeds=self._mtf_seeds,
                **({"recipe": self._recipe} if self._recipe else {}),
            )
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("Worker failed: evaluate_slice_quality raised %s", type(exc).__name__)
            self.signals.error.emit(str(exc))
            return
        logger.info("Worker finished successfully.")
        self.signals.finished.emit(result)
