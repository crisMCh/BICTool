"""Runs one pipeline step off the GUI thread - e.g. one ROI pair's
measurement (`api.steps.measure_pair`)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals emitted by a `Worker`.

    A separate `QObject` is needed because `QRunnable` itself cannot emit signals.
    """

    finished = Signal(object)  # what `job()` returned
    error = Signal(str)


class Worker(QRunnable):
    """Runs `job()` on a `QThreadPool` thread and reports its result or error.

    `job` must be a closure over a snapshot of its inputs (the caller builds
    it before submitting, so later changes in the GUI can't race with it).
    Connect to `signals` before handing this to `Scheduler.submit` - the job
    may start running, and emitting, on another thread immediately.
    """

    def __init__(self, job: Callable[[], object]) -> None:
        super().__init__()
        self._job = job
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self._job()
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("Worker failed: %s", type(exc).__name__)
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)
