"""Dispatches `Worker` jobs on a shared thread pool."""

from __future__ import annotations

from PySide6.QtCore import QRunnable, QThreadPool


class Scheduler:
    """Submits `QRunnable` jobs (e.g. `Worker`) to a `QThreadPool`.

    Deliberately minimal and generic: this project has exactly one Core API
    call to dispatch today (`evaluate_slice_quality`, via `Worker`) and no
    queueing/priority/retry requirements yet, so `Scheduler` only knows about
    `QRunnable`, not about `Worker`'s constructor or signals.
    """

    def __init__(self) -> None:
        self.pool = QThreadPool.globalInstance()

    def submit(self, job: QRunnable) -> None:
        """Starts `job` on the thread pool.

        Connect to the job's signals before calling this - it may start
        running, and emitting, on another thread immediately.
        """
        self.pool.start(job)
