"""Runs DICOM series discovery off the GUI thread.

A recursive folder scan over a real DICOM export - potentially hundreds of
files, each needing at least a header read to sort/group (see
`integration.dicom_file_loader`'s module docstring) - is real disk I/O and
must not run on the Qt GUI thread, or the window becomes unresponsive
("Not Responding") for however long the scan takes. Mirrors `Worker`'s
pattern for `evaluate_slice_quality`.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

from bictool.integration.dicom_file_loader import discover

logger = logging.getLogger(__name__)


class DicomLoadWorkerSignals(QObject):
    """Signals emitted by a `DicomLoadWorker`.

    A separate `QObject` is needed because `QRunnable` itself cannot emit signals.
    """

    finished = Signal(object)  # list[SeriesCandidate]
    error = Signal(str)
    progress = Signal(int)  # files scanned so far


class DicomLoadWorker(QRunnable):
    """Runs `discover(path)` on a `QThreadPool` thread.

    Emits the full list of discovered `SeriesCandidate`s (never just one) -
    the caller decides which to actually load (see `main_window`'s series
    selector); this worker's job is only the expensive disk scan. Connect
    to `signals` before handing this to `Scheduler.submit` - the job may
    start running, and emitting, on another thread immediately.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        self.signals = DicomLoadWorkerSignals()

    def run(self) -> None:
        logger.info("DicomLoadWorker starting: path=%s", self._path)
        try:
            candidates = discover(self._path, on_progress=self.signals.progress.emit)
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("DicomLoadWorker failed: discover raised %s", type(exc).__name__)
            self.signals.error.emit(str(exc))
            return
        logger.info("DicomLoadWorker finished: %d series candidate(s) found", len(candidates))
        self.signals.finished.emit(candidates)
