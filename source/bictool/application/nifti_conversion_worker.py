"""Runs `dicom_to_nifti.build_dicom_nifti_volume` off the GUI thread.

A real `dicom2nifti` conversion (temp-directory file copy + the actual
DICOM->NIfTI decode) measured a few seconds for a real ~40-slice head CT -
not the multi-minute cost segmentation itself can be, but real, non-instant
disk/CPU work that must not block the GUI thread, same reasoning as every
other worker in this package. Runs once per loaded study (the result is
cached by the caller - see `main_window._ensure_nifti_volume_then`), not
once per segmentation click.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from bictool.domain.segmentation._nifti_alignment import DicomNiftiVolume
from bictool.integration.dicom_to_nifti import build_dicom_nifti_volume

logger = logging.getLogger(__name__)


class NiftiConversionWorkerSignals(QObject):
    """Signals emitted by a `NiftiConversionWorker`.

    A separate `QObject` is needed because `QRunnable` itself cannot emit signals.
    """

    finished = Signal(object)  # DicomNiftiVolume
    error = Signal(str)


class NiftiConversionWorker(QRunnable):
    """Runs `build_dicom_nifti_volume` on a `QThreadPool` thread.

    Connect to `signals` before handing this to `Scheduler.submit` - the job
    may start running, and emitting, on another thread immediately.
    """

    def __init__(self, slice_paths: list[Path], get_slice_hu) -> None:
        super().__init__()
        self._slice_paths = slice_paths
        self._get_slice_hu = get_slice_hu
        self.signals = NiftiConversionWorkerSignals()

    def run(self) -> None:
        logger.info("NiftiConversionWorker starting: %d slice(s)", len(self._slice_paths))
        try:
            result: DicomNiftiVolume = build_dicom_nifti_volume(self._slice_paths, self._get_slice_hu)
        except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
            logger.exception("NiftiConversionWorker failed: %s", type(exc).__name__)
            self.signals.error.emit(str(exc))
            return
        logger.info("NiftiConversionWorker finished successfully.")
        self.signals.finished.emit(result)
