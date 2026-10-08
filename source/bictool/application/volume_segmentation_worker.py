"""Runs `segment_slice` across every not-yet-cached slice of a volume, off
the GUI thread - the batch counterpart to `SegmentationWorker`.

Each individual `segment_slice` call costs the same ~1-2s on CPU as
`SegmentationWorker`'s single-slice call (see `_unet_architecture.build_unet`'s
docstring for why - XLA JIT-compiled, not the ~157s an uncompiled forward
pass measured), so a full volume is a few minutes, not hours - still long
enough to be worth backgrounding with progress rather than blocking. This
reports progress per slice (`slice_finished`) rather than only a single
final result, so the GUI can cache and display each slice's result as soon
as it's ready instead of waiting for the whole volume, and supports
cooperative cancellation - checked between slices, not mid-inference, since
a single `model.predict()` call can't be interrupted.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from bictool.domain.segmentation.head_ct_unet import segment_slice

logger = logging.getLogger(__name__)


class VolumeSegmentationWorkerSignals(QObject):
    """Signals emitted by a `VolumeSegmentationWorker`.

    `slice_finished` fires once per completed slice; `progress` reports how
    many of the requested slices are done so far. Exactly one of
    `finished`/`cancelled`/`error` fires once, at the end.
    """

    slice_finished = Signal(int, object)  # slice_index, SegmentationResult
    progress = Signal(int, int)  # completed_count, total_count
    finished = Signal()
    cancelled = Signal(int, int)  # completed_count, total_count
    error = Signal(int, str)  # slice_index, message


class VolumeSegmentationWorker(QRunnable):
    """Runs `segment_slice` for each index in `slice_indices` on a
    `QThreadPool` thread. `model` must already have weights loaded (see
    `integration.model_files.load_anatomy_weights`) - loading isn't this worker's job, matching
    `SegmentationWorker`'s own contract.

    Call `cancel()` from the GUI thread to request a stop before the next
    slice starts; already-completed slices' results have already been
    emitted via `slice_finished` and are not lost.
    """

    def __init__(self, model, get_slice_hu, slice_indices: list[int]) -> None:
        super().__init__()
        self._model = model
        self._get_slice_hu = get_slice_hu
        self._slice_indices = slice_indices
        self._cancel_requested = False
        self.signals = VolumeSegmentationWorkerSignals()

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        total = len(self._slice_indices)
        logger.info("VolumeSegmentationWorker starting: %d slice(s) to segment", total)
        for completed, slice_index in enumerate(self._slice_indices):
            if self._cancel_requested:
                logger.info("VolumeSegmentationWorker cancelled after %d/%d slice(s)", completed, total)
                self.signals.cancelled.emit(completed, total)
                return
            slice_hu: np.ndarray = self._get_slice_hu(slice_index)
            try:
                result = segment_slice(self._model, slice_hu)
            except Exception as exc:  # report to the GUI thread instead of crashing the pool thread
                logger.exception("VolumeSegmentationWorker failed on slice %d", slice_index)
                self.signals.error.emit(slice_index, str(exc))
                return
            self.signals.slice_finished.emit(slice_index, result)
            self.signals.progress.emit(completed + 1, total)
        logger.info("VolumeSegmentationWorker finished: %d/%d slice(s) segmented", total, total)
        self.signals.finished.emit()
