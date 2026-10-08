"""StrokeViewModel: investigational stroke-lesion check of one slice.

Not an image-quality measurement (see `domain.segmentation.aisd_stroke_unet`)
- the feature is currently hidden in the GUI. Owns the per-slice results,
the loaded model and the running job; reports a status and messages, the
view draws the lesion overlay. TensorFlow is imported only on first use.
QtCore only - no widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import bictool.integration.model_files as model_files
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

NOT_CONFIGURED_TEXT = (
    "No stroke segmentation weights are configured. Set BICTOOL_STROKE_WEIGHTS "
    "or the segmentation.stroke_weights_path config key - see "
    "domain.segmentation.aisd_stroke_unet for where to obtain the weights file."
)


class StrokeViewModel(QObject):
    status_changed = Signal(str)
    status_detail_changed = Signal(str)
    busy_changed = Signal(bool)
    slice_checked = Signal(int, bool)   # (slice, from cache) - its result is in `cache`
    message = Signal(str, str, str)     # (level, title, text) for a dialog

    def __init__(
        self, submit: Callable[[object], None], config: Callable[[], dict], volume: Callable[[], object],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._config = config
        self._volume = volume
        self.cache: dict[int, object] = {}  # slice index -> StrokeSegmentationResult
        self.model = None
        self.active_worker = None
        self._checking: int | None = None

    def has_weights(self) -> bool:
        return bool(self._config().get("segmentation", {}).get("stroke_weights_path"))

    def lesion_mask(self, index: int):
        result = self.cache.get(index)
        return None if result is None else result.masks["Lesion"]

    def check_slice(self, index: int) -> None:
        if self._volume() is None:
            return
        if index in self.cache:
            self.status_changed.emit("Done (cached)")
            self.slice_checked.emit(index, True)
            return
        weights_path = self._config().get("segmentation", {}).get("stroke_weights_path")
        if not weights_path:
            self.message.emit("warning", "Stroke model not configured", NOT_CONFIGURED_TEXT)
            return
        try:
            # Deferred to first actual use (TensorFlow's import cost).
            from bictool.application.stroke_worker import StrokeWorker
            from bictool.domain.segmentation import aisd_stroke_unet
        except ImportError as exc:
            self.message.emit(
                "critical", "TensorFlow not installed", f"Stroke detection requires TensorFlow (pip install bictool[ml]): {exc}"
            )
            return
        if self.model is None:
            self.model = aisd_stroke_unet.build_model()
        try:
            model_files.load_stroke_weights(self.model, weights_path)
        except FileNotFoundError as exc:
            self.message.emit("critical", "Stroke model weights not found", str(exc))
            return
        logger.info("Checking slice %d for a stroke lesion...", index)
        self.status_changed.emit("Checking…")
        self.busy_changed.emit(True)
        self._checking = index
        worker = StrokeWorker(self.model, self._volume().get_slice_hu(index))
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self.active_worker = worker
        self._submit(worker)

    def reset(self) -> None:
        if self.active_worker is not None:
            self.active_worker.signals.blockSignals(True)
            self.active_worker = None
        self.cache = {}
        self.status_changed.emit("—")
        self.status_detail_changed.emit("")

    def _on_finished(self, result) -> None:
        index, self._checking = self._checking, None
        logger.info("Stroke check finished for slice %d.", index)
        self.active_worker = None
        self.cache[index] = result
        self.status_changed.emit("Done")
        self.busy_changed.emit(False)
        self.slice_checked.emit(index, False)

    def _on_error(self, message: str) -> None:
        logger.error("Stroke check failed: %s", message)
        self.active_worker = None
        self._checking = None
        self.status_changed.emit("Error")
        self.status_detail_changed.emit(message)
        self.busy_changed.emit(False)
