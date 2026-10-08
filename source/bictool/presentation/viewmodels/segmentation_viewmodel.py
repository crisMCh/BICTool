"""SegmentationViewModel: the anatomy segmentation (CNN) of the shown series.

Owns the per-slice results (`cache`), the DICOM->NIfTI conversion the model
needs (`nifti_volume`, built once per series), the background jobs (NIfTI
conversion, one slice, the whole volume - cancellable), the loaded model,
the primary series' results kept while another series is shown, and the
overlay choices (selected structures, opacity, show/hide).

The view binds to the signals; it shows `message` as a dialog and draws the
overlays. TensorFlow and the workers are imported only when a segmentation
is actually started (app start stays fast). QtCore only - no widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import bictool.integration.model_files as model_files
from PySide6.QtCore import QObject, Signal

from bictool.presentation.viewmodels.segmentation_overlay import decode_masks_for_checkboxes

logger = logging.getLogger(__name__)

DEFAULT_OPACITY = 0.6
NOT_CONFIGURED_TEXT = (
    "No anatomy segmentation weights are configured. Set BICTOOL_HEADCT_WEIGHTS "
    "or the segmentation.anatomy_weights_path config key - see "
    "domain.segmentation.head_ct_unet for where to obtain the weights file."
)


class SegmentationViewModel(QObject):
    status_changed = Signal(str)            # status-bar text ("Segmenting volume… 3/40", "Done", ...)
    status_detail_changed = Signal(str)     # e.g. an error message (tooltip)
    busy_changed = Signal(bool)             # a NIfTI conversion / segmentation job runs
    cancel_available_changed = Signal(bool)  # the running job (volume) can be cancelled
    slice_segmented = Signal(int)           # slice index whose result is now in `cache`
    message = Signal(str, str, str)         # (level "warning"/"critical", title, text) for a dialog

    def __init__(
        self,
        submit: Callable[[object], None],
        config: dict,
        volume: Callable[[], object],
        mapping: dict,
        parent: QObject | None = None,
    ) -> None:
        """`submit` runs a QRunnable in the background; `config` is the app
        config (weights paths); `volume()` returns the shown DICOM volume;
        `mapping` = display name -> model classes (`decode_masks_for_checkboxes`)."""
        super().__init__(parent)
        self._submit = submit
        self.config = config
        self._volume = volume
        self._mapping = mapping
        self.cache: dict[int, object] = {}  # slice index -> head_ct_unet.SegmentationResult (DICOM space)
        self.nifti_volume = None
        self.anatomy_model = None
        self.active_slice_worker = None
        self.active_volume_worker = None
        self.active_nifti_worker = None
        self.primary_cache: dict[int, object] | None = None
        self.primary_nifti_volume = None
        self.selected_names: set[str] = set(mapping)
        self.opacity = DEFAULT_OPACITY
        self.current_masks: dict | None = None  # decoded masks of the shown slice (or None)

    # ---- configuration ----

    @property
    def weights_path(self) -> str | None:
        return self.config.get("segmentation", {}).get("anatomy_weights_path")

    def has_weights(self) -> bool:
        return bool(self.weights_path)

    def is_busy(self) -> bool:
        return any(w is not None for w in (self.active_slice_worker, self.active_volume_worker, self.active_nifti_worker))

    # ---- per-slice display state ----

    def decoded_masks(self, index: int) -> dict | None:
        """The slice's masks by display name (None if not segmented)."""
        result = self.cache.get(index)
        return None if result is None else decode_masks_for_checkboxes(result, self._mapping)

    # ---- commands ----

    def segment_slice(self, index: int) -> None:
        """Segments slice `index` (or reports its cached result)."""
        if self._volume() is None or self.active_volume_worker is not None:
            return
        if index in self.cache:
            self._set_status("Done (cached)")
            self.slice_segmented.emit(index)
            return
        if not self._prepare_model("bictool.application.segmentation_worker"):
            return
        self.ensure_nifti_then(lambda: self._run_slice(index))

    def segment_volume(self) -> None:
        """Segments every slice not segmented yet, in the background (cancellable)."""
        volume = self._volume()
        if volume is None or self.active_volume_worker is not None:
            return
        if all(i in self.cache for i in range(volume.num_slices)):
            self._set_status("Done (all slices already segmented)")
            return
        if not self._prepare_model("bictool.application.volume_segmentation_worker"):
            return
        self.ensure_nifti_then(self._run_volume)

    def cancel_volume(self) -> None:
        if self.active_volume_worker is not None:
            self.active_volume_worker.cancel()
            self._set_status("Cancelling…")

    def reset(self) -> None:
        """A new/closed series: drop results, jobs and choices. A job still
        running can't be interrupted mid-slice - its late signals are blocked
        so a stale result never lands in another series' cache."""
        for worker in (self.active_slice_worker, self.active_nifti_worker):
            if worker is not None:
                worker.signals.blockSignals(True)
        if self.active_volume_worker is not None:
            self.active_volume_worker.cancel()
            self.active_volume_worker.signals.blockSignals(True)
        self.active_slice_worker = self.active_volume_worker = self.active_nifti_worker = None
        self.nifti_volume = None
        self.cache = {}
        self.current_masks = None
        self.selected_names = set(self._mapping)
        self.opacity = DEFAULT_OPACITY
        self._set_status("—")
        self.status_detail_changed.emit("")
        self.busy_changed.emit(False)
        self.cancel_available_changed.emit(False)

    def backup_primary(self) -> None:
        """Keep the primary series' results while another series is shown."""
        self.primary_cache = self.cache
        self.primary_nifti_volume = self.nifti_volume

    def forget_primary(self) -> None:
        self.primary_cache = None
        self.primary_nifti_volume = None

    def restore_primary(self) -> bool:
        """Back on the primary series: its results return. False if none kept."""
        if self.primary_cache is None:
            return False
        self.cache = self.primary_cache
        self.nifti_volume = self.primary_nifti_volume
        return True

    # ---- internals ----

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def _prepare_model(self, worker_module: str) -> bool:
        """Weights configured, TensorFlow and the worker importable, model
        built and loaded - else a message and False."""
        if not self.has_weights():
            self.message.emit("warning", "Segmentation model not configured", NOT_CONFIGURED_TEXT)
            return False
        try:
            # Deferred to first actual use (TensorFlow's import cost).
            __import__(worker_module)
            from bictool.domain.segmentation import head_ct_unet
        except ImportError as exc:
            self.message.emit(
                "critical", "TensorFlow not installed", f"Segmentation requires TensorFlow (pip install bictool[ml]): {exc}"
            )
            return False
        if self.anatomy_model is None:
            self.anatomy_model = head_ct_unet.build_model()
        try:
            model_files.load_anatomy_weights(self.anatomy_model, self.weights_path)
        except FileNotFoundError as exc:
            self.message.emit("critical", "Segmentation weights not found", str(exc))
            return False
        return True

    def ensure_nifti_then(self, on_ready: Callable[[], None]) -> None:
        """Runs `on_ready` once the NIfTI volume exists - built once per series
        in the background (`integration.dicom_to_nifti`)."""
        if self.nifti_volume is not None:
            on_ready()
            return
        if self.active_nifti_worker is not None:
            return  # already converting (e.g. a fast double-click)
        from bictool.application.nifti_conversion_worker import NiftiConversionWorker

        volume = self._volume()
        logger.info("Converting DICOM series to NIfTI for segmentation...")
        self._set_status("Converting to NIfTI…")
        self.busy_changed.emit(True)
        worker = NiftiConversionWorker(volume.slice_paths, volume.get_slice_hu)
        worker.signals.finished.connect(lambda result: self._on_nifti_finished(result, on_ready))
        worker.signals.error.connect(self._on_nifti_error)
        self.active_nifti_worker = worker  # keep a reference alive while it runs
        self._submit(worker)

    def _on_nifti_finished(self, result, on_ready) -> None:
        logger.info("NIfTI conversion finished.")
        self.active_nifti_worker = None
        self.nifti_volume = result
        self._set_status("—")
        self.busy_changed.emit(False)
        on_ready()

    def _on_nifti_error(self, message: str) -> None:
        logger.error("NIfTI conversion failed: %s", message)
        self.active_nifti_worker = None
        self._set_status("Error")
        self.status_detail_changed.emit(message)
        self.busy_changed.emit(False)

    def _run_slice(self, index: int) -> None:
        from bictool.application.segmentation_worker import SegmentationWorker

        if not self.nifti_volume.is_segmentable(index):
            self._set_status("Not segmentable")
            self.status_detail_changed.emit(
                "This slice has no corresponding processed volume slice (e.g. a scout/localizer "
                "sharing this series' geometry) - see integration.dicom_to_nifti."
            )
            return
        logger.info("Segmenting slice %d...", index)
        self._set_status("Segmenting…")
        self.busy_changed.emit(True)
        worker = SegmentationWorker(self.anatomy_model, self.nifti_volume.get_nifti_slice_hu(index))
        worker.signals.finished.connect(lambda result: self._on_slice_finished(index, result))
        worker.signals.error.connect(self._on_slice_error)
        self.active_slice_worker = worker
        self._submit(worker)

    def _on_slice_finished(self, index: int, result) -> None:
        logger.info("Segmentation finished for slice %d.", index)
        self.active_slice_worker = None
        # Computed on the NIfTI slice - mapped back onto the DICOM slice's own
        # pixel grid before anything else sees it.
        self.cache[index] = self.nifti_volume.remap_result_to_dicom_space(result)
        self._set_status("Done")
        self.busy_changed.emit(False)
        self.slice_segmented.emit(index)

    def _on_slice_error(self, message: str) -> None:
        logger.error("Segmentation failed: %s", message)
        self.active_slice_worker = None
        self._set_status("Error")
        self.status_detail_changed.emit(message)
        self.busy_changed.emit(False)

    def _run_volume(self) -> None:
        from bictool.application.volume_segmentation_worker import VolumeSegmentationWorker

        total = self._volume().num_slices
        pending = [i for i in range(total) if i not in self.cache and self.nifti_volume.is_segmentable(i)]
        not_segmentable = sum(1 for i in range(total) if i not in self.cache and not self.nifti_volume.is_segmentable(i))
        if not pending:
            self._set_status(
                "Done (all segmentable slices already segmented)" if not_segmentable else "Done (all slices already segmented)"
            )
            return
        logger.info(
            "Segmenting volume: %d of %d slice(s) pending (%d not segmentable)...", len(pending), total, not_segmentable
        )
        self._set_status(f"Segmenting volume… 0/{len(pending)}")
        self.busy_changed.emit(True)
        self.cancel_available_changed.emit(True)
        worker = VolumeSegmentationWorker(self.anatomy_model, self.nifti_volume.get_nifti_slice_hu, pending)
        worker.signals.slice_finished.connect(self._on_volume_slice_finished)
        worker.signals.progress.connect(lambda done, total_: self._set_status(f"Segmenting volume… {done}/{total_}"))
        worker.signals.finished.connect(self._on_volume_finished)
        worker.signals.cancelled.connect(self.on_volume_cancelled)
        worker.signals.error.connect(self._on_volume_error)
        self.active_volume_worker = worker
        self._submit(worker)

    def _on_volume_slice_finished(self, index: int, result) -> None:
        self.cache[index] = self.nifti_volume.remap_result_to_dicom_space(result)
        self.slice_segmented.emit(index)

    def _end_volume_job(self, status: str) -> None:
        self.active_volume_worker = None
        self._set_status(status)
        self.busy_changed.emit(False)
        self.cancel_available_changed.emit(False)

    def _on_volume_finished(self) -> None:
        logger.info("Volume segmentation finished.")
        self._end_volume_job("Done (volume)")

    def on_volume_cancelled(self, completed: int, total: int) -> None:
        logger.info("Volume segmentation cancelled after %d/%d slice(s).", completed, total)
        self._end_volume_job(f"Cancelled ({completed}/{total} segmented)")

    def _on_volume_error(self, index: int, message: str) -> None:
        logger.error("Volume segmentation failed on slice %d: %s", index, message)
        self._end_volume_job(f"Error (slice {index + 1})")
        self.status_detail_changed.emit(message)
