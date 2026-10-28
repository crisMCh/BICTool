"""StudyViewModel: which study/series is open and which slice is shown.

Owns the state the rest of the workflow hangs off - the loaded volume, the
series found in the opened folder, whether the shown series is the study's
primary one (`discover_series`' best guess, index 0), the current slice -
and the commands that change it (open a path, pick a series, reload, close,
go to a slice). Loading runs in the background (`DicomLoadWorker`); the
view only binds to the signals below. QtCore only - no widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from bictool.application.dicom_load_worker import DicomLoadWorker
from bictool.integration.dicom_file_loader import DicomVolume, SeriesCandidate, load_series

logger = logging.getLogger(__name__)

NO_STUDY_TEXT = "No study loaded — File → Open DICOM Folder/File…"


class StudyViewModel(QObject):
    status_changed = Signal(str)            # status-bar text ("Loading…", "Loaded: …")
    loading_changed = Signal(bool)          # a folder/file scan is running
    series_list_changed = Signal(list)      # series labels (index 0 = primary)
    # (volume, is_primary_series, force_reload, previous_was_primary): a
    # series was loaded; `previous_was_primary` = the series shown until now
    # was the study's primary one (its segmentation is worth keeping).
    volume_loaded = Signal(object, bool, bool, bool)
    load_failed = Signal(str)
    slice_changed = Signal(int)
    closed = Signal()

    def __init__(self, submit: Callable[[object], None], parent: QObject | None = None) -> None:
        """`submit` runs a QRunnable in the background (e.g. `Scheduler.submit`)."""
        super().__init__(parent)
        self._submit = submit
        self.volume: DicomVolume | None = None
        self.series_candidates: list[SeriesCandidate] = []
        self.series_index = -1
        self.is_primary_series = False
        self.current_slice = 0
        self.active_load_worker: DicomLoadWorker | None = None

    # ---- commands ----

    def open_path(self, path: str) -> None:
        """Scans `path` (folder or file) for DICOM series in the background."""
        logger.info("Open DICOM: %s", path)
        loader = DicomLoadWorker(path)
        loader.signals.finished.connect(self.on_series_discovered)
        loader.signals.error.connect(self._on_load_error)
        loader.signals.progress.connect(self._on_load_progress)
        self.active_load_worker = loader  # keep a reference alive while it runs
        self.loading_changed.emit(True)
        self.status_changed.emit("Loading DICOM data…")
        self._submit(loader)

    def on_series_discovered(self, candidates: list[SeriesCandidate]) -> None:
        logger.info("Discovered %d series candidate(s): %s", len(candidates), [c.label for c in candidates])
        self.active_load_worker = None
        self.loading_changed.emit(False)
        # A fresh discovery belongs to a (possibly) different study: the
        # series shown until now is no longer "this study's primary series".
        self.is_primary_series = False
        self.series_candidates = list(candidates)
        self.series_list_changed.emit([c.label for c in candidates])
        self.select_series(0)

    def select_series(self, index: int, *, force_reload: bool = False) -> None:
        if not 0 <= index < len(self.series_candidates):
            return
        logger.info("Loading series: %s", self.series_candidates[index].label)
        self.series_index = index
        self.adopt_volume(load_series(self.series_candidates[index]), is_primary_series=index == 0,
                          force_reload=force_reload)

    def reload_series(self) -> None:
        if self.series_candidates:
            index = max(self.series_index, 0)
            logger.info("Reloading series from disk: %s", self.series_candidates[index].label)
            self.select_series(index, force_reload=True)

    def adopt_volume(self, volume: DicomVolume, *, is_primary_series: bool = True, force_reload: bool = False) -> None:
        """Makes `volume` the shown series (also the entry point for a volume
        loaded by other means, e.g. tests)."""
        # No patient identifiers in the log - it is the file users send for troubleshooting.
        logger.info("Volume loaded: %d slice(s), pixel_spacing=%.4f mm", volume.num_slices, volume.pixel_spacing_mm)
        previous_was_primary = self.is_primary_series and self.volume is not None
        self.active_load_worker = None
        self.volume = volume
        self.is_primary_series = is_primary_series
        self.current_slice = 0
        study_label = volume.study_id or volume.patient_id or "DICOM data"
        self.status_changed.emit(f"Loaded: {study_label} ({volume.num_slices} slice(s))")
        self.volume_loaded.emit(volume, is_primary_series, force_reload, previous_was_primary)

    def set_slice(self, index: int) -> None:
        if self.volume is None:
            return
        index = max(0, min(index, self.volume.num_slices - 1))
        self.current_slice = index
        self.slice_changed.emit(index)

    def close(self) -> None:
        logger.info("Study closed by user")
        self.volume = None
        self.series_candidates = []
        self.series_index = -1
        self.is_primary_series = False
        self.current_slice = 0
        self.status_changed.emit(NO_STUDY_TEXT)
        self.closed.emit()

    # ---- background loading ----

    def _on_load_progress(self, files_scanned: int) -> None:
        self.status_changed.emit(f"Loading DICOM data… ({files_scanned} files scanned)")

    def _on_load_error(self, message: str) -> None:
        logger.error("Failed to open DICOM data: %s", message)
        self.active_load_worker = None
        self.loading_changed.emit(False)
        self.status_changed.emit(NO_STUDY_TEXT)
        self.load_failed.emit(message)
