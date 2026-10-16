"""ExportViewModel: saving results (FR-07) - the shown slice, or the whole study.

Keeps the shown slice's result and its export context (no patient
identifiers, see `integration.results_export`), writes Excel/CSV through
the integration layer and reports the outcome as a `message` the view shows.
Chart images need the widgets, so the view passes a function that saves them.
QtCore only - no widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from bictool.integration.results_export import (
    ExportContext,
    StudyExportContext,
    export_metrics_csv,
    export_metrics_excel,
    export_study_csv,
    export_study_excel,
)
from bictool.presentation.viewmodels.display_text import safe_file_stem
from bictool.presentation.viewmodels.roi_pairs import PAIR_LABELS, PAIRS

logger = logging.getLogger(__name__)


class ExportViewModel(QObject):
    message = Signal(str, str, str)  # (level "information"/"critical", title, text) for a dialog

    def __init__(self, qa, results, volume: Callable[[], object], parent: QObject | None = None) -> None:
        """`qa` = QaRunViewModel, `results` = ResultsViewModel (study verdict),
        `volume()` = the shown DICOM volume."""
        super().__init__(parent)
        self._qa = qa
        self._results = results
        self._volume = volume
        self.last_result = None
        self.last_context: ExportContext | None = None

    def can_export_slice(self) -> bool:
        return self.last_result is not None and self.last_context is not None

    def can_export_study(self) -> bool:
        return bool(self._qa.results)

    def show_slice(self, index: int, pair: str, result, series_label: str) -> None:
        """The Results page now shows `result` (slice `index`, pair `pair`) - what a slice export saves."""
        volume = self._volume()
        self.last_result = result
        self.last_context = ExportContext(
            series_label=series_label or "Series",
            slice_number=index + 1,
            slice_count=volume.num_slices,
            pixel_spacing_mm=volume.pixel_spacing_mm,
            acquisition_date=volume.acquisition_date,
            structure=PAIR_LABELS[pair],
        )

    def clear(self) -> None:
        self.last_result = None
        self.last_context = None

    def export_slice(
        self, items: set[str], destination: Path, save_charts: Callable[[Path, str], list[Path]] | None = None
    ) -> list[Path]:
        """The shown slice: Excel and/or CSV metrics, chart images ("jpg") via `save_charts`."""
        if not self.can_export_slice():
            return []
        context = self.last_context
        stem = f"{safe_file_stem(context.series_label)}_slice{context.slice_number:03d}"
        written: list[Path] = []
        try:
            destination.mkdir(parents=True, exist_ok=True)
            if "excel" in items:
                written.append(destination / f"{stem}_metrics.xlsx")
                export_metrics_excel(self.last_result, context, written[-1])
            if "csv" in items:
                written.append(destination / f"{stem}_metrics.csv")
                export_metrics_csv(self.last_result, context, written[-1])
            if "jpg" in items and save_charts is not None:
                written += save_charts(destination, stem)
        except OSError as exc:
            logger.exception("Export failed")
            self.message.emit("critical", "Export failed", str(exc))
            return []
        logger.info("Exported %d file(s) to %s", len(written), destination)
        self.message.emit("information", "Export complete", f"Wrote {len(written)} file(s) to:\n{destination}")
        return written

    def export_study(self, items: set[str], destination: Path, series_label: str) -> list[Path]:
        """Study averages + one row per slice from every QA result (failed
        slices listed, not averaged), with the study verdict."""
        volume = self._volume()
        if not self._qa.results or volume is None:
            return []
        context = StudyExportContext(
            series_label=series_label or "Series",
            slice_count=volume.num_slices,
            pixel_spacing_mm=volume.pixel_spacing_mm,
            acquisition_date=volume.acquisition_date,
            failed_slices=[f"{index + 1} ({PAIR_LABELS[pair]})" for index, pair in sorted(self._qa.errors)],
            head_ct_verdict=self._results.verdict,
            artefacts=self._results.artefacts,
            basal_ganglia=self._results.basal_ganglia,
        )
        by_pair = {PAIR_LABELS[pair]: self._qa.results_by_pair(pair) for pair in PAIRS if self._qa.results_by_pair(pair)}
        stem = f"{safe_file_stem(context.series_label)}_study"
        written: list[Path] = []
        try:
            destination.mkdir(parents=True, exist_ok=True)
            if "excel" in items:
                written.append(destination / f"{stem}_metrics.xlsx")
                export_study_excel(by_pair, context, written[-1])
            if "csv" in items:
                summary_path, slices_path = destination / f"{stem}_summary.csv", destination / f"{stem}_slices.csv"
                export_study_csv(by_pair, context, summary_path, slices_path)
                written += [summary_path, slices_path]
        except OSError as exc:
            logger.exception("Study export failed")
            self.message.emit("critical", "Export failed", str(exc))
            return []
        logger.info(
            "Exported study (%d ROI pair result(s)) to %d file(s) in %s", len(self._qa.results), len(written), destination
        )
        self.message.emit("information", "Export complete", f"Wrote {len(written)} file(s) to:\n{destination}")
        return written
