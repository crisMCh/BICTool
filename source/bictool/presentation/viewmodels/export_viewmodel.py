"""ExportViewModel: saving results (FR-07) - the shown slice, or the whole study.

Keeps the shown slice's result and its export context (no patient
identifiers, see `integration.results_export`), suggests a readable file
name, writes Excel/CSV through the integration layer and reports the
outcome as a `message` the view shows. The user picks the name and folder
(a save dialog in the view); every file of one export shares that name.
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
    iso_date,
)
from bictool.presentation.viewmodels.display_text import safe_file_name
from bictool.presentation.viewmodels.roi_pairs import PAIR_LABELS, PAIRS

logger = logging.getLogger(__name__)

SCOPE_SLICE = "slice"
SCOPE_STUDY = "study"
# Export items, in the order their file type is offered in the save dialog.
_FILE_TYPES = (
    ("excel", ".xlsx", "Excel workbook (*.xlsx)"),
    ("csv", ".csv", "CSV table (*.csv)"),
    ("jpg", ".jpg", "JPG images (*.jpg)"),
)


def _folder_and_name(path: Path) -> tuple[Path, str]:
    """The chosen save path without the extension the save dialog may add."""
    name = path.name
    for _item, extension, _filter in _FILE_TYPES:
        if name.lower().endswith(extension):
            name = name[: -len(extension)]
    return path.parent, safe_file_name(name)


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

    # ---- naming ----

    def suggested_name(self, scope: str) -> str:
        """A readable default file name, e.g. "BICTool study 2026-01-31 CCT nativ Hr44"
        or "BICTool slice 18 Falx 2026-01-31 CCT nativ Hr44"."""
        volume = self._volume()
        what = "study"
        if scope == SCOPE_SLICE and self.last_context is not None:
            what = f"slice {self.last_context.slice_number} {self.last_context.structure or ''}"
        parts = [
            "BICTool",
            what,
            iso_date(getattr(volume, "acquisition_date", None)),
            getattr(volume, "series_description", None) or "",
        ]
        return safe_file_name(" ".join(part for part in parts if part))

    @staticmethod
    def file_type(items: set[str]) -> tuple[str, str]:
        """(extension, save-dialog filter) of the export's main file."""
        return next((extension, name) for item, extension, name in _FILE_TYPES if item in items or item == "jpg")

    # ---- exporting ----

    def export_slice(
        self, items: set[str], path: Path, save_charts: Callable[[Path, str], list[Path]] | None = None
    ) -> list[Path]:
        """The shown slice: Excel and/or CSV metrics, chart images ("jpg") via
        `save_charts(folder, name)`. `path`: the chosen file (any extension)."""
        if not self.can_export_slice():
            return []
        folder, name = _folder_and_name(Path(path))
        written: list[Path] = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if "excel" in items:
                written.append(folder / f"{name}.xlsx")
                export_metrics_excel(self.last_result, self.last_context, written[-1])
            if "csv" in items:
                written.append(folder / f"{name}.csv")
                export_metrics_csv(self.last_result, self.last_context, written[-1])
            if "jpg" in items and save_charts is not None:
                written += save_charts(folder, name)
        except OSError as exc:
            logger.exception("Export failed")
            self.message.emit("critical", "Export failed", str(exc))
            return []
        logger.info("Exported %d file(s) to %s", len(written), folder)
        self._report_written(written, folder)
        return written

    def export_study(self, items: set[str], path: Path, series_label: str) -> list[Path]:
        """Study report (Excel) and/or tables (CSV: summary + one row per
        slice; failed slices listed, not averaged), with the study verdict.
        `path`: the chosen file (any extension)."""
        volume = self._volume()
        if not self._qa.results or volume is None:
            return []
        headline, tone, lines = self._results.report()
        context = StudyExportContext(
            series_label=series_label or "Series",
            slice_count=volume.num_slices,
            pixel_spacing_mm=volume.pixel_spacing_mm,
            acquisition_date=volume.acquisition_date,
            failed_slices=[f"{index + 1} ({PAIR_LABELS[pair]})" for index, pair in sorted(self._qa.errors)],
            head_ct_verdict=self._results.verdict,
            artefacts=self._results.artefacts,
            basal_ganglia=self._results.basal_ganglia,
            segmentation_plausibility=self._results.plausibility,
            headline=headline,
            headline_tone=tone,
            summary_lines=tuple(lines),
        )
        by_pair = {PAIR_LABELS[pair]: self._qa.results_by_pair(pair) for pair in PAIRS if self._qa.results_by_pair(pair)}
        folder, name = _folder_and_name(Path(path))
        written: list[Path] = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if "excel" in items:
                written.append(folder / f"{name}.xlsx")
                export_study_excel(by_pair, context, written[-1])
            if "csv" in items:
                summary_path, slices_path = folder / f"{name} - summary.csv", folder / f"{name} - per slice.csv"
                export_study_csv(by_pair, context, summary_path, slices_path)
                written += [summary_path, slices_path]
        except OSError as exc:
            logger.exception("Study export failed")
            self.message.emit("critical", "Export failed", str(exc))
            return []
        logger.info(
            "Exported study (%d ROI pair result(s)) to %d file(s) in %s", len(self._qa.results), len(written), folder
        )
        self._report_written(written, folder)
        return written

    def _report_written(self, written: list[Path], folder: Path) -> None:
        names = "\n".join(f"  {path.name}" for path in written)
        self.message.emit("information", "Export complete", f"Saved in {folder}:\n{names}")
