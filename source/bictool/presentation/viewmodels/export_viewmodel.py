"""ExportViewModel: saving results (FR-07) - the shown slice, or the whole study.

Keeps the shown slice's result and its export context (no patient
identifiers, see `integration.results_export`), suggests a readable file
name, writes Excel/CSV through the integration layer and reports the
outcome as a `message` the view shows. The user picks the name and folder
(a save dialog in the view); every file of one export shares that name.
Chart images need the widgets, so the view passes a function that saves them.
Plug-ins may add study file formats (`plugin_formats`); they get the run as the
API's `StudyResult` and write it themselves - a failing one is reported, the
other files are written. QtCore only - no widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from bictool.application import pipeline, plugin_manager
from bictool.integration.results_export import (
    ExportContext,
    StudyExportContext,
    export_metrics_csv,
    export_metrics_excel,
    export_study_csv,
    export_study_excel,
    plugin_rows,
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
    extensions = [extension for _item, extension, _filter in _FILE_TYPES]
    extensions += [fmt.extension for fmt in plugin_manager.current().export_formats.values()]
    for extension in extensions:
        if name.lower().endswith(extension.lower()):
            name = name[: -len(extension)]
    return path.parent, safe_file_name(name)


class ExportViewModel(QObject):
    message = Signal(str, str, str)  # (level "information"/"critical", title, text) for a dialog

    def __init__(
        self, qa, results, volume: Callable[[], object], parent: QObject | None = None,
        study_result: Callable[[], object] | None = None,
    ) -> None:
        """`qa` = QaRunViewModel, `results` = ResultsViewModel (study verdict),
        `volume()` = the shown DICOM volume, `study_result()` = the run as an
        `api.StudyResult` (for plug-in formats)."""
        super().__init__(parent)
        self._qa = qa
        self._results = results
        self._volume = volume
        self._study_result = study_result
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
        """A readable default file name, e.g. "BICTool study 2026-03-12 Head native 4.0 Hr40"
        or "BICTool slice 18 Falx 2026-03-12 Head native 4.0 Hr40"."""
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
        for item, extension, name in _FILE_TYPES:
            if item in items:
                return extension, name
        formats = plugin_manager.current().export_formats
        for item in sorted(items):
            if item in formats:
                return formats[item].extension, f"{formats[item].label} (*{formats[item].extension})"
        return _FILE_TYPES[-1][1], _FILE_TYPES[-1][2]

    def plugin_formats(self) -> list[tuple[str, str]]:
        """The study file formats plug-ins add: [(id, "label (.ext)")] (none without such plug-ins)."""
        formats = plugin_manager.current().export_formats
        if self._study_result is None:
            return []
        return [(format_id, f"{fmt.label} ({fmt.extension})") for format_id, fmt in sorted(formats.items())]

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
            plugin_rows=tuple(
                tuple(row) for row in plugin_rows(
                    self._qa.plugin_values, self._qa.study_plugin_values, PAIR_LABELS,
                    slice_values=self._qa.slice_plugin_values, hints=self._qa.plugin_hints,
                    tissue_method=self._qa.plugin_tissue_method and pipeline.tissue_method_label(self._qa.plugin_tissue_method),
                )
            ),
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
                plugins_path = folder / f"{name} - plug-ins.csv"
                export_study_csv(by_pair, context, summary_path, slices_path, plugins_path)
                written += [summary_path, slices_path] + ([plugins_path] if context.plugin_rows else [])
        except OSError as exc:
            logger.exception("Study export failed")
            self.message.emit("critical", "Export failed", str(exc))
            return []
        failures = self._write_plugin_formats(items, folder, name, written)
        logger.info(
            "Exported study (%d ROI pair result(s)) to %d file(s) in %s", len(self._qa.results), len(written), folder
        )
        self._report_written(written, folder, failures)
        return written

    def _write_plugin_formats(self, items: set[str], folder: Path, name: str, written: list[Path]) -> list[str]:
        """The plug-in formats among `items` (appended to `written`); returns what failed."""
        manager = plugin_manager.current()
        chosen = [item for item in sorted(items) if item in manager.export_formats]
        if not chosen or self._study_result is None:
            return []
        result = self._study_result()
        failures = []
        for format_id in chosen:
            fmt = manager.export_formats[format_id]
            target = folder / f"{name}{fmt.extension}"
            if target in written:  # e.g. a plug-in's ".xlsx" next to the report
                target = folder / f"{name} - {safe_file_name(format_id)}{fmt.extension}"
            problem = manager.write_export(format_id, result, target)
            if problem:
                failures.append(f"{fmt.label}: {problem}")
            else:
                written.append(target)
        return failures

    def _report_written(self, written: list[Path], folder: Path, failures: list[str] = ()) -> None:
        names = "\n".join(f"  {path.name}" for path in written)
        if failures:
            problems = "\n".join(f"  {failure}" for failure in failures)
            self.message.emit(
                "warning", "Export: plug-in problem", f"Saved in {folder}:\n{names}\n\nNot written (plug-in):\n{problems}"
            )
            return
        self.message.emit("information", "Export complete", f"Saved in {folder}:\n{names}")
