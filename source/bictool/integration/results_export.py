"""Writes a computed `SliceQualityResult` to Excel/ASCII files (FR-07:
"export results in Excel, ASCII, or DICOM-compatible formats for reporting
and integration" - DICOM-compatible/structured-report export is out of
scope for this pass, see project memory).

Deliberately excludes patient name/ID (see `ExportContext`'s docstring) -
an exported file is far more likely to leave the workstation (email, a
shared drive) than the live GUI, so it stays free of direct identifiers by
default even though the GUI itself shows them (NFR-06, GDPR/BDSG), matching
the intent already noted in `design/bictool_console_mockup.html`'s export
modal ("Exports contain only de-identified fields").

Chart-image export (JPG) isn't here - it needs a live `QChartView` to
render from, so `main_window` handles that step directly rather than
through this (Qt-free, testable) module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook

from bictool.domain.results.study_quality import SliceQualityResult

_FEATURE_NAMES = ["mtf_area", "nps_homo", "nps_struct", "var_struct", "ratio_lf_hf", "slope_avg", "loc_max"]


@dataclass
class ExportContext:
    """Non-identifying context describing which slice/series a result came
    from - everything here is safe to write to an exported file (no patient
    name/ID/institution).
    """

    series_label: str
    slice_number: int
    slice_count: int
    pixel_spacing_mm: float
    acquisition_date: str | None


def _export_rows(result: SliceQualityResult, context: ExportContext) -> list[tuple[str, str]]:
    """Builds the (label, value) rows shared by both the Excel and CSV
    writers, so the two formats can never drift apart in content.
    """
    is_acceptable = result.predicted_label == 1
    rows: list[tuple[str, str]] = [
        ("Series", context.series_label),
        ("Slice", f"{context.slice_number}/{context.slice_count}"),
        ("Pixel spacing (mm)", f"{context.pixel_spacing_mm:.4f}"),
        ("Acquisition date", context.acquisition_date or ""),
        ("", ""),
        ("Metric", "Value"),
        # Plain-ASCII unit notation ("mm^-1", not "mm⁻¹") - this is the
        # *ASCII* export format (FR-07), so the content itself should never
        # need Unicode superscripts, rather than relying on a lossy
        # replace-on-encode fallback for GUI-style symbols.
        ("MTF area (mm^-1)", f"{result.mtf_area:.4f}"),
        ("NPS variance (HU^2mm^2)", f"{result.nps_roi2.var_nps:.6f}"),
        ("PS/MTF ratio (LF/HF)", f"{result.ratio.ratio_lf_hf:.4f}"),
        ("Contrast (HU)", f"{result.contrast_roi1:.2f}"),
        ("Decision value", f"{result.decision_value:.4f}"),
        ("Verdict", "Acceptable" if is_acceptable else "Unacceptable"),
        ("", ""),
        ("Feature vector entry", "Value"),
    ]
    rows.extend((name, f"{value:.6g}") for name, value in zip(_FEATURE_NAMES, result.feature_vector))
    return rows


def export_metrics_csv(result: SliceQualityResult, context: ExportContext, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="ascii", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerows(_export_rows(result, context))


def export_metrics_excel(result: SliceQualityResult, context: ExportContext, path: str | Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "BICTool metrics"
    for row in _export_rows(result, context):
        sheet.append(row)
    sheet.column_dimensions["A"].width = 26
    sheet.column_dimensions["B"].width = 16
    workbook.save(str(path))
