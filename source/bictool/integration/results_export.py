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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook

from bictool.domain.results.study_quality import SliceQualityResult
from bictool.domain.results.study_summary import study_verdict, summarize_metrics

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


# --- Whole-study export (global QA): summary + one row per slice ---

# Study summary metric -> ASCII label (same wording as the per-slice export).
_STUDY_METRIC_LABELS = {
    "mtf_area": "MTF area (mm^-1)",
    "nps_variance": "NPS variance (HU^2mm^2)",
    "ps_mtf_ratio": "PS/MTF ratio (LF/HF)",
    "contrast": "Contrast (HU)",
    "decision_value": "Decision value",
}
_SLICE_TABLE_HEADER = ["Slice", *_STUDY_METRIC_LABELS.values(), "Verdict", *_FEATURE_NAMES]


@dataclass
class StudyExportContext:
    """Non-identifying context for a whole-study export (see `ExportContext`).
    `failed_slices` are 1-based slice numbers whose QA failed - listed, not
    averaged.
    """

    series_label: str
    slice_count: int
    pixel_spacing_mm: float
    acquisition_date: str | None
    failed_slices: list[int]


def _study_summary_rows(results: Mapping[int, SliceQualityResult], context: StudyExportContext) -> list[list[str]]:
    verdict = study_verdict(results.values())
    summary = summarize_metrics(results.values())
    rows = [
        ["Series", context.series_label],
        ["Slices in series", str(context.slice_count)],
        ["Slices analysed", str(len(results))],
        ["Failed slices", ", ".join(str(number) for number in context.failed_slices)],
        ["Pixel spacing (mm)", f"{context.pixel_spacing_mm:.4f}"],
        ["Acquisition date", context.acquisition_date or ""],
        [],
        ["Study verdict", verdict.verdict if verdict else ""],
        ["Acceptable slices", f"{verdict.acceptable} of {verdict.total}" if verdict else ""],
        [],
        ["Metric", "Mean", "SD", "Slices"],
    ]
    for name, label in _STUDY_METRIC_LABELS.items():
        metric = summary.get(name)
        if metric is not None:
            rows.append([label, f"{metric.mean:.6g}", f"{metric.std:.6g}", str(metric.count)])
    return rows


def _study_slice_rows(results: Mapping[int, SliceQualityResult]) -> list[list[str]]:
    """`_SLICE_TABLE_HEADER` then one row per slice (1-based numbers, in order)."""
    rows = [list(_SLICE_TABLE_HEADER)]
    for index in sorted(results):
        result = results[index]
        rows.append(
            [
                str(index + 1),
                f"{result.mtf_area:.6g}",
                f"{result.nps_roi2.var_nps:.6g}",
                f"{result.ratio.ratio_lf_hf:.6g}",
                f"{result.contrast_roi1:.6g}",
                f"{result.decision_value:.6g}",
                "Acceptable" if result.predicted_label == 1 else "Unacceptable",
                *(f"{value:.6g}" for value in result.feature_vector),
            ]
        )
    return rows


def export_study_csv(
    results: Mapping[int, SliceQualityResult],
    context: StudyExportContext,
    summary_path: str | Path,
    slices_path: str | Path,
) -> None:
    """Two CSV files (CSV has no sheets): the study summary and the per-slice
    table. `results` is keyed by 0-based slice index.
    """
    for path, rows in ((summary_path, _study_summary_rows(results, context)), (slices_path, _study_slice_rows(results))):
        with open(path, "w", newline="", encoding="ascii", errors="replace") as f:
            csv.writer(f).writerows(rows)


def export_study_excel(results: Mapping[int, SliceQualityResult], context: StudyExportContext, path: str | Path) -> None:
    """One workbook: "Study summary" and "Per slice" sheets. `results` is
    keyed by 0-based slice index.
    """
    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Study summary"
    for row in _study_summary_rows(results, context):
        summary_sheet.append(row)
    summary_sheet.column_dimensions["A"].width = 26
    for column in "BCD":
        summary_sheet.column_dimensions[column].width = 14

    slices_sheet = workbook.create_sheet("Per slice")
    for row in _study_slice_rows(results):
        slices_sheet.append(row)
    for (header_cell,) in slices_sheet.iter_cols(min_row=1, max_row=1):
        slices_sheet.column_dimensions[header_cell.column_letter].width = max(len(str(header_cell.value)) + 2, 10)
    workbook.save(str(path))
