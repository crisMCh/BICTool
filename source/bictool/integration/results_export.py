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

Two audiences: the **Excel** workbooks are readable reports (plain names,
units, the verdict in words, formatting); the **CSV** files are the complete
plain-ASCII tables for analysis.

Chart-image export (JPG) isn't here - it needs a live `QChartView` to
render from, so `main_window` handles that step directly rather than
through this (Qt-free, testable) module.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from bictool import __version__

from bictool.domain.artefacts.artefact_flags import BAND_DEPTH_HU, METAL_HU, METAL_STREAK_SHARE, ArtefactFlags
from bictool.domain.metrics.basal_ganglia import NOT_PLAUSIBLE, PLAUSIBLE_DIFFERENCE_HU, BasalGangliaSlice
from bictool.domain.segmentation.segmentation_plausibility import FRAGMENTATION_LIMIT, SegmentationPlausibility
from bictool.domain.results.head_ct_verdict import (
    FEATURE_LABELS,
    MIN_SLICES,
    OUTLIER_FACTOR,
    PROVISIONAL_NOTE,
    HeadCtVerdict,
)
from bictool.domain.results.study_quality import RECIPE_PHYSICAL, SliceQualityResult
from bictool.domain.roi_selection.slice_eligibility import MIN_BRAIN_FRACTION
from bictool.domain.roi_selection.structured_roi import DURAL_FOLDS
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
    # Which ROI pair the result is from (e.g. "Falx", "Septum"), if known.
    structure: str | None = None


def _export_rows(result: SliceQualityResult, context: ExportContext) -> list[tuple[str, str]]:
    """Builds the (label, value) rows shared by both the Excel and CSV
    writers, so the two formats can never drift apart in content.
    """
    is_acceptable = result.predicted_label == 1
    suffix = _LEGACY_SVM if _is_physical([result]) else ""
    rows: list[tuple[str, str]] = [
        ("Series", context.series_label),
        ("Slice", f"{context.slice_number}/{context.slice_count}"),
        *([("Structure", context.structure)] if context.structure else []),
        ("Pixel spacing (mm)", f"{context.pixel_spacing_mm:.4f}"),
        ("QA recipe", _recipe_label([result])),
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
        # The SVM is trained on chest CT; under the physical recipe the verdict
        # of a head CT is the study-level one (study export), not this.
        (f"Decision value{suffix}", f"{result.decision_value:.4f}"),
        (f"Verdict{suffix}", "Acceptable" if is_acceptable else "Unacceptable"),
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
    """The shown slice as a readable one-sheet report."""
    workbook = Workbook()
    report = _Report(workbook.active, "Slice")
    report.title(f"{_REPORT_TITLE} - single slice")
    report.study_block([
        ("Series", context.series_label),
        ("Slice", f"{context.slice_number} of {context.slice_count}"),
        *([("ROI pair", context.structure)] if context.structure else []),
        ("Acquisition date", iso_date(context.acquisition_date)),
        ("Pixel spacing", f"{context.pixel_spacing_mm:.3f} mm"),
        ("QA method", _method_label([result])),
    ])
    report.heading("Measurements")
    report.header("Measurement", "Value", "Unit")
    values = {"mtf_area": result.mtf_area, "nps_variance": result.nps_roi2.var_nps,
              "ps_mtf_ratio": result.ratio.ratio_lf_hf, "contrast": result.contrast_roi1}
    for name, value in values.items():
        label, unit, number_format = _PLAIN_METRICS[name]
        report.row(label, value, unit, formats={2: number_format})
    report.blank()
    if _is_physical([result]):
        report.note("The study verdict covers the whole study - export the whole study to get it.")
    else:
        report.row("Verdict (chest-CT classifier)", "Acceptable" if result.predicted_label == 1 else "Unacceptable")
        report.row("Decision value", result.decision_value, formats={2: "0.000"})
    report.blank()
    report.heading("Technical details")
    if _is_physical([result]):
        report.row(f"Decision value{_LEGACY_SVM}", result.decision_value, formats={2: "0.000"})
        report.row(f"Verdict{_LEGACY_SVM}", "Acceptable" if result.predicted_label == 1 else "Unacceptable")
    report.header("Feature vector entry", "Value")
    for name, value in zip(_FEATURE_NAMES, result.feature_vector):
        report.row(name, float(value), formats={2: "0.0000"})
    report.widths(44, 16, 12)
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
_SLICE_TABLE_HEADER = ["Slice", "Structure", *_STUDY_METRIC_LABELS.values(), "Verdict", *_FEATURE_NAMES]
# Physical recipe: the chest-CT SVM's per-slice output is kept but marked, and
# the head-CT study model's per-slice view is added.
_LEGACY_SVM = " (legacy chest SVM, untrusted)"
_SLICE_TABLE_HEADER_PHYSICAL = [
    *(f"{name}{_LEGACY_SVM}" if name in ("Decision value", "Verdict") else name for name in _SLICE_TABLE_HEADER),
    "Slice score (indicative)",
    "Noise outlier",
]

# Study results per ROI pair: structure name (e.g. "Falx") -> {0-based slice: result}.
StudyResults = Mapping[str, Mapping[int, SliceQualityResult]]


@dataclass
class StudyExportContext:
    """Non-identifying context for a whole-study export (see `ExportContext`).
    `failed_slices` name the slices whose QA failed (e.g. "12 (Septum)") -
    listed, not averaged.
    """

    series_label: str
    slice_count: int
    pixel_spacing_mm: float
    acquisition_date: str | None
    failed_slices: list
    # The study-level head-CT verdict (physical recipe; None without one).
    head_ct_verdict: HeadCtVerdict | None = None
    # Possible-artefact hints (metal, posterior-fossa bands); None if not computed.
    artefacts: ArtefactFlags | None = None
    # Basal-ganglia visibility per slice (provisional); None if not computed.
    basal_ganglia: dict[int, BasalGangliaSlice] | None = None
    # Study-level segmentation check (provisional); None if it did not run.
    segmentation_plausibility: SegmentationPlausibility | None = None
    # The Study summary as the app words it (Excel report): headline, its tone
    # ("good"/"bad"/"neutral") and the plain lines below it.
    headline: str = ""
    headline_tone: str = ""
    summary_lines: tuple[str, ...] = ()


def _is_physical(results) -> bool:
    results = list(results)
    return bool(results) and all(getattr(r, "recipe", "octave") == RECIPE_PHYSICAL for r in results)


def _head_ct_verdict_rows(verdict: HeadCtVerdict | None, falx_slices: int) -> list[list[str]]:
    """The study-level verdict block: every value with its 95 % interval or
    reference range (docs/qa_method/head_ct_classifier.md, D7-D9)."""
    rows = [[], ["Study verdict (head-CT model, falx)"]]
    if verdict is None:
        return rows + [
            ["Verdict", f"not available - needs at least {MIN_SLICES} measured falx slices (this study: {falx_slices})"],
            ["Note", PROVISIONAL_NOTE],
        ]
    rows.append(["Verdict", verdict.verdict])
    if verdict.metal_rule:
        rows.append([
            "Metal rule", f"applied - metal streaks reach the brain on slices {_listed(verdict.metal_slices)}",
            f"model alone: {verdict.model_verdict}",
        ])
    rows += [
        ["P(acceptable)", f"{verdict.probability:.3f}"],
        ["95 % CI", f"{verdict.probability_low:.3f}", f"{verdict.probability_high:.3f}"],
        ["Falx slices used", str(verdict.n_slices)],
        ["Model", verdict.model_version],
        [],
        ["Feature (falx median)", "Median", "95 % CI low", "95 % CI high", "Good studies min", "Good studies max",
         "Outside good range", "Reason (towards not acceptable)"],
    ]
    reasons = {f.name for f in verdict.reasons}
    for f in verdict.features:
        rows.append([
            FEATURE_LABELS[f.name], f"{f.value:.6g}", f"{f.low:.6g}", f"{f.high:.6g}",
            f"{f.good_low:.6g}", f"{f.good_high:.6g}", f.outside or "", "yes" if f.name in reasons else "",
        ])
    a = verdict.artefact
    rows.append([])
    if a is None:
        rows.append(["Artefact indicator (provisional)", "not available - too few falx slices"])
    else:
        level = "confirmed" if a.flagged else "above good range, not confirmed" if a.above_good_range else "within good range"
        rows += [
            ["Artefact indicator (provisional)", "Noise scatter (IQR/median)", "95 % CI low", "95 % CI high",
             "Good studies max", "Assessment"],
            ["", f"{a.value:.4g}", f"{a.low:.4g}", f"{a.high:.4g}", f"{a.good_high:.4g}", level],
        ]
    rows.append([
        f"Slices with noise > {OUTLIER_FACTOR:g}x median",
        ", ".join(str(i + 1) for i in verdict.outlier_slices),
    ])
    for check in verdict.plausibility:
        rows.append([
            f"Septum - falx, {check.name}", f"{check.difference:.4g}",
            f"training {check.low:.4g} to {check.high:.4g}", "consistent" if check.consistent else "outside training range",
        ])
    rows.append(["Note", PROVISIONAL_NOTE])
    return rows


def _listed(indices) -> str:
    return ", ".join(str(index + 1) for index in indices)


def _artefact_rows(flags: ArtefactFlags | None) -> list[list[str]]:
    """Possible-artefact hints - not part of the verdict (metal streaks in the brain: see the metal rule)."""
    if flags is None:
        return []
    rows = [[], ["Possible artefacts (hints, not part of the verdict)"]]
    rows.append([
        f"Metal (> {METAL_HU:.0f} HU) on slices",
        ", ".join(str(index + 1) for index in flags.metal_slices) or "none",
        f"highest {flags.metal_max_hu:.0f} HU" if flags.metal_max_hu is not None else "",
    ])
    if flags.metal_streak_shares:
        rows.append([
            f"Metal streaks in the brain (>= {100 * METAL_STREAK_SHARE:.0f} % disturbed, provisional) on slices",
            _listed(flags.metal_streak_slices) or "none",
        ])
        rows.append(["Metal slice", "Brain near metal disturbed (share)"])
        rows += [[str(index + 1), f"{share:.3f}"] for index, share in sorted(flags.metal_streak_shares.items())]
    rows.append([
        f"Brainstem dark band >= {BAND_DEPTH_HU:g} HU (provisional) on slices",
        ", ".join(str(index + 1) for index in flags.band_slices) or "none",
    ])
    if flags.band_depths:
        rows.append(["Posterior-fossa slice", "Band depth (HU)"])
        rows += [[str(index + 1), f"{depth:.2f}"] for index, depth in sorted(flags.band_depths.items())]
    return rows


def _basal_ganglia_rows(slices: dict[int, BasalGangliaSlice] | None) -> list[list[str]]:
    """Basal-ganglia visibility per slice - provisional, not part of the verdict."""
    if not slices:
        return []
    low, high = PLAUSIBLE_DIFFERENCE_HU
    rows = [
        [],
        ["Basal ganglia visibility (provisional, not part of the verdict)"],
        ["Slice", "Visibility", "Border contrast", "Lentiform - internal capsule (HU)",
         "Caudate - internal capsule (HU)", f"Plausible ({low:g}-{high:g} HU)"],
    ]
    for index, s in sorted(slices.items()):
        rows.append([
            str(index + 1), s.level, f"{s.border:.3f}",
            *(f"{s.differences[name]:.2f}" if name in s.differences else "" for name in ("Lentiform Nucleus", "Caudate Nucleus")),
            "yes" if s.plausible else "no",
        ])
    return rows


def _plausibility_rows(plausibility: SegmentationPlausibility | None) -> list[list[str]]:
    """Study-level segmentation check - provisional, not part of the verdict."""
    if plausibility is None or plausibility.slices == 0:
        return []
    return [
        [],
        ["Segmentation check (provisional, not part of the verdict)"],
        ["Segmentation", "may be unreliable - check the overlays" if plausibility.unreliable else "looks consistent"],
        ["Fragmentation (pieces per slice, median)", f"{plausibility.fragmentation:.1f}", f"limit {FRAGMENTATION_LIMIT:g}"],
        ["Most fragmented slices", ", ".join(str(index + 1) for index in plausibility.worst_slices)],
        ["Lobe islands", str(plausibility.lobe_islands)],
    ]


def _study_summary_rows(results: StudyResults, context: StudyExportContext) -> list[list[str]]:
    """Study context, then per structure (ROI pair) its verdict and metric
    averages - structures are never averaged together."""
    all_results = [result for by_slice in results.values() for result in by_slice.values()]
    analysed = sorted({index for by_slice in results.values() for index in by_slice})
    rows = [
        ["Series", context.series_label],
        ["Slices in series", str(context.slice_count)],
        ["Slices analysed", str(len(analysed))],
        ["Failed slices", ", ".join(str(item) for item in context.failed_slices)],
        ["Pixel spacing (mm)", f"{context.pixel_spacing_mm:.4f}"],
        ["QA recipe", _recipe_label(all_results)],
        ["Acquisition date", context.acquisition_date or ""],
    ]
    physical = _is_physical(all_results)
    if physical:
        rows += _head_ct_verdict_rows(context.head_ct_verdict, len(results.get("Falx", {})))
    rows += _artefact_rows(context.artefacts)
    rows += _basal_ganglia_rows(context.basal_ganglia)
    rows += _plausibility_rows(context.segmentation_plausibility)
    for structure, by_slice in results.items():
        summary = summarize_metrics(by_slice.values())
        rows += [[], [structure]]
        if not physical:  # legacy recipe: majority vote of the chest-CT SVM
            verdict = study_verdict(by_slice.values())
            rows += [
                ["Study verdict", verdict.verdict if verdict else ""],
                ["Acceptable slices", f"{verdict.acceptable} of {verdict.total}" if verdict else ""],
            ]
        rows.append(["Metric", "Mean", "SD", "Slices"])
        for name, label in _STUDY_METRIC_LABELS.items():
            if physical and name == "decision_value":
                continue
            metric = summary.get(name)
            if metric is not None:
                rows.append([label, f"{metric.mean:.6g}", f"{metric.std:.6g}", str(metric.count)])
    return rows


def _study_slice_rows(
    results: StudyResults,
    head_ct_verdict: HeadCtVerdict | None = None,
    artefacts: ArtefactFlags | None = None,
    basal_ganglia: dict[int, BasalGangliaSlice] | None = None,
) -> list[list[str]]:
    """Header, then one row per slice and structure (1-based slice numbers,
    in slice order). Physical recipe: plus the head-CT model's indicative
    slice score and noise-outlier flag for falx slices. Last column: possible
    artefacts on that slice."""
    physical = _is_physical(r for by_slice in results.values() for r in by_slice.values())
    rows = [[*(_SLICE_TABLE_HEADER_PHYSICAL if physical else _SLICE_TABLE_HEADER), "Possible artefact",
             "Basal ganglia (provisional)"]]
    v = head_ct_verdict
    jobs = sorted(
        ((index, position, structure) for position, (structure, by_slice) in enumerate(results.items()) for index in by_slice),
    )
    for index, _position, structure in jobs:
        result = results[structure][index]
        extra = []
        if physical:
            falx = structure == "Falx" and v is not None and index in v.slice_ids
            extra = [
                f"{v.slice_probabilities[v.slice_ids.index(index)]:.4g}" if falx else "",
                "yes" if falx and index in v.outlier_slices else "",
            ]
        rows.append(
            [
                str(index + 1),
                structure,
                f"{result.mtf_area:.6g}",
                f"{result.nps_roi2.var_nps:.6g}",
                f"{result.ratio.ratio_lf_hf:.6g}",
                f"{result.contrast_roi1:.6g}",
                f"{result.decision_value:.6g}",
                "Acceptable" if result.predicted_label == 1 else "Unacceptable",
                *(f"{value:.6g}" for value in result.feature_vector),
                *extra,
                "; ".join(artefacts.on_slice(index)) if artefacts is not None else "",
                basal_ganglia[index].level if basal_ganglia and index in basal_ganglia else "",
            ]
        )
    return rows


def export_study_csv(
    results: StudyResults,
    context: StudyExportContext,
    summary_path: str | Path,
    slices_path: str | Path,
) -> None:
    """Two CSV files (CSV has no sheets): the study summary and the per-slice
    table. `results`: structure name -> {0-based slice index: result}.
    """
    for path, rows in (
        (summary_path, _study_summary_rows(results, context)),
        (slices_path, _study_slice_rows(results, context.head_ct_verdict, context.artefacts, context.basal_ganglia)),
    ):
        with open(path, "w", newline="", encoding="ascii", errors="replace") as f:
            csv.writer(f).writerows(rows)


def export_study_excel(results: StudyResults, context: StudyExportContext, path: str | Path) -> None:
    """A readable workbook: "Summary" (study, result in words, measurements,
    technical details below) and "Per slice" (one row per slice and ROI pair).
    `results`: structure name -> {0-based slice index: result}.
    """
    workbook = Workbook()
    _write_study_report(_Report(workbook.active, "Summary"), results, context)
    _write_slice_table(workbook.create_sheet("Per slice"), results, context)
    workbook.save(str(path))


# --- Excel report layout ---

_REPORT_TITLE = "BICTool image-quality report"
_REPORT_COLUMNS = 8  # columns a report sheet uses (widths, the headline band)
_TONE_FILLS = {"good": "C6EFCE", "bad": "FFC7CE", "neutral": "FFEB9C"}
# Study metric -> (plain name, unit, Excel number format) for people reading the report.
_PLAIN_METRICS = {
    "mtf_area": ("Sharpness (MTF area)", "1/mm", "0.000"),
    "nps_variance": ("Noise (NPS variance)", "HU²·mm²", "0.0"),
    "ps_mtf_ratio": ("Structure-to-noise (PS/MTF ratio, LF/HF)", "", "0.00"),
    "contrast": ("Contrast", "HU", "0.00"),
    "decision_value": ("Decision value (chest-CT classifier)", "", "0.000"),
}
_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
_MUTED = "6B7480"


def iso_date(dicom_date: str | None) -> str:
    """DICOM `YYYYMMDD` -> `YYYY-MM-DD`; anything else unchanged ("" if missing)."""
    if dicom_date and len(dicom_date) == 8 and dicom_date.isdigit():
        return f"{dicom_date[:4]}-{dicom_date[4:6]}-{dicom_date[6:]}"
    return dicom_date or ""


def _method_label(results) -> str:
    return "Physical QA recipe (head CT)" if _is_physical(results) else "Legacy recipe (Octave port, chest-CT classifier)"


def _excel_value(value, digits: int | None = None):
    """Numbers written as text by the shared row builders become real numbers
    in Excel, rounded to `digits` significant digits if given."""
    if isinstance(value, str) and _NUMBER.fullmatch(value.strip()):
        if "." not in value and "e" not in value.lower():
            return int(value)
        value = float(value)
    if digits is not None and isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def _is_number(value) -> bool:
    return isinstance(_excel_value(value), (int, float))


class _Report:
    """Writes a report sheet top to bottom: title, headings, label/value rows."""

    def __init__(self, sheet, title: str) -> None:
        self.sheet = sheet
        self.sheet.title = title
        self.sheet.sheet_view.showGridLines = False
        self.next_row = 1

    def _put(self, values, font=None, fill=None, formats=None, digits=None, wrap=False) -> None:
        for column, value in enumerate(values, start=1):
            if value is None or value == "":
                continue
            cell = self.sheet.cell(row=self.next_row, column=column, value=_excel_value(value, digits))
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=wrap)
            if font is not None:
                cell.font = font
            if formats and column in formats and isinstance(cell.value, float):
                cell.number_format = formats[column]
        if fill is not None:  # the whole band, not only the text's cell
            for column in range(1, _REPORT_COLUMNS + 1):
                self.sheet.cell(row=self.next_row, column=column).fill = PatternFill("solid", start_color=fill)
        self.next_row += 1

    def title(self, text: str) -> None:
        self._put([text], font=Font(bold=True, size=15))
        self._put([f"Exported {date.today():%Y-%m-%d} with BICTool {__version__}. Contains no patient identifiers."],
                  font=Font(italic=True, color=_MUTED))
        self.blank()

    def heading(self, text: str) -> None:
        self._put([text], font=Font(bold=True, size=12))

    def subheading(self, text: str) -> None:
        self._put([text], font=Font(bold=True))

    def header(self, *values) -> None:
        self._put(values, font=Font(bold=True, color=_MUTED), wrap=len(values) > 2)

    def row(self, *values, formats=None) -> None:
        self._put(values, formats=formats)

    def note(self, text: str) -> None:
        self._put([text], font=Font(italic=True, color=_MUTED))

    def headline(self, text: str, tone: str) -> None:
        self._put([text], font=Font(bold=True, size=13), fill=_TONE_FILLS.get(tone, _TONE_FILLS["neutral"]))

    def blank(self) -> None:
        self.next_row += 1

    def study_block(self, pairs) -> None:
        self.heading("Study")
        for label, value in pairs:
            self._put([label, value])
            self.sheet.cell(row=self.next_row - 1, column=1).font = Font(color=_MUTED)
        self.blank()

    def rows(self, rows) -> None:
        """Rows from the shared CSV builders: one-cell rows are sub-headings,
        rows of four or more labels are table headers; numbers rounded."""
        for row in rows:
            if not row:
                self.blank()
            elif len(row) == 1:
                self.subheading(row[0])
            elif len(row) >= 4 and not any(_is_number(value) for value in row):
                self.header(*row)
            else:
                self._put(row, digits=4)

    def widths(self, *widths) -> None:
        for column, width in enumerate(widths, start=1):
            self.sheet.column_dimensions[self.sheet.cell(row=1, column=column).column_letter].width = width


def _write_study_report(report: _Report, results: StudyResults, context: StudyExportContext) -> None:
    all_results = [result for by_slice in results.values() for result in by_slice.values()]
    analysed = sorted({index for by_slice in results.values() for index in by_slice})
    physical = _is_physical(all_results)
    per_pair = ", ".join(f"{structure} {len(by_slice)}" for structure, by_slice in results.items())
    report.title(_REPORT_TITLE)
    report.study_block([
        ("Series", context.series_label),
        ("Acquisition date", iso_date(context.acquisition_date)),
        ("Slices analysed", f"{len(analysed)} of {context.slice_count} ({per_pair})"),
        *([("Failed slices", ", ".join(str(item) for item in context.failed_slices))] if context.failed_slices else []),
        ("Pixel spacing", f"{context.pixel_spacing_mm:.3f} mm"),
        ("QA method", _method_label(all_results)),
    ])

    report.heading("Result")
    headline, tone, lines = context.headline, context.headline_tone, list(context.summary_lines)
    if not headline:  # no text from the app: the bare verdict
        headline, tone, lines = _bare_result(results, context, physical)
    report.headline(headline, tone)
    for line in lines:
        report.row(line)
    if physical:
        report.note(PROVISIONAL_NOTE)
    report.blank()

    report.heading("Measurements (mean and SD over the slices)")
    report.header("Measurement", "Unit", *(f"{structure} {stat}" for structure in results for stat in ("mean", "SD")))
    summaries = {structure: summarize_metrics(by_slice.values()) for structure, by_slice in results.items()}
    for name, (label, unit, number_format) in _PLAIN_METRICS.items():
        if physical and name == "decision_value":
            continue
        cells, formats = [label, unit], {}
        for structure in results:
            metric = summaries[structure].get(name)
            cells += [metric.mean, metric.std] if metric is not None else ["", ""]
            formats.update({len(cells) - 1: number_format, len(cells): number_format})
        report.row(*cells, formats=formats)
    report.blank()

    report.heading("Technical details")
    report.note("The numbers behind the result, for experts.")
    if physical:
        report.rows(_head_ct_verdict_rows(context.head_ct_verdict, len(results.get("Falx", {})))[1:])
    report.rows(_artefact_rows(context.artefacts))
    report.rows(_basal_ganglia_rows(context.basal_ganglia))
    report.rows(_plausibility_rows(context.segmentation_plausibility))
    report.widths(44, 16, 16, 16, 16, 16, 16, 16)


def _bare_result(results: StudyResults, context: StudyExportContext, physical: bool) -> tuple[str, str, list[str]]:
    """Headline without the app's wording (e.g. exports outside the GUI)."""
    if physical:
        v = context.head_ct_verdict
        if v is None:
            return "Study verdict: not available", "neutral", []
        tone = {"Acceptable": "good", "Not acceptable": "bad"}.get(v.verdict, "neutral")
        return f"Study verdict: {v.verdict}", tone, []
    lines = []
    for structure, by_slice in results.items():
        verdict = study_verdict(by_slice.values())
        if verdict is not None:
            lines.append(f"{structure}: {verdict.verdict} - {verdict.acceptable} of {verdict.total} slice(s) acceptable")
    return "Study verdict (legacy majority vote)", "neutral", lines


def _write_slice_table(sheet, results: StudyResults, context: StudyExportContext) -> None:
    """One row per slice and ROI pair - plain names, numbers as numbers, filterable."""
    physical = _is_physical(r for by_slice in results.values() for r in by_slice.values())
    v, artefacts, basal = context.head_ct_verdict, context.artefacts, context.basal_ganglia
    columns = [("Slice", None), ("ROI pair", None)] + [
        (f"{label} ({unit})" if unit else label, number_format)
        for name, (label, unit, number_format) in _PLAIN_METRICS.items() if name != "decision_value"
    ]
    if physical:
        columns += [("Quality hint (0-1, indicative)", "0.00"), ("Noise outlier", None)]
    else:
        columns += [("Verdict (chest-CT classifier)", None), ("Decision value", "0.000")]
    columns += [("Possible artefact", None), ("Basal ganglia (provisional)", None)]
    sheet.title = "Per slice"
    sheet.append([name for name, _format in columns])
    jobs = sorted((index, position, structure)
                  for position, (structure, by_slice) in enumerate(results.items()) for index in by_slice)
    for index, _position, structure in jobs:
        result = results[structure][index]
        row = [index + 1, structure, result.mtf_area, result.nps_roi2.var_nps, result.ratio.ratio_lf_hf,
               result.contrast_roi1]
        if physical:
            falx = structure == "Falx" and v is not None and index in v.slice_ids
            row += [float(v.slice_probabilities[v.slice_ids.index(index)]) if falx else None,
                    "yes" if falx and index in v.outlier_slices else None]
        else:
            row += ["Acceptable" if result.predicted_label == 1 else "Unacceptable", result.decision_value]
        row += ["; ".join(artefacts.on_slice(index)) if artefacts is not None else None,
                basal[index].level if basal and index in basal else None]
        sheet.append([float(value) if hasattr(value, "dtype") else value for value in row])
    for column, (name, number_format) in enumerate(columns, start=1):
        header = sheet.cell(row=1, column=column)
        header.font = Font(bold=True)
        header.alignment = Alignment(horizontal="center", vertical="top", wrap_text=True)
        sheet.column_dimensions[header.column_letter].width = 8 if column == 1 else 16
        for (cell,) in sheet.iter_rows(min_row=2, min_col=column, max_col=column):
            cell.alignment = Alignment(horizontal="center")
            if number_format:
                cell.number_format = number_format
    sheet.row_dimensions[1].height = 45
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = sheet.dimensions


# --- Dataset export (many volumes, ROI1 structures kept apart) ---

_UNTRUSTED_NOTE = (
    "Columns marked 'legacy chest SVM, untrusted' and the decision value: the SVM classifier was trained on "
    "chest CT and is not validated for head CT. The study verdict (physical recipe) is the head-CT model's, "
    "on each study's Dural Folds (falx) row; metal streaks in the brain make it Not assessable (metal rule). "
    + PROVISIONAL_NOTE
)
# Study-level head-CT verdict columns of the dataset summary (falx row only).
_DATASET_VERDICT_HEADER = [
    "Study verdict (head-CT model)",
    "Metal rule: streaks in the brain on slices",
    "Model verdict before the metal rule",
    "P(acceptable)",
    "P 95 % CI low",
    "P 95 % CI high",
    "Falx slices used",
    "Outside good range (towards not acceptable)",
    "Artefact indicator (provisional)",
    "Noise outlier slices",
    "Model",
]
# Possible-artefact hints, per study (on its first structure row).
_DATASET_ARTEFACT_HEADER = [
    "Metal slices",
    "Brainstem dark band slices (provisional)",
    "Basal ganglia visible slices (provisional)",
    "Basal ganglia label not plausible slices",
    "Segmentation fragmentation (provisional)",
    "Segmentation may be unreliable (provisional)",
]
_DATASET_SUMMARY_HEADER = [
    "Dataset",
    "Volume",
    "Structure",
    "Slices",
    "Segmented",
    "Analysed",
    "Failed",
    "Failed slices",
    *[f"{label} {stat}" for label in _STUDY_METRIC_LABELS.values() for stat in ("mean", "SD")],
    f"Verdict{_LEGACY_SVM}",
    f"Acceptable slices{_LEGACY_SVM}",
    *_DATASET_VERDICT_HEADER,
    *_DATASET_ARTEFACT_HEADER,
    "Sheet",
    "Note",
]
_DATASET_SLICE_HEADER = [
    "Slice",
    "ROI1 x",
    "ROI1 y",
    "ROI1 size",
    "ROI2 x",
    "ROI2 y",
    "ROI2 size",
    "ROI2 region",
    "MTF seed",
    *_STUDY_METRIC_LABELS.values(),
    "NPS f_av (1/mm)",
    "NPS f_peak (1/mm)",
    f"Verdict{_LEGACY_SVM}",
    *_FEATURE_NAMES,
    "Slice score (indicative)",
    "Noise outlier",
]


def _recipe_label(results) -> str:
    """The QA recipe(s) behind `results`, with a pointer to their documentation."""
    recipes = sorted({getattr(result, "recipe", "octave") for result in results})
    if not recipes:
        return ""
    return ", ".join(recipes) + " (see docs/qa_method/physical_qa_recipe.md)"


def _all_results(volumes):
    return [result for volume in volumes for qa in volume.structures.values() for result in qa.results.values()]


def _sheet_names(volumes) -> list[str]:
    """Unique Excel sheet names (<= 31 chars, none of Excel's forbidden
    characters) per volume, in order: number, dataset and the study name
    without its leading timestamp.
    """
    names, used = [], {"Summary"}
    for number, volume in enumerate(volumes, start=1):
        label = re.sub(r"^\d{8}_\d{6}(?:_\d+)?[\s_-]*", "", volume.name) or volume.name
        base = re.sub(r"[\[\]:*?/\\]", "_", f"{number} {volume.group} {label}")[:31].strip()
        name, suffix = base, 2
        while name in used:
            name = f"{base[:28]}~{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)
    return names


def _artefact_text(artefact) -> str:
    if artefact is None:
        return "not available (too few falx slices)"
    level = (
        "confirmed" if artefact.flagged
        else "above good range, not confirmed" if artefact.above_good_range
        else "within good range"
    )
    return f"{artefact.value:.2f} [{artefact.low:.2f}, {artefact.high:.2f}] - {level}"


def _dataset_verdict_cells(volume, structure: str) -> list:
    """The head-CT verdict columns: filled on the falx (Dural Folds) row only."""
    empty = [""] * len(_DATASET_VERDICT_HEADER)
    qa = volume.structures.get(structure)
    if structure != DURAL_FOLDS or qa is None:
        return empty
    if not _is_physical(_all_results([volume])):
        return ["not available (legacy recipe)", *empty[1:]]
    v = getattr(volume, "head_ct_verdict", None)
    if v is None:
        return [f"not available - needs at least {MIN_SLICES} measured falx slices", *empty[1:]]
    return [
        v.verdict,
        _listed(v.metal_slices),
        v.model_verdict or v.verdict,
        round(v.probability, 4),
        round(v.probability_low, 4),
        round(v.probability_high, 4),
        v.n_slices,
        "; ".join(FEATURE_LABELS[f.name] for f in v.reasons),
        _artefact_text(v.artefact),
        ", ".join(str(index + 1) for index in v.outlier_slices),
        v.model_version,
    ]


def _dataset_artefact_cells(volume, first_row: bool) -> list:
    flags = getattr(volume, "artefact_flags", None)
    basal = getattr(volume, "basal_ganglia", None)
    plausibility = getattr(volume, "segmentation_plausibility", None)
    if not first_row or (flags is None and basal is None and plausibility is None):
        return [""] * len(_DATASET_ARTEFACT_HEADER)

    def listed(indices):
        return ", ".join(str(index + 1) for index in indices) or "none"

    return [
        listed(flags.metal_slices) if flags is not None else "",
        listed(flags.band_slices) if flags is not None else "",
        listed(sorted(i for i, s in basal.items() if s.visible)) if basal is not None else "",
        listed(sorted(i for i, s in basal.items() if s.level == NOT_PLAUSIBLE)) if basal is not None else "",
        f"{plausibility.fragmentation:.1f}" if plausibility is not None and plausibility.slices else "",
        ("yes" if plausibility.unreliable else "no") if plausibility is not None and plausibility.slices else "",
    ]


def _dataset_summary_rows(volumes, sheet_names) -> list[list]:
    rows = []
    for volume, sheet in zip(volumes, sheet_names):
        if volume.error is not None:
            rows.append([volume.group, volume.name, "", volume.slice_count, volume.segmented_slices,
                         *[""] * (len(_DATASET_SUMMARY_HEADER) - 7), sheet, volume.error])
            continue
        for position, (structure, qa) in enumerate(volume.structures.items()):
            summary = summarize_metrics(qa.results.values())
            verdict = study_verdict(qa.results.values())
            notes = [] if qa.results or qa.errors else [f"No {structure} ROI found on any slice"]
            skipped = getattr(qa, "skipped", [])
            if skipped:
                notes.append(
                    f"Skipped (< {MIN_BRAIN_FRACTION:.0%} of largest brain area): "
                    + ", ".join(str(index + 1) for index in skipped)
                )
            note = "; ".join(notes)
            stats = []
            for name in _STUDY_METRIC_LABELS:
                metric = summary.get(name)
                stats += [metric.mean, metric.std] if metric else ["", ""]
            rows.append(
                [
                    volume.group,
                    volume.name,
                    structure,
                    volume.slice_count,
                    volume.segmented_slices,
                    len(qa.results),
                    len(qa.errors),
                    ", ".join(str(index + 1) for index in sorted(qa.errors)),
                    *stats,
                    verdict.verdict if verdict else "",
                    f"{verdict.acceptable} of {verdict.total}" if verdict else "",
                    *_dataset_verdict_cells(volume, structure),
                    *_dataset_artefact_cells(volume, position == 0),
                    sheet,
                    note,
                ]
            )
    return rows


def _dataset_slice_rows(qa, head_ct_verdict=None) -> list[list]:
    """One row per successfully analysed slice of one structure (1-based
    slice numbers). Falx slices also get the head-CT model's indicative
    slice score and noise-outlier flag."""
    v = head_ct_verdict if qa.structure == DURAL_FOLDS else None
    rows = []
    for index in sorted(qa.results):
        result = qa.results[index]
        roi1, roi2 = qa.rois[index]
        seed = getattr(result, "mtf_seed", None)
        rows.append(
            [
                index + 1,
                roi1.col0,
                roi1.row0,
                roi1.size,
                roi2.col0,
                roi2.row0,
                roi2.size,
                roi2.structure,
                f"{seed[0]},{seed[1]}" if seed else "",
                result.mtf_area,
                result.nps_roi2.var_nps,
                result.ratio.ratio_lf_hf,
                result.contrast_roi1,
                result.decision_value,
                getattr(result, "nps_f_av", float("nan")),
                getattr(result, "nps_f_peak", float("nan")),
                "Acceptable" if result.predicted_label == 1 else "Unacceptable",
                *(float(value) for value in result.feature_vector),
                float(v.slice_probabilities[v.slice_ids.index(index)]) if v and index in v.slice_ids else "",
                ("yes" if index in v.outlier_slices else "") if v else "",
            ]
        )
    return rows


def export_dataset_excel(volumes, path: str | Path) -> None:
    """"Summary" sheet (one row per volume and ROI1 structure - structures
    are never averaged together) + one sheet per volume with each
    structure's averages and per-slice table. `volumes` are
    `domain.results.volume_qa.VolumeQa`.
    """
    sheet_names = _sheet_names(volumes)
    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    summary_sheet.append(["BICTool dataset QA - automatic ROIs, averages per volume and ROI1 structure"])
    summary_sheet.append([_UNTRUSTED_NOTE])
    summary_sheet.append([f"QA recipe: {_recipe_label(_all_results(volumes))}"])
    summary_sheet.append([])
    summary_sheet.append(_DATASET_SUMMARY_HEADER)
    for row in _dataset_summary_rows(volumes, sheet_names):
        summary_sheet.append(row)
    summary_sheet.freeze_panes = "D6"
    for column, header in enumerate(_DATASET_SUMMARY_HEADER, start=1):
        summary_sheet.column_dimensions[summary_sheet.cell(5, column).column_letter].width = max(len(header) + 2, 10)
    summary_sheet.column_dimensions["B"].width = 40

    for volume, sheet_name in zip(volumes, sheet_names):
        sheet = workbook.create_sheet(sheet_name)
        for row in (
            ["Dataset", volume.group],
            ["Volume", volume.name],
            ["Series", volume.series_label],
            ["Slices", volume.slice_count],
            ["Segmented slices", volume.segmented_slices],
            ["Pixel spacing (mm)", volume.pixel_spacing_mm],
            ["Acquisition date", volume.acquisition_date or ""],
        ):
            sheet.append(row)
        if volume.error is not None:
            sheet.append(["Error", volume.error])
            continue
        sheet.append([_UNTRUSTED_NOTE])
        sheet.append(["QA recipe", _recipe_label(_all_results([volume]))])
        if _is_physical(_all_results([volume])):
            falx = volume.structures.get(DURAL_FOLDS)
            for row in _head_ct_verdict_rows(getattr(volume, "head_ct_verdict", None), len(falx.results) if falx else 0):
                sheet.append(row)
        for row in _artefact_rows(getattr(volume, "artefact_flags", None)):
            sheet.append(row)
        for row in _basal_ganglia_rows(getattr(volume, "basal_ganglia", None)):
            sheet.append(row)
        for structure, qa in volume.structures.items():
            summary = summarize_metrics(qa.results.values())
            verdict = study_verdict(qa.results.values())
            sheet.append([])
            sheet.append([structure])
            if not qa.results and not qa.errors:
                sheet.append([f"No {structure} ROI found on any slice"])
                continue
            sheet.append(["Analysed slices", len(qa.results)])
            sheet.append(["Failed slices", ", ".join(str(index + 1) for index in sorted(qa.errors))])
            sheet.append([
                f"Verdict{_LEGACY_SVM}",
                f"{verdict.verdict} - {verdict.acceptable} of {verdict.total} acceptable" if verdict else "",
            ])
            sheet.append(["Metric", "Mean", "SD", "Slices"])
            for name, label in _STUDY_METRIC_LABELS.items():
                metric = summary.get(name)
                if metric is not None:
                    sheet.append([label, metric.mean, metric.std, metric.count])
            sheet.append([])
            sheet.append(_DATASET_SLICE_HEADER)
            for row in _dataset_slice_rows(qa, getattr(volume, "head_ct_verdict", None)):
                sheet.append(row)
        sheet.column_dimensions["A"].width = 26
    workbook.save(str(path))


def export_dataset_csv(volumes, summary_path: str | Path, slices_path: str | Path) -> None:
    """Two CSV files: the Summary sheet's table, and every analysed slice of
    every volume and structure in one long table (Dataset/Volume/Structure
    columns first).
    """
    sheet_names = _sheet_names(volumes)
    with open(summary_path, "w", newline="", encoding="ascii", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerow(_DATASET_SUMMARY_HEADER)
        writer.writerows(_dataset_summary_rows(volumes, sheet_names))
    with open(slices_path, "w", newline="", encoding="ascii", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Volume", "Structure", *_DATASET_SLICE_HEADER])
        for volume in volumes:
            for structure, qa in volume.structures.items():
                for row in _dataset_slice_rows(qa, getattr(volume, "head_ct_verdict", None)):
                    writer.writerow([volume.group, volume.name, structure, *row])
