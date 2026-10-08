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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook

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
    rows += [
        ["Verdict", verdict.verdict],
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


def _study_slice_rows(results: StudyResults, head_ct_verdict: HeadCtVerdict | None = None) -> list[list[str]]:
    """Header, then one row per slice and structure (1-based slice numbers,
    in slice order). Physical recipe: plus the head-CT model's indicative
    slice score and noise-outlier flag for falx slices."""
    physical = _is_physical(r for by_slice in results.values() for r in by_slice.values())
    rows = [list(_SLICE_TABLE_HEADER_PHYSICAL if physical else _SLICE_TABLE_HEADER)]
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
        (slices_path, _study_slice_rows(results, context.head_ct_verdict)),
    ):
        with open(path, "w", newline="", encoding="ascii", errors="replace") as f:
            csv.writer(f).writerows(rows)


def export_study_excel(results: StudyResults, context: StudyExportContext, path: str | Path) -> None:
    """One workbook: "Study summary" and "Per slice" sheets. `results`:
    structure name -> {0-based slice index: result}.
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
    for row in _study_slice_rows(results, context.head_ct_verdict):
        slices_sheet.append(row)
    for (header_cell,) in slices_sheet.iter_cols(min_row=1, max_row=1):
        slices_sheet.column_dimensions[header_cell.column_letter].width = max(len(str(header_cell.value)) + 2, 10)
    workbook.save(str(path))


# --- Dataset export (many volumes, ROI1 structures kept apart) ---

_UNTRUSTED_NOTE = (
    "Columns marked 'legacy chest SVM, untrusted' and the decision value: the SVM classifier was trained on "
    "chest CT and is not validated for head CT. The study verdict (physical recipe) is the head-CT model's, "
    "on each study's Dural Folds (falx) row. " + PROVISIONAL_NOTE
)
# Study-level head-CT verdict columns of the dataset summary (falx row only).
_DATASET_VERDICT_HEADER = [
    "Study verdict (head-CT model)",
    "P(acceptable)",
    "P 95 % CI low",
    "P 95 % CI high",
    "Falx slices used",
    "Outside good range (towards not acceptable)",
    "Artefact indicator (provisional)",
    "Noise outlier slices",
    "Model",
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
        round(v.probability, 4),
        round(v.probability_low, 4),
        round(v.probability_high, 4),
        v.n_slices,
        "; ".join(FEATURE_LABELS[f.name] for f in v.reasons),
        _artefact_text(v.artefact),
        ", ".join(str(index + 1) for index in v.outlier_slices),
        v.model_version,
    ]


def _dataset_summary_rows(volumes, sheet_names) -> list[list]:
    rows = []
    for volume, sheet in zip(volumes, sheet_names):
        if volume.error is not None:
            rows.append([volume.group, volume.name, "", volume.slice_count, volume.segmented_slices,
                         *[""] * (len(_DATASET_SUMMARY_HEADER) - 7), sheet, volume.error])
            continue
        for structure, qa in volume.structures.items():
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
