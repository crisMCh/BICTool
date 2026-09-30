import csv
from types import SimpleNamespace

import openpyxl
import pytest

from bictool.integration.results_export import (
    ExportContext,
    StudyExportContext,
    export_metrics_csv,
    export_metrics_excel,
    export_study_csv,
    export_study_excel,
)


def _fake_result() -> SimpleNamespace:
    return SimpleNamespace(
        mtf_area=0.418,
        nps_roi2=SimpleNamespace(var_nps=0.00291),
        ratio=SimpleNamespace(ratio_lf_hf=1.86),
        contrast_roi1=812.3,
        decision_value=1.42,
        predicted_label=1,
        feature_vector=[0.418, 0.00142, 0.00187, 12.4, 1.86, -0.031, 0.34],
    )


def _fake_context() -> ExportContext:
    return ExportContext(
        series_label="CT · 467 slice(s) · 512×512",
        slice_number=12,
        slice_count=467,
        pixel_spacing_mm=0.4880,
        acquisition_date="20260101",
    )


def test_export_metrics_csv_writes_the_expected_rows(tmp_path):
    path = tmp_path / "result.csv"

    export_metrics_csv(_fake_result(), _fake_context(), path)

    with open(path, newline="", encoding="ascii") as f:
        rows = [tuple(row) for row in csv.reader(f)]

    assert ("Slice", "12/467") in rows
    assert ("MTF area (mm^-1)", "0.4180") in rows
    assert ("Verdict", "Acceptable") in rows
    assert ("mtf_area", "0.418") in rows


def test_export_metrics_csv_never_contains_patient_identifiers(tmp_path):
    # NFR-06/GDPR - exports must stay de-identified even though the live GUI
    # shows patient name/ID.
    path = tmp_path / "result.csv"

    export_metrics_csv(_fake_result(), _fake_context(), path)

    text = path.read_text(encoding="ascii")
    assert "patient" not in text.lower()


def test_export_metrics_csv_marks_unacceptable_verdict(tmp_path):
    path = tmp_path / "result.csv"
    result = _fake_result()
    result.predicted_label = 0

    export_metrics_csv(result, _fake_context(), path)

    with open(path, newline="", encoding="ascii") as f:
        rows = [tuple(row) for row in csv.reader(f)]
    assert ("Verdict", "Unacceptable") in rows


def test_export_metrics_excel_writes_the_expected_rows(tmp_path):
    path = tmp_path / "result.xlsx"

    export_metrics_excel(_fake_result(), _fake_context(), path)

    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    rows = [tuple(cell.value for cell in row) for row in sheet.iter_rows()]

    assert ("Slice", "12/467") in rows
    assert ("Verdict", "Acceptable") in rows
    assert any(r[0] == "MTF area (mm^-1)" for r in rows)


def _study_results() -> dict:
    first = _fake_result()
    second = _fake_result()
    second.mtf_area = 0.218
    second.predicted_label = 0
    third = _fake_result()
    return {11: first, 12: second, 20: third}  # 0-based slice indices


def _study_context() -> StudyExportContext:
    return StudyExportContext(
        series_label="CT head",
        slice_count=467,
        pixel_spacing_mm=0.4880,
        acquisition_date="20260101",
        failed_slices=[14, 30],
    )


def _read_csv(path) -> list[list[str]]:
    with open(path, newline="", encoding="ascii") as f:
        return list(csv.reader(f))


def test_export_study_csv_writes_the_averages_and_a_row_per_slice(tmp_path):
    summary_path, slices_path = tmp_path / "summary.csv", tmp_path / "slices.csv"

    export_study_csv(_study_results(), _study_context(), summary_path, slices_path)

    summary = _read_csv(summary_path)
    assert ["Study verdict", "Acceptable"] in summary
    assert ["Acceptable slices", "2 of 3"] in summary
    assert ["Failed slices", "14, 30"] in summary
    mtf_row = next(row for row in summary if row and row[0] == "MTF area (mm^-1)")
    assert float(mtf_row[1]) == pytest.approx((0.418 + 0.218 + 0.418) / 3)
    assert mtf_row[3] == "3"

    slices = _read_csv(slices_path)
    assert slices[0][:2] == ["Slice", "MTF area (mm^-1)"]
    assert [row[0] for row in slices[1:]] == ["12", "13", "21"]  # 1-based, in order
    assert slices[2][6] == "Unacceptable"


def test_export_study_excel_has_a_summary_and_a_per_slice_sheet(tmp_path):
    path = tmp_path / "study.xlsx"

    export_study_excel(_study_results(), _study_context(), path)

    workbook = openpyxl.load_workbook(path)
    assert workbook.sheetnames == ["Study summary", "Per slice"]
    summary = list(workbook["Study summary"].iter_rows(values_only=True))
    assert summary[7][:2] == ("Study verdict", "Acceptable")
    per_slice = list(workbook["Per slice"].iter_rows(values_only=True))
    assert len(per_slice) == 4  # header + 3 slices


def test_export_study_never_contains_patient_identifiers(tmp_path):
    summary_path, slices_path = tmp_path / "summary.csv", tmp_path / "slices.csv"

    export_study_csv(_study_results(), _study_context(), summary_path, slices_path)

    text = summary_path.read_text(encoding="ascii") + slices_path.read_text(encoding="ascii")
    assert "patient" not in text.lower()
