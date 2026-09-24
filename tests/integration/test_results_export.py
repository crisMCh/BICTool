import csv
from types import SimpleNamespace

import openpyxl

from bictool.integration.results_export import ExportContext, export_metrics_csv, export_metrics_excel


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
