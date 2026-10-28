from bictool.presentation.viewmodels.display_text import format_dicom_date, format_slice_ranges, safe_file_name


def test_slice_ranges_are_one_based_and_collapse_runs():
    assert format_slice_ranges([19, 20, 21, 30]) == "20–22, 31"
    assert format_slice_ranges([0]) == "1"
    assert format_slice_ranges([0, 2, 3, 5]) == "1, 3–4, 6"
    assert format_slice_ranges([]) == ""


def test_dicom_dates_are_reformatted_and_odd_values_pass_through():
    assert format_dicom_date("20250911") == "11-September-2025"
    assert format_dicom_date("2025-09-11") == "2025-09-11"
    assert format_dicom_date("20251399") == "20251399"
    assert format_dicom_date(None) == "—"
    assert format_dicom_date("") == "—"


def test_file_names_stay_readable_and_drop_forbidden_characters():
    assert safe_file_name("BICTool study 2026-01-31 CCT nativ 4,00 Hr44") == "BICTool study 2026-01-31 CCT nativ 4,00 Hr44"
    assert safe_file_name('Head: 4/5 mm  "axial"?') == "Head- 4-5 mm -axial--"
    assert safe_file_name("  . ") == "BICTool export"
