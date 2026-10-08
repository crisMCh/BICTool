from bictool.presentation.viewmodels.display_text import format_dicom_date, format_slice_ranges, safe_file_stem


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


def test_file_stem_replaces_symbols_and_collapses_underscores():
    assert safe_file_stem("CT · 1 slice(s) · 512×434") == "CT_1_slice_s_512_434"
    assert safe_file_stem("  Head 4.0 mm  ") == "Head_4_0_mm"
