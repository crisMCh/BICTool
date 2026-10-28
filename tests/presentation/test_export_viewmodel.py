"""Tests `ExportViewModel` without widgets - writes real files to a temp folder."""

from types import SimpleNamespace

import openpyxl

from bictool.presentation.viewmodels.export_viewmodel import ExportViewModel


def _result():
    return SimpleNamespace(
        mtf_area=0.5, nps_roi2=SimpleNamespace(var_nps=20.0), ratio=SimpleNamespace(ratio_lf_hf=11.0),
        contrast_roi1=0.2, decision_value=0.3, predicted_label=1, feature_vector=[0.5, 20, 100, 70, 11, -1, 0.1],
        recipe="octave",
    )


def _vm():
    volume = SimpleNamespace(num_slices=40, pixel_spacing_mm=0.45, acquisition_date="20260101",
                             series_description="CCT nativ 4,00 Hr44")
    qa = SimpleNamespace(results={(3, "falx"): _result()}, errors={(4, "falx"): "no edges"})
    qa.results_by_pair = lambda pair: {i: r for (i, p), r in qa.results.items() if p == pair}
    results = SimpleNamespace(
        verdict=None, artefacts=None, basal_ganglia=None, plausibility=None,
        report=lambda: ("Study verdict (majority of slices)", "good", ["Falx: Acceptable - 1 of 1 slice(s) acceptable"]),
    )
    vm = ExportViewModel(qa, results, lambda: volume)
    messages = []
    vm.message.connect(lambda level, title, text: messages.append((level, title)))
    return vm, messages


def test_suggested_names_are_readable_and_name_the_scope():
    vm, _messages = _vm()
    assert vm.suggested_name("study") == "BICTool study 2026-01-01 CCT nativ 4,00 Hr44"
    vm.show_slice(3, "falx", _result(), "CT")
    assert vm.suggested_name("slice") == "BICTool slice 4 Falx 2026-01-01 CCT nativ 4,00 Hr44"


def test_the_save_dialog_offers_the_main_file_type():
    assert ExportViewModel.file_type({"excel", "csv"}) == (".xlsx", "Excel workbook (*.xlsx)")
    assert ExportViewModel.file_type({"csv", "jpg"})[0] == ".csv"
    assert ExportViewModel.file_type({"jpg"})[0] == ".jpg"


def test_slice_export_names_every_file_after_the_chosen_name(tmp_path):
    vm, messages = _vm()
    assert not vm.can_export_slice()
    vm.show_slice(3, "falx", _result(), "CT · Head")

    written = vm.export_slice(
        {"excel", "csv", "jpg"}, tmp_path / "out" / "My scan.xlsx", lambda folder, name: [folder / f"{name} - MTF.jpg"]
    )

    assert sorted(p.name for p in written) == ["My scan - MTF.jpg", "My scan.csv", "My scan.xlsx"]
    assert messages == [("information", "Export complete")]


def test_study_export_writes_the_report_and_the_tables(tmp_path):
    vm, messages = _vm()

    written = vm.export_study({"excel", "csv"}, tmp_path / "Head QA", "CT")

    assert sorted(p.name for p in written) == ["Head QA - per slice.csv", "Head QA - summary.csv", "Head QA.xlsx"]
    assert "5 (Falx)" in (tmp_path / "Head QA - summary.csv").read_text(encoding="ascii")  # failed slice (1-based)
    first_column = [row[0] for row in openpyxl.load_workbook(tmp_path / "Head QA.xlsx")["Summary"].iter_rows(values_only=True)]
    result = first_column.index("Result")
    assert first_column[result + 1:result + 3] == [  # the Study summary as the app words it
        "Study verdict (majority of slices)", "Falx: Acceptable - 1 of 1 slice(s) acceptable",
    ]
    assert messages == [("information", "Export complete")]


def test_a_failed_export_is_reported(tmp_path):
    vm, messages = _vm()
    vm.show_slice(3, "falx", _result(), "CT")
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where the folder should go

    assert vm.export_slice({"csv"}, blocker / "sub" / "name") == []
    assert messages == [("critical", "Export failed")]
