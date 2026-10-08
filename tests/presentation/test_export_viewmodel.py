"""Tests `ExportViewModel` without widgets - writes real files to a temp folder."""

from types import SimpleNamespace

from bictool.presentation.viewmodels.export_viewmodel import ExportViewModel


def _result():
    return SimpleNamespace(
        mtf_area=0.5, nps_roi2=SimpleNamespace(var_nps=20.0), ratio=SimpleNamespace(ratio_lf_hf=11.0),
        contrast_roi1=0.2, decision_value=0.3, predicted_label=1, feature_vector=[0.5, 20, 100, 70, 11, -1, 0.1],
        recipe="octave",
    )


def _vm():
    volume = SimpleNamespace(num_slices=40, pixel_spacing_mm=0.45, acquisition_date="20260101")
    qa = SimpleNamespace(results={(3, "falx"): _result()}, errors={(4, "falx"): "no edges"})
    qa.results_by_pair = lambda pair: {i: r for (i, p), r in qa.results.items() if p == pair}
    vm = ExportViewModel(qa, SimpleNamespace(verdict=None), lambda: volume)
    messages = []
    vm.message.connect(lambda level, title, text: messages.append((level, title)))
    return vm, messages


def test_slice_export_writes_the_chosen_formats_and_charts(tmp_path):
    vm, messages = _vm()
    assert not vm.can_export_slice()
    vm.show_slice(3, "falx", _result(), "CT · Head")

    written = vm.export_slice({"excel", "csv", "jpg"}, tmp_path / "out", lambda d, stem: [d / f"{stem}_mtf.jpg"])

    names = sorted(p.name for p in written)
    assert names == ["CT_Head_slice004_metrics.csv", "CT_Head_slice004_metrics.xlsx", "CT_Head_slice004_mtf.jpg"]
    assert messages == [("information", "Export complete")]


def test_study_export_writes_summary_and_slices(tmp_path):
    vm, messages = _vm()

    written = vm.export_study({"csv"}, tmp_path, "CT")

    assert sorted(p.name for p in written) == ["CT_study_slices.csv", "CT_study_summary.csv"]
    assert "5 (Falx)" in (tmp_path / "CT_study_summary.csv").read_text(encoding="ascii")  # failed slice (1-based)
    assert messages == [("information", "Export complete")]


def test_a_failed_export_is_reported(tmp_path):
    vm, messages = _vm()
    vm.show_slice(3, "falx", _result(), "CT")
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where the folder should go

    assert vm.export_slice({"csv"}, blocker / "sub") == []
    assert messages == [("critical", "Export failed")]
