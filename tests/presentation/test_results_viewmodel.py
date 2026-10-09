"""Tests `ResultsViewModel` without widgets - fake QA results."""

from types import SimpleNamespace

from PySide6.QtCore import QObject, Signal

from bictool.presentation.viewmodels.head_ct_verdict_text import TONE_BAD, TONE_GOOD
from bictool.presentation.viewmodels.results_viewmodel import STUDY_TABLE_COLUMNS, ResultsViewModel


def _result(label=1, recipe="octave"):
    return SimpleNamespace(
        mtf_area=0.5, nps_roi2=SimpleNamespace(var_nps=20.0), ratio=SimpleNamespace(ratio_lf_hf=11.0),
        predicted_label=label, decision_value=0.3 if label else -0.3, recipe=recipe,
    )


class _FakeQa(QObject):
    finished = Signal(bool)
    reset_done = Signal()


def _qa(results, errors=None):
    qa = _FakeQa()
    qa.results, qa.errors = results, errors or {}
    qa.results_by_pair = lambda pair: {i: r for (i, p), r in qa.results.items() if p == pair}
    qa.is_physical = lambda: bool(qa.results) and all(r.recipe == "physical" for r in qa.results.values())
    return qa


def test_slices_table_and_legacy_verdict_follow_the_results():
    qa = _qa({(4, "falx"): _result(1), (2, "falx"): _result(0), (4, "septum"): _result(1)})
    vm = ResultsViewModel(qa, lambda: None)

    vm.refresh()

    assert vm.slices == [2, 4] and vm.table_jobs == [(2, "falx"), (4, "falx"), (4, "septum")]
    assert vm.table_columns() == STUDY_TABLE_COLUMNS
    rows = vm.table_rows()
    assert rows[0].cells[:2] == ["3", "Falx"] and rows[0].last_tone == TONE_BAD
    lines, tones = vm.legacy_verdict()
    assert lines[0].startswith("Falx: ") and len(lines) == 2


def test_slice_view_prefers_the_chosen_pair_and_falls_back():
    qa = _qa({(4, "falx"): _result(1), (5, "septum"): _result(1)})
    vm = ResultsViewModel(qa, lambda: None)
    vm.refresh()
    vm.select_pair("septum")

    assert vm.slice_view(4).pair == "falx"  # no septum result there
    view = vm.slice_view(5)
    assert view.pair == "septum" and view.position == 1 and view.count == 2
    assert view.verdict_text == "Acceptable" and view.verdict_tone == TONE_GOOD
    assert vm.slice_view(9) is None


def test_physical_results_get_the_head_ct_verdict_text_without_a_model():
    qa = _qa({(1, "falx"): _result(recipe="physical")}, errors={(2, "falx"): "no edges"})
    vm = ResultsViewModel(qa, lambda: None)
    vm.refresh()

    text, details = vm.head_ct_text(model_available=False)

    assert vm.is_physical() and vm.verdict is None
    assert "model file missing" in text.headline
    assert "Failed (not used): 3 (Falx)" in details and "Provisional model" in details


def test_the_results_follow_the_qa_run():
    qa = _qa({})
    vm = ResultsViewModel(qa, lambda: None)
    changed = []
    vm.changed.connect(lambda: changed.append(True))

    qa.results = {(4, "falx"): _result(1)}
    qa.finished.emit(False)
    assert vm.slices == [4]

    qa.results = {}
    qa.reset_done.emit()
    assert vm.slices == [] and changed == [True, True]


def test_artefact_flags_are_computed_from_the_shown_series():
    import numpy as np

    slices = [np.zeros((40, 40)), np.zeros((40, 40))]
    slices[1][5:8, 5:8] = 3500.0  # metal
    volume = SimpleNamespace(num_slices=2, pixel_spacing_mm=0.5, get_slice_hu=lambda i: slices[i])
    qa = _qa({(0, "falx"): _result(1)})
    vm = ResultsViewModel(qa, lambda: None, volume=lambda: volume, segmentation_cache=lambda: {})

    vm.refresh()

    assert vm.artefacts.metal_slices == (1,)
    assert vm.artefact_lines() == ["Artefacts (metal): on slice 2 - expect streaks nearby."]
    qa.results = {}
    vm.refresh()
    assert vm.artefacts is None and vm.artefact_lines() == []


def test_without_a_series_there_are_no_artefact_flags():
    vm = ResultsViewModel(_qa({(0, "falx"): _result(1)}), lambda: None)
    vm.refresh()
    assert vm.artefacts is None


def test_the_table_note_and_the_slice_view_mention_artefacts_on_that_slice():
    from bictool.domain.artefacts.artefact_flags import ArtefactFlags

    qa = _qa({(4, "falx"): _result(1, "physical"), (5, "falx"): _result(1, "physical")})
    vm = ResultsViewModel(qa, lambda: None)
    vm.refresh()
    vm.artefacts = ArtefactFlags(metal_slices=(4,), metal_max_hu=3071.0)

    notes = {row.job[0]: row.cells[-1] for row in vm.table_rows()}
    assert notes == {4: "metal", 5: ""}
    assert vm.slice_view(4).note_text == "Possible artefact on this slice: metal"
    assert vm.slice_view(5).note_text == ""
