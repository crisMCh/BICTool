"""Tests `QaRunViewModel` without widgets - fake slices, fake measurements."""

from types import SimpleNamespace

import numpy as np

import bictool.application.worker as worker_module
from bictool.presentation.viewmodels.qa_run_viewmodel import QaRunViewModel
from bictool.presentation.viewmodels.roi_selection import RoiRect


def _vm():
    volume = SimpleNamespace(pixel_spacing_mm=0.5, get_slice_hu=lambda i: np.random.default_rng(i).normal(30, 5, (64, 64)))
    rois = {"falx.roi1": RoiRect(5, 5, 25, 25), "falx.roi2": RoiRect(30, 30, 50, 50)}
    workers = []
    vm = QaRunViewModel(workers.append, lambda: volume, lambda i: rois, lambda: {})
    events = {"progress": [], "finished": [], "done": []}
    vm.progress_changed.connect(events["progress"].append)
    vm.finished.connect(events["finished"].append)
    vm.job_done.connect(events["done"].append)
    return vm, workers, events


def _run_queued(workers):
    while workers:
        workers.pop(0).run()  # synchronous - the signals deliver directly


def test_every_job_runs_once_and_results_are_kept_per_slice_and_pair(monkeypatch):
    monkeypatch.setattr(worker_module, "evaluate_slice_quality", lambda *a, **k: SimpleNamespace(
        mtf_area=0.5, predicted_label=1, decision_value=0.1, recipe="physical"))
    vm, workers, events = _vm()

    vm.run([(0, "falx"), (3, "falx")])
    _run_queued(workers)

    assert set(vm.results) == {(0, "falx"), (3, "falx")} and vm.is_physical()
    assert events["done"] == [(0, "falx"), (3, "falx")] and events["finished"] == [False]
    assert events["progress"] == ["QA 0/2", "QA 1/2", "QA 2/2", "QA done: 2 ROI pair(s)"]
    assert not vm.is_running() and list(vm.results_by_pair("falx")) == [0, 3]


def test_errors_are_kept_and_cancel_stops_further_jobs(monkeypatch):
    def evaluate(*a, **k):
        raise ValueError("no edges")

    monkeypatch.setattr(worker_module, "evaluate_slice_quality", evaluate)
    vm, workers, events = _vm()

    vm.run([(0, "falx"), (1, "falx"), (2, "falx")])
    vm.cancel()
    _run_queued(workers)

    assert vm.errors == {(0, "falx"): "no edges"}  # the running job finished; no further one started
    assert events["finished"] == [True]
    assert events["progress"][-1] == "QA cancelled: 1/3 ROI pair(s), 1 failed"


def test_switching_recipe_drops_results_of_the_other_recipe():
    vm, _workers, _events = _vm()
    vm.results = {(0, "falx"): object()}
    resets = []
    vm.reset_done.connect(lambda: resets.append(True))

    assert vm.set_recipe("octave") is True
    assert vm.results == {} and resets == [True] and vm.recipe == "octave"
    assert vm.set_recipe("octave") is False
