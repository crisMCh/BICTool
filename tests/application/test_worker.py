"""`Worker.run()` is called directly (not via `QThreadPool`) so these tests are
synchronous: signal emission to a same-thread, directly-connected slot happens
immediately, no event loop or waiting required.
"""

import numpy as np

import bictool.application.worker as worker_module
from bictool.application.worker import Worker

IMG_ROI1 = np.array(
    [
        [3, 1, 4, 1, 5, 9, 2],
        [6, 5, 3, 5, 8, 9, 7],
        [9, 3, 2, 3, 8, 4, 6],
        [2, 6, 4, 3, 3, 8, 3],
        [2, 7, 9, 5, 0, 2, 8],
        [8, 4, 1, 9, 7, 1, 6],
        [9, 3, 9, 9, 3, 7, 5],
    ],
    dtype=np.float64,
)
IMG_ROI2 = IMG_ROI1 + 100.0
PIX_SPC = 0.5


def test_run_emits_finished_with_the_core_api_result(monkeypatch):
    sentinel_result = object()
    monkeypatch.setattr(worker_module, "evaluate_slice_quality", lambda *a, **k: sentinel_result)

    worker = Worker(IMG_ROI1, IMG_ROI2, PIX_SPC)
    finished_calls = []
    error_calls = []
    worker.signals.finished.connect(finished_calls.append)
    worker.signals.error.connect(error_calls.append)

    worker.run()

    assert finished_calls == [sentinel_result]
    assert error_calls == []


def test_run_emits_error_instead_of_raising(monkeypatch):
    def _raise(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(worker_module, "evaluate_slice_quality", _raise)

    worker = Worker(IMG_ROI1, IMG_ROI2, PIX_SPC)
    finished_calls = []
    error_calls = []
    worker.signals.finished.connect(finished_calls.append)
    worker.signals.error.connect(error_calls.append)

    worker.run()  # must not raise

    assert finished_calls == []
    assert error_calls == ["boom"]


def test_run_passes_through_all_arguments(monkeypatch):
    received_args = []
    monkeypatch.setattr(
        worker_module,
        "evaluate_slice_quality",
        lambda *a, **k: received_args.append(a) or object(),
    )

    worker = Worker(IMG_ROI1, IMG_ROI2, PIX_SPC, act_cont_mask_x=5, act_cont_mask_y=7)
    worker.run()

    assert len(received_args) == 1
    args = received_args[0]
    np.testing.assert_array_equal(args[0], IMG_ROI1)
    np.testing.assert_array_equal(args[1], IMG_ROI2)
    assert args[2] == PIX_SPC
    assert args[3] == 5
    assert args[4] == 7
