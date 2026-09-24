"""`StrokeWorker.run()` is called directly (not via `QThreadPool`) so these
tests are synchronous - see `test_worker.py`'s module docstring for why.
`segment_slice_for_stroke` is monkeypatched throughout, so no real model/
TensorFlow inference ever runs here.
"""

import numpy as np

import bictool.application.stroke_worker as stroke_worker_module
from bictool.application.stroke_worker import StrokeWorker

SLICE_HU = np.zeros((512, 512), dtype=np.float64)


def test_run_emits_finished_with_the_stroke_result(monkeypatch):
    sentinel_result = object()
    monkeypatch.setattr(stroke_worker_module, "segment_slice_for_stroke", lambda *a, **k: sentinel_result)

    worker = StrokeWorker(model=object(), slice_hu=SLICE_HU)
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

    monkeypatch.setattr(stroke_worker_module, "segment_slice_for_stroke", _raise)

    worker = StrokeWorker(model=object(), slice_hu=SLICE_HU)
    finished_calls = []
    error_calls = []
    worker.signals.finished.connect(finished_calls.append)
    worker.signals.error.connect(error_calls.append)

    worker.run()  # must not raise

    assert finished_calls == []
    assert error_calls == ["boom"]


def test_run_passes_the_model_and_slice_through(monkeypatch):
    received_args = []
    monkeypatch.setattr(
        stroke_worker_module,
        "segment_slice_for_stroke",
        lambda *a, **k: received_args.append(a) or object(),
    )
    sentinel_model = object()

    worker = StrokeWorker(model=sentinel_model, slice_hu=SLICE_HU)
    worker.run()

    assert len(received_args) == 1
    model_arg, slice_arg = received_args[0]
    assert model_arg is sentinel_model
    np.testing.assert_array_equal(slice_arg, SLICE_HU)
