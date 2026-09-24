"""`VolumeSegmentationWorker.run()` is called directly (not via
`QThreadPool`), so these tests are synchronous - see `test_worker.py`'s
module docstring for why. `segment_slice` is monkeypatched throughout, so
no real model/TensorFlow inference ever runs here.
"""

import numpy as np

import bictool.application.volume_segmentation_worker as volume_worker_module
from bictool.application.volume_segmentation_worker import VolumeSegmentationWorker

SLICE_HU = np.zeros((512, 512), dtype=np.float64)


def _get_slice_hu(index: int) -> np.ndarray:
    return SLICE_HU


def test_run_emits_slice_finished_and_progress_for_every_slice_then_finished(monkeypatch):
    monkeypatch.setattr(volume_worker_module, "segment_slice", lambda model, slice_hu: f"result-{model}")

    worker = VolumeSegmentationWorker(model="m", get_slice_hu=_get_slice_hu, slice_indices=[0, 1, 2])
    slice_finished_calls = []
    progress_calls = []
    finished_calls = []
    worker.signals.slice_finished.connect(lambda index, result: slice_finished_calls.append((index, result)))
    worker.signals.progress.connect(lambda completed, total: progress_calls.append((completed, total)))
    worker.signals.finished.connect(lambda: finished_calls.append(True))

    worker.run()

    assert slice_finished_calls == [(0, "result-m"), (1, "result-m"), (2, "result-m")]
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]
    assert finished_calls == [True]


def test_cancel_before_run_stops_before_the_first_slice(monkeypatch):
    monkeypatch.setattr(volume_worker_module, "segment_slice", lambda model, slice_hu: "result")

    worker = VolumeSegmentationWorker(model="m", get_slice_hu=_get_slice_hu, slice_indices=[0, 1, 2])
    slice_finished_calls = []
    cancelled_calls = []
    worker.signals.slice_finished.connect(lambda index, result: slice_finished_calls.append(index))
    worker.signals.cancelled.connect(lambda completed, total: cancelled_calls.append((completed, total)))
    worker.cancel()

    worker.run()

    assert slice_finished_calls == []
    assert cancelled_calls == [(0, 3)]


def test_cancel_requested_mid_run_stops_before_the_next_slice(monkeypatch):
    completed_slices = []

    def _segment(model, slice_hu):
        completed_slices.append(slice_hu)
        if len(completed_slices) == 2:
            worker.cancel()
        return "result"

    monkeypatch.setattr(volume_worker_module, "segment_slice", _segment)
    worker = VolumeSegmentationWorker(model="m", get_slice_hu=_get_slice_hu, slice_indices=[0, 1, 2, 3])
    slice_finished_calls = []
    cancelled_calls = []
    worker.signals.slice_finished.connect(lambda index, result: slice_finished_calls.append(index))
    worker.signals.cancelled.connect(lambda completed, total: cancelled_calls.append((completed, total)))

    worker.run()

    # cancel() is requested while slice 1 is being processed, but the flag
    # is only checked before starting the *next* slice - so slice 1 still
    # finishes and is emitted; slices 2/3 never run.
    assert slice_finished_calls == [0, 1]
    assert cancelled_calls == [(2, 4)]


def test_run_emits_error_and_stops_instead_of_raising(monkeypatch):
    def _raise(model, slice_hu):
        raise ValueError("boom")

    monkeypatch.setattr(volume_worker_module, "segment_slice", _raise)
    worker = VolumeSegmentationWorker(model="m", get_slice_hu=_get_slice_hu, slice_indices=[0, 1])
    error_calls = []
    finished_calls = []
    worker.signals.error.connect(lambda index, message: error_calls.append((index, message)))
    worker.signals.finished.connect(lambda: finished_calls.append(True))

    worker.run()  # must not raise

    assert error_calls == [(0, "boom")]
    assert finished_calls == []


def test_run_passes_each_slices_own_pixel_data(monkeypatch):
    received = []
    monkeypatch.setattr(volume_worker_module, "segment_slice", lambda model, slice_hu: received.append(slice_hu) or "r")

    def get_slice_hu(index: int) -> np.ndarray:
        return np.full((4, 4), index, dtype=np.float64)

    worker = VolumeSegmentationWorker(model="m", get_slice_hu=get_slice_hu, slice_indices=[0, 1, 2])
    worker.run()

    assert len(received) == 3
    for i, arr in enumerate(received):
        np.testing.assert_array_equal(arr, np.full((4, 4), i, dtype=np.float64))
