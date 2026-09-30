"""`BrainTissueVolumeWorker.run()` is called directly (not via `QThreadPool`),
so these tests are synchronous - same approach as
`test_volume_segmentation_worker.py`.
"""

from bictool.application.brain_tissue_volume_worker import BrainTissueVolumeWorker


def test_run_classifies_every_slice_then_finishes():
    worker = BrainTissueVolumeWorker(lambda index: f"tissue-{index}", [3, 5])
    slices, finished = [], []
    worker.signals.slice_finished.connect(lambda index, result: slices.append((index, result)))
    worker.signals.finished.connect(lambda: finished.append(True))

    worker.run()

    assert slices == [(3, "tissue-3"), (5, "tissue-5")]
    assert finished == [True]


def test_cancel_stops_before_the_next_slice():
    worker = BrainTissueVolumeWorker(lambda index: index, [0, 1, 2])
    slices, cancelled = [], []
    worker.signals.slice_finished.connect(lambda index, result: (slices.append(index), worker.cancel()))
    worker.signals.cancelled.connect(lambda: cancelled.append(True))

    worker.run()

    assert slices == [0]
    assert cancelled == [True]


def test_error_is_reported_with_the_failing_slice():
    def classify(index):
        if index == 1:
            raise ValueError("boom")
        return index

    worker = BrainTissueVolumeWorker(classify, [0, 1, 2])
    errors, finished = [], []
    worker.signals.error.connect(lambda index, message: errors.append((index, message)))
    worker.signals.finished.connect(lambda: finished.append(True))

    worker.run()

    assert errors == [(1, "boom")]
    assert finished == []
