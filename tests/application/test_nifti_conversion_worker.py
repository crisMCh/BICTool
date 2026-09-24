"""`NiftiConversionWorker.run()` is called directly (not via `QThreadPool`)
so these tests are synchronous - see `test_worker.py`'s module docstring
for why. `build_dicom_nifti_volume` is monkeypatched throughout, so no real
`dicom2nifti` conversion ever runs here.
"""

import bictool.application.nifti_conversion_worker as nifti_conversion_worker_module
from bictool.application.nifti_conversion_worker import NiftiConversionWorker

SLICE_PATHS = ["a.dcm", "b.dcm"]


def test_run_emits_finished_with_the_dicom_nifti_volume(monkeypatch):
    sentinel_result = object()
    monkeypatch.setattr(
        nifti_conversion_worker_module, "build_dicom_nifti_volume", lambda *a, **k: sentinel_result
    )

    worker = NiftiConversionWorker(slice_paths=SLICE_PATHS, get_slice_hu=lambda i: i)
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

    monkeypatch.setattr(nifti_conversion_worker_module, "build_dicom_nifti_volume", _raise)

    worker = NiftiConversionWorker(slice_paths=SLICE_PATHS, get_slice_hu=lambda i: i)
    finished_calls = []
    error_calls = []
    worker.signals.finished.connect(finished_calls.append)
    worker.signals.error.connect(error_calls.append)

    worker.run()  # must not raise

    assert finished_calls == []
    assert error_calls == ["boom"]


def test_run_passes_the_slice_paths_and_get_slice_hu_through(monkeypatch):
    received_args = []
    monkeypatch.setattr(
        nifti_conversion_worker_module,
        "build_dicom_nifti_volume",
        lambda *a, **k: received_args.append(a) or object(),
    )
    sentinel_get_slice_hu = object()

    worker = NiftiConversionWorker(slice_paths=SLICE_PATHS, get_slice_hu=sentinel_get_slice_hu)
    worker.run()

    assert len(received_args) == 1
    paths_arg, get_slice_hu_arg = received_args[0]
    assert paths_arg == SLICE_PATHS
    assert get_slice_hu_arg is sentinel_get_slice_hu
