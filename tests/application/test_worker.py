"""`Worker.run()` is called directly (not via `QThreadPool`) so these tests are
synchronous: signal emission to a same-thread, directly-connected slot happens
immediately, no event loop or waiting required.
"""

from bictool.application.worker import Worker


def _run(job):
    worker = Worker(job)
    finished, errors = [], []
    worker.signals.finished.connect(finished.append)
    worker.signals.error.connect(errors.append)
    worker.run()
    return finished, errors


def test_run_emits_finished_with_the_jobs_result():
    sentinel = object()
    assert _run(lambda: sentinel) == ([sentinel], [])


def test_run_emits_error_instead_of_raising():
    def _raise():
        raise ValueError("boom")

    assert _run(_raise) == ([], ["boom"])
