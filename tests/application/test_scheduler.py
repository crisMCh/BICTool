from PySide6.QtCore import QRunnable

from bictool.application.scheduler import Scheduler


class _RecordingRunnable(QRunnable):
    def __init__(self):
        super().__init__()
        self.ran = False

    def run(self):
        self.ran = True


def test_submit_runs_the_job_on_the_thread_pool():
    scheduler = Scheduler()
    job = _RecordingRunnable()

    scheduler.submit(job)
    scheduler.pool.waitForDone(2000)

    assert job.ran is True
