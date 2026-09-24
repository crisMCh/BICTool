import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qt_app():
    """`Worker`/`Scheduler` signal-slot machinery needs a Qt application instance.

    Uses `QApplication`, not the lighter `QCoreApplication`: the test suite may
    also run `tests/presentation`'s QWidget-based tests in the same session, and
    Qt allows only one application instance per process - whichever conftest
    runs first (alphabetically, this one) must create the QWidget-capable kind,
    or constructing widgets later hangs.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
