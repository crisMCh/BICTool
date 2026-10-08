import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_ui_settings(tmp_path_factory):
    """Remembered UI choices (`infrastructure.ui_settings`) go to a fresh
    folder per test - never the real user's settings, never shared between tests."""
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path_factory.mktemp("ui_settings")))
    yield


@pytest.fixture(autouse=True)
def non_blocking_message_boxes(monkeypatch):
    """A modal message box would wait forever for a click in a headless run;
    in tests they return at once. Tests that check dialogs patch them again."""
    from PySide6.QtWidgets import QMessageBox

    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    yield
