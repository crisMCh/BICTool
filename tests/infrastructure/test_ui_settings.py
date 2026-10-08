"""Tests `ui_settings.py` against a throwaway INI file - never the real user settings."""

from PySide6.QtCore import QSettings

from bictool.infrastructure.ui_settings import ROI_DETAILS_OPEN, SHOW_TRACE_LOG, UiSettings


def test_defaults_and_round_trip(tmp_path):
    store = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    settings = UiSettings(store)

    assert settings.get_bool(SHOW_TRACE_LOG) is False  # trace off by default
    assert settings.get_bool(ROI_DETAILS_OPEN) is False

    settings.set_bool(SHOW_TRACE_LOG, True)
    reread = UiSettings(QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat))
    assert reread.get_bool(SHOW_TRACE_LOG) is True
    assert reread.get_bool(ROI_DETAILS_OPEN) is False
