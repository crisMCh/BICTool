"""Text size setting: scaling helpers, saved value, dialog. Uses the isolated UI settings of conftest."""

import pytest
from PySide6.QtWidgets import QPushButton

import bictool.presentation.views.text_scale as text_scale
from bictool.infrastructure.ui_settings import UiSettings
from bictool.presentation.views.text_scale import (
    DEFAULT_TEXT_SCALE,
    TEXT_SCALE_KEY,
    load_text_scale,
    px,
    save_text_scale,
    saved_text_scale,
)
from bictool.presentation.views.text_size_dialog import TextSizeDialog


@pytest.fixture(autouse=True)
def restore_scale():
    yield
    text_scale._scale = DEFAULT_TEXT_SCALE


def test_sizes_scale_with_the_factor():
    text_scale._scale = 1.5
    assert px(12) == 18 and px(10) == 15
    text_scale._scale = 1.0
    assert px(12) == 12
    text_scale._scale = 0.8
    assert px(1) >= 1


def test_the_saved_size_is_used_from_the_next_start():
    settings = UiSettings()
    assert load_text_scale(settings) == DEFAULT_TEXT_SCALE  # nothing saved yet: default (125 %)

    save_text_scale(1.6, settings)

    assert text_scale.text_scale() == DEFAULT_TEXT_SCALE  # the running window keeps its size
    assert saved_text_scale(settings) == 1.6
    assert load_text_scale(settings) == 1.6 and px(10) == 16


def test_unusable_saved_values_fall_back_to_the_default():
    settings = UiSettings()
    for bad in ("huge", "5.0", "0.1"):
        settings.set_str(TEXT_SCALE_KEY, bad)
        assert saved_text_scale(settings) == DEFAULT_TEXT_SCALE
    with pytest.raises(ValueError):
        save_text_scale(3.0, settings)


def test_the_dialog_returns_the_chosen_percentage_and_restores_the_default():
    dialog = TextSizeDialog(1.5)
    assert dialog.value() == 1.5

    dialog._percent_spin.setValue(170)
    assert dialog.value() == 1.7
    assert "font-size: 20px" in dialog._preview.styleSheet()  # preview follows the choice (12 px x 1.7)

    restore = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Restore Defaults")
    restore.click()
    assert dialog.value() == DEFAULT_TEXT_SCALE
