"""Text size of the whole GUI (Settings > Text size..., 80-200 %).

Every font size in the views is written as `px(n)` / `pt(n)`, with `n` the size at 100 %; the factor is the
user's saved choice. It is read once at startup (`load_text_scale`, before the main window is built) and a new
choice applies after a restart, because stylesheets are built when the widgets are created.

Default 125 %: the original sizes (10-12 px) were too small for some users (2026-10-20).
"""

from __future__ import annotations

from bictool.infrastructure.ui_settings import UiSettings

TEXT_SCALE_KEY = "view/text_scale"
MIN_TEXT_SCALE, MAX_TEXT_SCALE = 0.8, 2.0
DEFAULT_TEXT_SCALE = 1.25

_scale = DEFAULT_TEXT_SCALE


def saved_text_scale(settings: UiSettings | None = None) -> float:
    """The saved factor (the default for missing or out-of-range values) - may differ from the running one
    until the next start."""
    saved = (settings or UiSettings()).get_str(TEXT_SCALE_KEY, "")
    try:
        value = float(saved)
    except ValueError:
        return DEFAULT_TEXT_SCALE
    return value if MIN_TEXT_SCALE <= value <= MAX_TEXT_SCALE else DEFAULT_TEXT_SCALE


def load_text_scale(settings: UiSettings | None = None) -> float:
    """Use the saved factor from now on (call once at startup, before any widget is built)."""
    global _scale
    _scale = saved_text_scale(settings)
    return _scale


def save_text_scale(value: float, settings: UiSettings | None = None) -> None:
    """Remember a new factor for the next start (the running window keeps its sizes)."""
    if not MIN_TEXT_SCALE <= value <= MAX_TEXT_SCALE:
        raise ValueError(f"Text scale {value!r} outside {MIN_TEXT_SCALE}-{MAX_TEXT_SCALE}")
    (settings or UiSettings()).set_str(TEXT_SCALE_KEY, str(value))


def text_scale() -> float:
    return _scale


def px(size: float) -> int:
    """A font size given in pixels at 100 %, scaled."""
    return max(1, round(size * _scale))


def pt(size: float) -> float:
    """A font size given in points at 100 %, scaled."""
    return size * _scale
