"""The BICTool app icon (window title bar, taskbar, Alt-Tab).

The icon files live in `bictool/presentation/resources/` (PNGs 16-256 px rendered from the logo SVG by
`design/logo/make_app_icon.py`). PNGs are used rather than the SVG so the icon does not depend on Qt's SVG
image plugin being deployed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
WINDOWS_APP_ID = "OVGU.BICTool"


def app_icon() -> QIcon:
    """All available sizes in one QIcon (Qt picks the best one per place). Empty if the files are missing."""
    icon = QIcon()
    for size in ICON_SIZES:
        path = RESOURCES / f"app_icon_{size}.png"
        if path.is_file():
            icon.addFile(str(path))
    return icon


def register_windows_app_id() -> None:
    """On Windows, give the process its own taskbar identity. Without it, a program started with python.exe
    is grouped under Python and the taskbar shows the Python icon instead of the window icon.
    Must run before the first window is created. No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError):
        pass  # very old Windows or restricted shell - only the taskbar grouping is affected
