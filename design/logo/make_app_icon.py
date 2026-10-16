"""Builds the app icon from the chosen logo (logo5_flat_ring_dark_transparent.svg - the bright variant, readable on dark title bars and taskbars; chosen 2026-10-16, temporary):
a tighter-cropped SVG plus PNGs (16-256 px) and a Windows .ico in source/bictool/presentation/resources/.

Run after changing the logo:  .venv/Scripts/python.exe design/logo/make_app_icon.py
"""

import re
import sys
from pathlib import Path

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "logo5_flat_ring_dark_transparent.svg"
RES = HERE.parents[1] / "source" / "bictool" / "presentation" / "resources"
ICON_VIEWBOX = "290 163 642 642"   # content bounding box (ring + couch) plus a small margin
SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    svg = re.sub(r'viewBox="[^"]+"', f'viewBox="{ICON_VIEWBOX}"', SOURCE.read_text(encoding="utf-8"), count=1)
    (RES / "app_icon.svg").write_text(svg, encoding="utf-8")

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    renderer = QSvgRenderer(str(RES / "app_icon.svg"))
    images = {}
    for size in SIZES:
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()
        img.save(str(RES / f"app_icon_{size}.png"))
        images[size] = img
    if not images[256].save(str(RES / "app_icon.ico")):
        print("note: Qt could not write .ico (missing imageformats plugin) - PNGs and SVG are enough for the app")
    print("written to", RES)
    del app


if __name__ == "__main__":
    main()
