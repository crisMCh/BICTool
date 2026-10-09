"""Results > Per slice > Advanced: the slice's intermediate QA steps
(`viewmodels.slice_steps.SliceStepsView`) in four tabs - MTF steps, MTF
spread, ROIs & 2D spectra, PS/MTF & features.

Pure view: it draws what the ViewModel prepared (images with overlays,
curve sets, a histogram). The colour palette is duplicated from
`main_window.py` (as in the settings dialogs) to avoid a circular import.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCharts import QAreaSeries, QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QTabWidget, QVBoxLayout, QWidget

from bictool.presentation.viewmodels.slice_steps import (
    ROLE_DROPPED,
    ROLE_EDGE,
    ROLE_HIGH,
    CurvePanel,
    HistogramPanel,
    ImagePanel,
    SliceStepsView,
)

SURFACE_SUNKEN = "#0c0f13"
TEXT = "#e7eaed"
TEXT_MUTED = "#8d96a0"
ACCENT = "#49c8de"
GOOD = "#5fbf83"
WARNING = "#d9a441"
CRITICAL = "#e0685c"

_ROLE_COLORS = {ROLE_EDGE: GOOD, ROLE_DROPPED: "#6b7480"}
_CONTOUR, _PROFILE, _SEED = ACCENT, WARNING, CRITICAL
_BAND_COLORS = {ROLE_HIGH: WARNING}  # everything else: ACCENT
# Low (blurry) -> high (sharp) MTF area: dark violet -> teal -> yellow (perceptually ordered).
_RAMP = [(0.0, (68, 1, 84)), (0.5, (33, 145, 140)), (1.0, (253, 231, 37))]


def ramp_color(fraction: float) -> QColor:
    """A colour of the low->high ramp for `fraction` in 0..1."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    for (f0, c0), (f1, c1) in zip(_RAMP, _RAMP[1:]):
        if fraction <= f1:
            t = (fraction - f0) / (f1 - f0)
            return QColor(*(round(a + t * (b - a)) for a, b in zip(c0, c1)))
    return QColor(*_RAMP[-1][1])


class ImagePanelWidget(QWidget):
    """One `ImagePanel`: a greyscale image scaled to fit, with its overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel: ImagePanel | None = None
        self._qimage: QImage | None = None
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_panel(self, panel: ImagePanel | None) -> None:
        self._panel = panel
        self._qimage = None
        if panel is not None:
            grey = np.ascontiguousarray((np.clip(panel.image, 0, 1) * 255).astype(np.uint8))
            height, width = grey.shape
            self._qimage = QImage(grey.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        self.update()

    def image_rect(self) -> QRectF:
        """Where the image is drawn (aspect kept, centred; room for axis labels)."""
        if self._qimage is None:
            return QRectF()
        margin = 18.0 if self._panel is not None and self._panel.extent else 2.0
        avail_w, avail_h = self.width() - 2 * margin, self.height() - 2 * margin
        scale = min(avail_w / self._qimage.width(), avail_h / self._qimage.height())
        w, h = self._qimage.width() * scale, self._qimage.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(SURFACE_SUNKEN))
        panel, image = self._panel, self._qimage
        if panel is None or image is None:
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "—")
            return
        rect = self.image_rect()
        painter.drawImage(rect, image)
        sx, sy = rect.width() / image.width(), rect.height() / image.height()

        def point(row: float, col: float) -> QPointF:
            return QPointF(rect.left() + (col + 0.5) * sx, rect.top() + (row + 0.5) * sy)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dot = max(1.5, min(sx, sy) * 0.6)
        if panel.contour is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(_CONTOUR))
            for row, col in panel.contour:
                painter.drawEllipse(point(row, col), dot * 0.6, dot * 0.6)
        for row0, row1, col0, col1 in panel.boxes:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(_SEED), 1.5))
            painter.drawRect(QRectF(point(row0 - 0.5, col0 - 0.5), point(row1 - 0.5, col1 - 0.5)))
        if panel.lines is not None:
            painter.setPen(QPen(QColor(_PROFILE), 1.3))
            for row0, col0, row1, col1 in panel.lines:
                painter.drawLine(point(row0, col0), point(row1, col1))
        painter.setPen(Qt.PenStyle.NoPen)
        for points, role in panel.points:
            painter.setBrush(QColor(_ROLE_COLORS.get(role, TEXT)))
            for row, col in points:
                painter.drawEllipse(point(row, col), dot, dot)
        if panel.coloured_points is not None and panel.point_values is not None:
            low, high = panel.value_range or (0.0, 1.0)
            span = (high - low) or 1.0
            for (row, col), value in zip(panel.coloured_points, panel.point_values):
                if np.isfinite(value):
                    painter.setBrush(ramp_color((value - low) / span))
                    painter.drawEllipse(point(row, col), dot * 1.4, dot * 1.4)
        if panel.extent:
            x0, x1, y0, y1 = panel.extent
            painter.setPen(QColor(TEXT_MUTED))
            font = painter.font()
            font.setPointSizeF(7.5)
            painter.setFont(font)
            painter.drawText(QPointF(rect.left(), rect.bottom() + 12), f"{x0:.2f}")
            painter.drawText(QPointF(rect.right() - 28, rect.bottom() + 12), f"{x1:.2f}")
            painter.drawText(QPointF(rect.center().x() - 3, rect.bottom() + 12), "0")
            painter.drawText(QPointF(rect.left() - 16, rect.top() + 8), f"{y0:.1f}")


def _titled(widget: QWidget, title: str) -> tuple[QWidget, QLabel, QLabel]:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    title_label = QLabel(title)
    title_label.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600;")
    caption = QLabel("")
    caption.setWordWrap(True)
    caption.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
    layout.addWidget(title_label)
    layout.addWidget(widget, 1)
    layout.addWidget(caption)
    return box, title_label, caption


class _ImageCell:
    def __init__(self) -> None:
        self.view = ImagePanelWidget()
        self.box, self.title, self.caption = _titled(self.view, "")

    def show(self, panel: ImagePanel | None) -> None:
        self.view.set_panel(panel)
        self.title.setText(panel.title if panel else "")
        self.caption.setText(panel.caption if panel else "")


class _ChartCell:
    def __init__(self) -> None:
        chart = QChart()
        chart.legend().setVisible(False)
        chart.setBackgroundBrush(QBrush(QColor(SURFACE_SUNKEN)))
        chart.setBackgroundRoundness(6)
        self.x_axis, self.y_axis = QValueAxis(), QValueAxis()
        for axis in (self.x_axis, self.y_axis):
            axis.setLabelFormat("%.2g")
            axis.setTickCount(5)
            axis.setLabelsBrush(QBrush(QColor(TEXT_MUTED)))
            axis.setTitleBrush(QBrush(QColor(TEXT_MUTED)))
        chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)
        self.chart = chart
        # QAreaSeries doesn't own its boundary series - keep them alive while shown.
        self._area_bounds: list[QLineSeries] = []
        self.view = QChartView(chart)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setMinimumSize(220, 170)
        self.box, self.title, self.caption = _titled(self.view, "")

    def clear(self) -> None:
        for series in list(self.chart.series()):
            self.chart.removeSeries(series)
        self._area_bounds = []

    def _add(self, series, color: str, width: float, alpha: int = 255) -> None:
        pen = QPen(QColor(color))
        pen.setWidthF(width)
        c = pen.color()
        c.setAlpha(alpha)
        pen.setColor(c)
        series.setPen(pen)
        self.chart.addSeries(series)
        series.attachAxis(self.x_axis)
        series.attachAxis(self.y_axis)

    def _line(self, x: np.ndarray, y: np.ndarray) -> QLineSeries:
        series = QLineSeries()
        series.replace([QPointF(float(a), float(b)) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)])
        return series

    def _shade(self, x0: float, x1: float, y0: float, y1: float, color: str) -> None:
        upper, lower = QLineSeries(), QLineSeries()
        upper.append(x0, y1)
        upper.append(x1, y1)
        lower.append(x0, y0)
        lower.append(x1, y0)
        area = QAreaSeries(upper, lower)
        self._area_bounds += [upper, lower]
        fill = QColor(color)
        fill.setAlpha(40)
        area.setBrush(QBrush(fill))
        area.setPen(QPen(Qt.PenStyle.NoPen))
        self.chart.addSeries(area)
        area.attachAxis(self.x_axis)
        area.attachAxis(self.y_axis)

    def _set_ranges(self, x: np.ndarray, ys: list[np.ndarray], x_pad: float = 0.0) -> tuple[float, float, float, float]:
        finite_x = x[np.isfinite(x)]
        values = np.concatenate([y[np.isfinite(y)] for y in ys]) if ys else np.zeros(0)
        x0, x1 = (float(finite_x.min()), float(finite_x.max())) if finite_x.size else (0.0, 1.0)
        y0, y1 = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
        if x1 <= x0:
            x1 = x0 + 1.0
        pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
        y0, y1 = y0 - pad, y1 + pad
        x_margin = x_pad * (x1 - x0)
        self.x_axis.setRange(x0 - x_margin, x1 + x_margin)
        self.y_axis.setRange(y0, y1)
        return x0, x1, y0, y1

    def show_curves(self, panel: CurvePanel | None) -> None:
        self.clear()
        self.title.setText(panel.title if panel else "")
        self.caption.setText(panel.note if panel else "")
        if panel is None or len(panel.x) == 0:
            return
        self.x_axis.setTitleText(panel.x_label)
        self.y_axis.setTitleText(panel.y_label)
        ys = list(panel.faint) + ([panel.mean] if panel.mean is not None else [])
        # A little room left and right so a marker at the very edge stays visible.
        _x0, _x1, y0, y1 = self._set_ranges(np.asarray(panel.x, dtype=float), ys, 0.02 if panel.marks else 0.0)
        for x0, x1, role in panel.bands:
            self._shade(x0, x1, y0, y1, _BAND_COLORS.get(role, ACCENT))
        for y in panel.faint:
            self._add(self._line(panel.x, y), TEXT_MUTED, 1.0, alpha=70)
        if panel.mean is not None:
            self._add(self._line(panel.x, panel.mean), ACCENT, 2.2)
        for mark in panel.marks:
            self._add(self._line(np.array([mark, mark]), np.array([y0, y1])), CRITICAL, 2.0)

    def show_histogram(self, panel: HistogramPanel | None) -> None:
        self.clear()
        self.title.setText(panel.title if panel else "")
        self.caption.setText(panel.note if panel else "")
        if panel is None or len(panel.counts) == 0:
            return
        self.x_axis.setTitleText(panel.x_label)
        self.y_axis.setTitleText("Edges")
        edges, counts = np.asarray(panel.edges, dtype=float), np.asarray(panel.counts, dtype=float)
        x = np.repeat(edges, 2)[1:-1]
        y = np.repeat(counts, 2)
        upper = self._line(x, y)
        lower = self._line(np.array([x[0], x[-1]]), np.zeros(2))
        area = QAreaSeries(upper, lower)
        self._area_bounds += [upper, lower]
        fill = QColor(ACCENT)
        fill.setAlpha(110)
        area.setBrush(QBrush(fill))
        area.setPen(QPen(QColor(ACCENT), 1.2))
        self.x_axis.setRange(float(edges[0]), float(edges[-1]))
        self.y_axis.setRange(0.0, float(counts.max()) * 1.1 + 0.5)
        self.chart.addSeries(area)
        area.attachAxis(self.x_axis)
        area.attachAxis(self.y_axis)
        for mark in panel.marks:
            self._add(self._line(np.array([mark, mark]), np.array([0.0, counts.max() * 1.1 + 0.5])), WARNING, 1.5)


class SliceStepsPanel(QTabWidget):
    """The four Advanced tabs; `show_steps(None)` shows why there is nothing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.roi1_edges, self.edge_map = _ImageCell(), _ImageCell()
        self.roi1, self.roi2, self.ps_2d, self.nps_2d = _ImageCell(), _ImageCell(), _ImageCell(), _ImageCell()
        self.esf, self.lsf, self.mtf = _ChartCell(), _ChartCell(), _ChartCell()
        self.histogram, self.ratio = _ChartCell(), _ChartCell()
        self.edge_counts = QLabel("")
        self.edge_counts.setStyleSheet(f"color: {TEXT};")
        self.edge_counts.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.ratio_features = QLabel("")
        self.ratio_features.setStyleSheet(f"color: {TEXT};")
        self.empty = QLabel("")
        self.empty.setStyleSheet(f"color: {TEXT_MUTED};")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)

        self._add_grid("MTF steps", [[self.roi1_edges.box, self.esf.box], [self.lsf.box, self.mtf.box]])
        counts_box = QWidget()
        counts_layout = QVBoxLayout(counts_box)
        counts_layout.addWidget(self.edge_counts)
        counts_layout.addStretch(1)
        self._add_grid("MTF spread", [[self.edge_map.box, self.histogram.box], [counts_box, None]], row_stretch=(3, 1))
        self._add_grid("ROIs && 2D spectra", [[self.roi1.box, self.roi2.box], [self.ps_2d.box, self.nps_2d.box]])
        # The feature vector box (owned by the window) is added to this tab by `set_feature_widget`.
        self._ratio_tab = self._add_grid("PS/MTF && features", [[self.ratio.box, None]])
        self._ratio_tab.layout().addWidget(self.ratio_features, 1, 0)
        self._empty_index = self.addTab(self.empty, "")
        self.setTabVisible(self._empty_index, False)

    def _add_grid(self, title: str, cells, row_stretch=(1, 1)) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(4, 6, 4, 4)
        grid.setSpacing(10)
        for row, row_cells in enumerate(cells):
            for col, cell in enumerate(row_cells):
                if cell is not None:
                    grid.addWidget(cell, row, col)
            grid.setRowStretch(row, row_stretch[row] if row < len(row_stretch) else 1)
        self.addTab(page, title)
        return page

    def set_feature_widget(self, widget: QWidget) -> None:
        """Puts the window's feature-vector box next to the PS/MTF ratio."""
        self._ratio_tab.layout().addWidget(widget, 0, 1, 2, 1)

    def show_steps(self, view: SliceStepsView | None, reason: str = "") -> None:
        has_steps = view is not None
        for index in range(self.count()):
            self.setTabVisible(index, has_steps != (index == self._empty_index))
        if not has_steps:
            self.empty.setText(reason or "No intermediate steps for this slice.")
            self.setCurrentIndex(self._empty_index)
            return
        if self.currentIndex() == self._empty_index:
            self.setCurrentIndex(0)
        self.roi1_edges.show(view.roi1_edges)
        self.esf.show_curves(view.esf)
        self.lsf.show_curves(view.lsf)
        self.mtf.show_curves(view.mtf)
        self.edge_map.show(view.edge_map)
        self.histogram.show_histogram(view.area_histogram)
        self.edge_counts.setText("\n".join(view.edge_counts))
        self.roi1.show(view.roi1)
        self.roi2.show(view.roi2)
        self.ps_2d.show(view.ps_2d)
        self.nps_2d.show(view.nps_2d)
        self.ratio.show_curves(view.ratio)
        self.ratio_features.setText("\n".join(view.ratio_features))
