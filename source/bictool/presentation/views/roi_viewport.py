"""Interactive image viewport: draw-then-assign ROI selection, live cursor
readout, pan/zoom.

Coordinate/cropping math lives in `viewmodels.roi_selection`, clinical
windowing math in `viewmodels.windowing` (both pure, testable without Qt);
this widget only turns mouse events into calls into that math and draws the
result.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QToolTip,
    QWidget,
)

from bictool.presentation.viewmodels.roi_selection import RoiRect, make_square, normalize_rect
from bictool.presentation.viewmodels.windowing import DEFAULT_WINDOW_PRESET, WINDOW_PRESETS, apply_window

logger = logging.getLogger(__name__)

# ROI slots: color, on-image label, ROI2 drawn dotted (ROI1 solid). The falx
# pair keeps the original ROI1/ROI2 colors; the septum pair gets its own. The
# plain "roi1"/"roi2" keys remain for callers without named pairs.
_ROI_STYLES = {
    "falx.roi1": ("#49c8de", "F1", False),
    "falx.roi2": ("#d98bd8", "F2", True),
    "septum.roi1": ("#6fcf97", "S1", False),
    "septum.roi2": ("#f2994a", "S2", True),
    "roi1": ("#49c8de", "ROI1", False),
    "roi2": ("#d98bd8", "ROI2", True),
}
_ROI_PEN_WIDTH = 1.5
_SELECTED_PEN_WIDTH = 3.5
# The in-progress/uncommitted rectangle - visually distinct (amber, dashed)
# from either committed ROI's solid color so "drawn but not yet saved" never
# looks like it's already ROI1 or ROI2.
_DRAFT_COLOR = "#f2c94c"

_ZOOM_STEP = 1.15
_MIN_ZOOM_SCALE = 0.2
_MAX_ZOOM_SCALE = 20.0

# Screen-space (not scene-space) corner text, matching the standard DICOM
# viewer convention of overlaying patient/study/series identity directly on
# the image (RadiAnt/OsiriX/Weasis all do this) rather than in a side panel -
# radiologists expect to read this without looking away from the image.
_OVERLAY_MARGIN = 10
_OVERLAY_ALIGNMENT = {
    "top_left": Qt.AlignmentFlag.AlignLeft,
    "top_right": Qt.AlignmentFlag.AlignRight,
    "bottom_left": Qt.AlignmentFlag.AlignLeft,
    "bottom_right": Qt.AlignmentFlag.AlignRight,
}


class RoiViewport(QGraphicsView):
    """Shows a windowed grayscale slice with draw-then-assign ROI selection.

    Interaction model: a plain left-drag always draws a single "draft"
    rectangle (`draft_changed`, amber/dashed) - there is no "arm" step
    first. The caller decides what to do with it via `commit_current_as`
    ("roi1"/"roi2"), which saves the current draft into that slot (drawn in
    that slot's solid color) without clearing it, so the same draft can be
    committed to the other slot too if wanted. Drawing a new rectangle
    replaces the previous draft. Panning is a middle-button drag (scrollbars
    are otherwise always hidden, see `__init__`); Ctrl+wheel zooms (centered
    on the cursor), plain wheel navigates slices (`wheelEvent`).
    Double-click resets zoom/pan to fit the view. Zoom/pan persists across
    slice changes (`set_image`) - only reset on double-click or first image.

    Hovering over the image (no button held) emits `cursor_moved` with the
    pixel coordinates and HU value under the cursor - the live "what am I
    looking at" readout real DICOM viewers show, driven the same way a
    QLabel-based status bar would be, just via a Qt signal so this widget
    stays ignorant of how the caller displays it (see `main_window`'s
    corner-overlay text).

    Also shows patient/study/series/technical identity as four screen-corner
    text labels (`set_overlay_text`), the standard DICOM-viewer convention
    (RadiAnt/OsiriX/Weasis) for keeping that information on the image itself
    instead of a side panel.

    Also supports named, semi-transparent mask overlays (`set_overlay`/
    `clear_overlay`) layered above the base image and below ROI rectangles -
    the rendering mechanism for FR-06, ready for whenever a real mask
    source (FR-02 automatic segmentation) exists. Cleared per-slice, same
    as ROI rectangles. While at least one overlay is visible, hovering over
    a masked pixel shows that overlay's display name (`set_overlay`'s
    `label` argument) as a native tooltip, so structures with similar
    overlay colors can still be told apart by hovering.
    """

    draft_changed = Signal(object)  # RoiRect | None - the in-progress/last-drawn, not-yet-necessarily-saved rectangle
    # (key, RoiRect) - something was dragged to a new position: key "" is the
    # hand-drawn draft (which also emits draft_changed), "roi1"/"roi2" a
    # committed ROI (only while not locked - see `set_locked`).
    draft_moved = Signal(str, object)
    # The committed ROI the user clicked (its key), or None when the selection
    # is cleared (click on empty image, slice change). See `select_roi`.
    roi_selected = Signal(object)
    # Delete/Backspace pressed while not locked, on the selected ROI (its key)
    # or - with no ROI selected - on the hand-drawn rectangle (key ""). The
    # caller decides what deleting means (see `main_window`).
    roi_delete_requested = Signal(str)
    # Right-click on an ROI (its key) or on the hand-drawn rectangle (""),
    # with the global screen position for a context menu.
    context_menu_requested = Signal(str, object)
    cursor_moved = Signal(object)  # (x: int, y: int, value_hu: float) | None - None when off-image or cursor leaves
    slice_scroll = Signal(int)  # +1 (next slice) or -1 (previous slice)
    zoom_changed = Signal(float)  # new zoom percentage - see `current_zoom_percent`

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#05070a")))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Left-drag always draws a draft rectangle (see class docstring) -
        # Qt's built-in hand-drag pan mode would fight that, so panning is
        # handled manually (middle-button, see mouse*Event below) and this
        # stays NoDrag permanently rather than toggling per "armed" state.
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Mouse-move events must fire even with no button held, for the
        # hover cursor/pixel-value readout.
        self.setMouseTracking(True)
        # This view always keeps the image either exactly fitted or panned
        # by our own drag handling - a visible scrollbar is never part of
        # the design, and letting Qt show/hide one automatically (the
        # default `ScrollBarAsNeeded`) risks a feedback loop where a
        # scrollbar appearing shrinks the viewport out from under an
        # already-computed fit, shifting the image by the scrollbar's width.
        # Manual middle-button panning still works with bars hidden - it
        # sets the scrollbars' values directly, not via their widgets.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Keyboard focus on click, so Delete reaches `keyPressEvent`.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._roi_items: dict[str, QGraphicsRectItem] = {}
        self._overlay_items: dict[str, QGraphicsPixmapItem] = {}
        # Raw mask data + a friendly display name per overlay key, kept
        # alongside `_overlay_items` purely so hovering the mouse can report
        # which named overlay(s) sit under the cursor (see
        # `_update_overlay_tooltip`) - the rendered pixmap alone can't answer
        # "which mask is this" once composited to RGBA.
        self._overlay_masks: dict[str, np.ndarray] = {}
        self._overlay_display_names: dict[str, str] = {}
        self._image_size = (0, 0)  # (width, height)
        self._drag_start: QPointF | None = None
        # Set while a left-drag that started inside the visible draft is
        # moving it (see `mousePressEvent`), instead of drawing a new one.
        self._move_start: QPointF | None = None
        self._move_origin: RoiRect | None = None
        self._move_key = ""  # "" = the hand-drawn draft, else the committed ROI being moved
        # Committed ROIs ("roi1"/"roi2") shown in their slot's solid color -
        # draggable until `set_locked(True)` (the "Confirmed" state), which
        # also stops new rectangles from being drawn.
        self._committed_rois: dict[str, RoiRect] = {}
        self._selected_key: str | None = None
        self._locked = False
        self._current_rect: RoiRect | None = None
        self._current_rect_item: QGraphicsRectItem | None = None
        self._pan_last_pos = None
        self._current_pixels_hu: np.ndarray | None = None
        self._window_width, self._window_center = WINDOW_PRESETS[DEFAULT_WINDOW_PRESET]
        self._is_zoomed = False

        # Parented to the viewport (not `self`/the scene) so this text stays
        # fixed to the widget's corners in screen pixels regardless of
        # zoom/pan - scene items would scale and drift with the image, which
        # is not how any real DICOM viewer's overlay behaves. Transparent to
        # mouse events so it never steals an ROI drag happening underneath.
        self._overlay_labels: dict[str, QLabel] = {}
        for corner, alignment in _OVERLAY_ALIGNMENT.items():
            label = QLabel(self.viewport())
            label.setAlignment(alignment)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            label.setStyleSheet(
                "QLabel { color: #e7eaed; background: rgba(5, 7, 10, 0.62); "
                "font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 10px; "
                "padding: 4px 7px; border-radius: 4px; }"
            )
            label.setVisible(False)
            self._overlay_labels[corner] = label

    def set_overlay_text(self, corner: str, lines: list[str]) -> None:
        """Sets the text shown in one screen corner ("top_left", "top_right",
        "bottom_left", "bottom_right"), or hides it if `lines` is empty.
        """
        label = self._overlay_labels[corner]
        label.setText("\n".join(lines))
        label.setVisible(bool(lines))
        label.adjustSize()
        self._reposition_overlay_labels()

    def _reposition_overlay_labels(self) -> None:
        vp_width, vp_height = self.viewport().width(), self.viewport().height()
        m = _OVERLAY_MARGIN
        self._overlay_labels["top_left"].move(m, m)
        self._overlay_labels["bottom_left"].move(m, vp_height - self._overlay_labels["bottom_left"].height() - m)
        for corner in ("top_right", "bottom_right"):
            label = self._overlay_labels[corner]
            y = m if corner == "top_right" else vp_height - label.height() - m
            label.move(vp_width - label.width() - m, y)

    def set_image(self, pixels_hu: np.ndarray) -> None:
        """Displays a new slice, windowed per the current WW/WL (see `set_window`).

        Resets ROI overlays (committed and draft) - a freshly displayed
        slice has no ROIs drawn on it yet (the caller, `main_window`, also
        resets its own ROI state) - but preserves the current zoom/pan (see
        class docstring).
        """
        self._current_pixels_hu = pixels_hu
        height, width = pixels_hu.shape
        is_first_image = self._pixmap_item is None
        size_changed = (width, height) != self._image_size
        self._image_size = (width, height)

        for item in self._roi_items.values():
            self._scene.removeItem(item)
        self._roi_items = {}
        if self._current_rect_item is not None:
            self._scene.removeItem(self._current_rect_item)
            self._current_rect_item = None
        self._current_rect = None
        self._move_start = None
        self._move_origin = None
        self._committed_rois = {}
        self.draft_changed.emit(None)
        self._set_selection(None)
        self.clear_all_overlays()  # overlays (e.g. segmentation masks) are per-slice too

        pixmap = self._make_windowed_pixmap()
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pixmap)

        if size_changed:
            self._scene.setSceneRect(0, 0, width, height)
        if is_first_image or not self._is_zoomed:
            self._fit_in_view()

    def clear(self) -> None:
        """Removes the displayed image and all ROI/overlay/draft state -
        back to the pre-`set_image` empty viewport, for `main_window`'s
        "Close Study" action. Unlike `set_image`, does not replace the
        pixmap with a new one - there is no image to show at all afterwards.
        """
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        for item in self._roi_items.values():
            self._scene.removeItem(item)
        self._roi_items = {}
        if self._current_rect_item is not None:
            self._scene.removeItem(self._current_rect_item)
            self._current_rect_item = None
        self._current_rect = None
        self._move_start = None
        self._move_origin = None
        self._committed_rois = {}
        self.draft_changed.emit(None)
        self._set_selection(None)
        self.clear_all_overlays()
        self._current_pixels_hu = None
        self._image_size = (0, 0)
        self._is_zoomed = False
        for corner in self._overlay_labels:
            self.set_overlay_text(corner, [])

    def set_window(self, width: float, center: float) -> None:
        """Re-windows the currently displayed slice (WW/WL), keeping ROI overlays intact."""
        self._window_width = width
        self._window_center = center
        if self._current_pixels_hu is not None and self._pixmap_item is not None:
            self._pixmap_item.setPixmap(self._make_windowed_pixmap())

    def current_window(self) -> tuple[float, float]:
        """Current (width, center) WW/WL, for the caller's corner-overlay text."""
        return self._window_width, self._window_center

    def current_zoom_percent(self) -> float:
        """Current zoom as a percentage where 100% means one image pixel is
        shown as one screen pixel (the common uncalibrated-display
        convention real viewers use, e.g. RadiAnt's `Zoom: 145%`) - this
        tool has no monitor-calibration step, so it can't offer a
        true-physical-size 100% the way a calibrated PACS workstation would.
        """
        return self.transform().m11() * 100.0

    def _fit_in_view(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.current_zoom_percent())

    def _make_windowed_pixmap(self) -> QPixmap:
        height, width = self._current_pixels_hu.shape
        img8 = np.ascontiguousarray(apply_window(self._current_pixels_hu, self._window_width, self._window_center))
        image = QImage(img8.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(image)

    def set_draft(self, rect: RoiRect) -> None:
        """Shows `rect` as the draft, exactly as if it had just been drawn -
        used to place an automatically selected ROI that the user can then
        move (drag inside it) and accept with Set ROI1/ROI2.
        """
        self._current_rect = rect
        self._draw_current_rect(rect)
        self.draft_changed.emit(rect)

    def has_draft(self) -> bool:
        """A hand-drawn rectangle is visible (drawn, not yet saved as an ROI)."""
        return self._current_rect is not None and self._current_rect_item is not None

    def clear_draft(self) -> None:
        """Removes the draft (data and on-screen preview) without touching committed ROIs."""
        if self._current_rect_item is not None:
            self._scene.removeItem(self._current_rect_item)
            self._current_rect_item = None
        self._current_rect = None
        self.draft_changed.emit(None)

    def set_roi(self, roi_key: str, rect: RoiRect) -> None:
        """Shows `rect` as committed ROI `roi_key` (e.g. "falx.roi1", see
        `_ROI_STYLES`) in that slot's color and label, replacing whatever was
        shown there - e.g. an automatically selected ROI, or a slice's saved
        ROI after navigating.
        """
        self._draw_roi(roi_key, rect)

    def select_roi(self, roi_key: str | None) -> None:
        """Highlights a committed ROI (or none) - e.g. when its row is clicked
        in the ROI table. Emits `roi_selected` when the selection changes."""
        if roi_key is not None and roi_key not in self._committed_rois:
            roi_key = None
        self._set_selection(roi_key)

    def selected_roi(self) -> str | None:
        return self._selected_key

    def _set_selection(self, roi_key: str | None) -> None:
        if roi_key == self._selected_key:
            return
        previous, self._selected_key = self._selected_key, roi_key
        for key in (previous, roi_key):
            if key is not None and key in self._committed_rois:
                self._draw_roi(key, self._committed_rois[key])
        self.roi_selected.emit(roi_key)

    def roi(self, roi_key: str) -> RoiRect | None:
        return self._committed_rois.get(roi_key)

    def clear_rois(self) -> None:
        """Removes every committed ROI from the image. The selection key is
        kept, so redrawing the same ROIs (e.g. after an edit) keeps it
        highlighted; `select_roi(None)` clears it."""
        for item in self._roi_items.values():
            self._scene.removeItem(item)
        self._roi_items = {}
        self._committed_rois = {}

    def set_locked(self, locked: bool) -> None:
        """Locked (the ROIs have been confirmed): left-drags neither move ROIs
        nor draw new rectangles.
        """
        self._locked = locked
        self._drag_start = None
        self._move_start = None

    def commit_current_as(self, roi_key: str) -> RoiRect | None:
        """Saves the current draft rectangle (the last one drawn - see
        `draft_changed`) as `roi_key` ("roi1" or "roi2"), drawing it in that
        slot's solid color and replacing whatever was previously saved
        there. Returns the saved rect, or None if nothing has been drawn
        yet (the caller should keep the Set ROI1/2 buttons disabled until a
        first `draft_changed` with a non-None rect has fired).

        The draft itself is *not* cleared afterwards (only its on-screen
        dashed preview is, since it's now shown as the committed color
        instead) - the same draft can still be committed to the other slot
        too, since the request was "draw once, decide which slot(s) after".
        """
        if self._current_rect is None:
            return None
        self._draw_roi(roi_key, self._current_rect)
        if self._current_rect_item is not None:
            self._scene.removeItem(self._current_rect_item)
            self._current_rect_item = None
        return self._current_rect

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        super().resizeEvent(event)
        if self._pixmap_item is not None and not self._is_zoomed:
            self._fit_in_view()
        self._reposition_overlay_labels()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_item is not None and not self._locked:
            self.setFocus()
            scene_pos = self.mapToScene(event.position().toPoint())
            grabbed = self._draft_under(scene_pos)
            if grabbed is not None:
                # Grabbing a draft or ROI moves it (e.g. to adjust an
                # auto-selected ROI) and selects it (Delete removes it);
                # pressing anywhere else draws a new one.
                self._move_key, self._move_origin = grabbed
                self._move_start = scene_pos
                self._set_selection(self._move_key or None)
            else:
                self._set_selection(None)
                self._drag_start = scene_pos
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        """Delete/Backspace asks the caller to delete the selected ROI, or the
        hand-drawn rectangle when no ROI is selected."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and not self._locked:
            if self._selected_key is not None:
                self.roi_delete_requested.emit(self._selected_key)
                event.accept()
                return
            if self.has_draft():
                self.roi_delete_requested.emit("")
                event.accept()
                return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        """Right-click on an ROI or the hand-drawn rectangle: selects it and
        asks the caller for its menu (Delete, Save as ...)."""
        if self._pixmap_item is None:
            return
        hit = self._draft_under(self.mapToScene(event.pos()))
        if hit is None:
            return
        key = hit[0]
        self._set_selection(key or None)
        self.context_menu_requested.emit(key, event.globalPos())
        event.accept()

    def _draft_under(self, scene_pos) -> tuple[str, RoiRect] | None:
        """(key, rect) of the visible draft or committed ROI under `scene_pos` -
        the hand-drawn draft first (drawn most recently, on top), then ROI1, ROI2.
        """
        drafts = []
        if self._current_rect is not None and self._current_rect_item is not None:
            drafts.append(("", self._current_rect))
        drafts.extend(sorted(self._committed_rois.items()))
        for key, rect in drafts:
            if rect.x0 <= scene_pos.x() < rect.x1 and rect.y0 <= scene_pos.y() < rect.y1:
                return key, rect
        return None

    def _moved_draft(self, view_pos) -> RoiRect:
        """`_move_origin` translated by the drag so far, clamped inside the image (size unchanged)."""
        pos = self.mapToScene(view_pos)
        origin = self._move_origin
        width, height = self._image_size
        dx = int(round(pos.x() - self._move_start.x()))
        dy = int(round(pos.y() - self._move_start.y()))
        x0 = max(0, min(origin.x0 + dx, width - origin.width))
        y0 = max(0, min(origin.y0 + dy, height - origin.height))
        return RoiRect(x0, y0, x0 + origin.width, y0 + origin.height)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        """Three mutually exclusive behaviors, in priority order: an active
        middle-button pan; an active left-drag, which live-updates the draft
        rectangle and emits `draft_changed` (using the exact same squared
        rect `mouseReleaseEvent` will finalize, so nothing changes shape at
        the moment of release); otherwise, if an image is loaded, a hover
        cursor/pixel-value reading (`cursor_moved`).
        """
        if self._pan_last_pos is not None:
            delta = event.position().toPoint() - self._pan_last_pos
            self._pan_last_pos = event.position().toPoint()
            h_bar, v_bar = self.horizontalScrollBar(), self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            event.accept()
            return
        if self._move_start is not None:
            rect = self._moved_draft(event.position().toPoint())
            if self._move_key:
                if rect != self._committed_rois.get(self._move_key):
                    self.set_roi(self._move_key, rect)
                    self.draft_moved.emit(self._move_key, rect)
            elif rect != self._current_rect:
                self._current_rect = rect
                self._draw_current_rect(rect)
                self.draft_changed.emit(rect)
                self.draft_moved.emit("", rect)
        elif self._drag_start is not None:
            rect = self._rect_from_drag_endpoint(event.position().toPoint())
            if rect is not None:
                self._current_rect = rect
                self._draw_current_rect(rect)
                self.draft_changed.emit(rect)
        elif self._current_pixels_hu is not None:
            self._emit_cursor_reading(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_last_pos is not None:
            self._pan_last_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._move_start is not None:
            self._move_start = None
            self._move_origin = None
            super().mouseReleaseEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            rect = self._rect_from_drag_endpoint(event.position().toPoint())
            self._drag_start = None
            if rect is not None:
                self._current_rect = rect
                self._draw_current_rect(rect)
                self.draft_changed.emit(rect)
            else:
                logger.warning(
                    "Drag produced an empty rectangle (no selection made) - the drag may have been "
                    "outside the image, or too short to register."
                )
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        self.cursor_moved.emit(None)
        QToolTip.hideText()
        super().leaveEvent(event)

    def _emit_cursor_reading(self, view_pos) -> None:
        scene_pos = self.mapToScene(view_pos)
        x, y = int(scene_pos.x()), int(scene_pos.y())
        width, height = self._image_size
        if 0 <= x < width and 0 <= y < height:
            value = float(self._current_pixels_hu[y, x])
            self.cursor_moved.emit((x, y, value))
            self._update_overlay_tooltip(view_pos, x, y)
        else:
            self.cursor_moved.emit(None)
            QToolTip.hideText()

    def _update_overlay_tooltip(self, view_pos, x: int, y: int) -> None:
        """Shows a native tooltip naming whichever visible overlay(s) sit
        under (x, y) - lets structures sharing a similar overlay color
        still be told apart by hovering (see the class docstring).
        """
        names = [
            self._overlay_display_names[key]
            for key, mask in self._overlay_masks.items()
            if mask[y, x]
        ]
        if names:
            QToolTip.showText(self.mapToGlobal(view_pos), ", ".join(names), self)
        else:
            QToolTip.hideText()

    def _rect_from_drag_endpoint(self, view_pos) -> RoiRect | None:
        """Builds the (squared) `RoiRect` from `self._drag_start` to
        `view_pos`, or None if that produces an empty rectangle (e.g. no
        real drag distance yet, or the drag is outside the image).
        """
        end = self.mapToScene(view_pos)
        width, height = self._image_size
        raw_rect = normalize_rect(
            int(self._drag_start.x()), int(self._drag_start.y()), int(end.x()), int(end.y()), width, height
        )
        if raw_rect.width <= 0 or raw_rect.height <= 0:
            return None
        # calc_digital_nps requires an exactly square ROI - a manual drag
        # essentially never lands on one by hand, so square it here (see
        # make_square's docstring). Live preview and final selection both
        # use this same squared rect, so nothing changes shape at release.
        rect = make_square(raw_rect, width, height)
        if rect != raw_rect:
            logger.debug("ROI drag: squared %dx%d to %dx%d", raw_rect.width, raw_rect.height, rect.width, rect.height)
        return rect

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        """Resets zoom/pan to fit the current slice in the viewport."""
        self._is_zoomed = False
        if self._pixmap_item is not None:
            self._fit_in_view()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        """Ctrl+wheel zooms (centered on the cursor); plain wheel navigates slices."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = _ZOOM_STEP if event.angleDelta().y() > 0 else 1 / _ZOOM_STEP
            new_scale = self.transform().m11() * factor
            if _MIN_ZOOM_SCALE <= new_scale <= _MAX_ZOOM_SCALE:
                self.scale(factor, factor)
                self._is_zoomed = True
                self.zoom_changed.emit(self.current_zoom_percent())
            event.accept()
            return

        delta = event.angleDelta().y()
        if delta != 0:
            self.slice_scroll.emit(1 if delta < 0 else -1)
            event.accept()
        else:
            super().wheelEvent(event)

    def _draw_roi(self, key: str, rect: RoiRect) -> None:
        existing = self._roi_items.get(key)
        if existing is not None:
            self._scene.removeItem(existing)
        color_hex, label, dotted = _ROI_STYLES[key]
        color = QColor(color_hex)
        fill = QColor(color)
        fill.setAlpha(45)
        selected = key == self._selected_key
        pen = QPen(color, _SELECTED_PEN_WIDTH if selected else _ROI_PEN_WIDTH)
        if dotted:
            pen.setStyle(Qt.PenStyle.DotLine)
        item = self._scene.addRect(rect.x0, rect.y0, rect.width, rect.height, pen, QBrush(fill))
        item.setZValue(3 if selected else 2)  # above overlays (z=1) and the base image (z=0)
        # Constant on-screen size whatever the zoom - a label, not part of the image.
        text = QGraphicsSimpleTextItem(label, item)
        text.setBrush(QBrush(color))
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        text.setPos(rect.x0 + 0.5, rect.y0 + 0.5)
        self._roi_items[key] = item
        self._committed_rois[key] = rect

    def _draw_current_rect(self, rect: RoiRect) -> None:
        if self._current_rect_item is not None:
            self._scene.removeItem(self._current_rect_item)
        self._current_rect_item = self._add_dashed_rect(rect, _DRAFT_COLOR)

    def _add_dashed_rect(self, rect: RoiRect, color_hex: str) -> QGraphicsRectItem:
        color = QColor(color_hex)
        fill = QColor(color)
        fill.setAlpha(35)
        pen = QPen(color, 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        item = self._scene.addRect(rect.x0, rect.y0, rect.width, rect.height, pen, QBrush(fill))
        item.setZValue(2)
        return item

    def set_overlay(
        self, key: str, mask: np.ndarray, color: str = "#5fbf83", opacity: float = 0.6, label: str | None = None
    ) -> None:
        """Displays `mask` (2D boolean/0-1 array, same shape as the current
        slice) as a semi-transparent colored layer above the base image,
        keyed by `key` so multiple named overlays (e.g. one per segmented
        structure) can coexist and be independently replaced/cleared.

        `label` is what the hover tooltip shows for this overlay (see the
        class docstring) - defaults to `key` itself if not given, but
        callers keying overlays with an internal prefix (e.g. `f"seg_
        {display_name}"`) should pass the clean `display_name` here instead.

        This is the rendering mechanism for FR-06 (visualize segmented
        structures/metrics/noise maps); it does not generate masks itself -
        automatic segmentation (FR-02) is a separate, not-yet-implemented
        concern, so nothing calls this with real data yet.
        """
        expected_shape = (self._image_size[1], self._image_size[0])  # (height, width)
        if mask.shape != expected_shape:
            raise ValueError(f"Overlay mask shape {mask.shape} does not match image shape {expected_shape}")

        self.clear_overlay(key)

        height, width = mask.shape
        mask_bool = mask.astype(bool)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        qcolor = QColor(color)
        rgba[..., 0] = qcolor.red()
        rgba[..., 1] = qcolor.green()
        rgba[..., 2] = qcolor.blue()
        rgba[..., 3] = np.where(mask_bool, round(255 * opacity), 0).astype(np.uint8)

        image = QImage(np.ascontiguousarray(rgba).tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
        item = self._scene.addPixmap(QPixmap.fromImage(image))
        item.setZValue(1)  # above the base image (z=0), below ROI rects (z=2)
        self._overlay_items[key] = item
        self._overlay_masks[key] = mask_bool
        self._overlay_display_names[key] = label if label is not None else key

    def clear_overlay(self, key: str) -> None:
        """Removes a previously set overlay layer, if present. No-op otherwise."""
        item = self._overlay_items.pop(key, None)
        if item is not None:
            self._scene.removeItem(item)
        self._overlay_masks.pop(key, None)
        self._overlay_display_names.pop(key, None)

    def clear_all_overlays(self) -> None:
        for key in list(self._overlay_items):
            self.clear_overlay(key)
