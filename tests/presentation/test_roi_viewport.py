import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from bictool.presentation.views.roi_viewport import RoiViewport


def _make_wheel_event(viewport: RoiViewport, *, angle_delta_y: int, ctrl: bool) -> QWheelEvent:
    modifiers = Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    pos = QPointF(viewport.viewport().rect().center())
    return QWheelEvent(
        pos,
        viewport.mapToGlobal(pos.toPoint()).toPointF(),
        QPoint(0, 0),
        QPoint(0, angle_delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_default_drag_mode_is_no_drag_since_left_drag_always_draws():
    # Left-drag draws a rectangle by default (no "arm" step) - Qt's built-in
    # hand-drag pan mode would fight that, so this view never uses it;
    # panning is a manual middle-button drag instead (see below).
    viewport = RoiViewport()
    assert viewport.dragMode() == QGraphicsView.DragMode.NoDrag


def test_middle_button_drag_pans_by_adjusting_scrollbar_values():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((400, 400), dtype=np.float64))
    viewport.resize(100, 100)
    viewport.scale(3.0, 3.0)  # zoom in first - nothing to pan when fully fit
    h_bar = viewport.horizontalScrollBar()
    start_value = h_bar.value()

    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(50, 50),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewport.mousePressEvent(press)
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(20, 50),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewport.mouseMoveEvent(move)

    assert h_bar.value() != start_value
    # Dragging is still NoDrag throughout - middle-button pan is handled
    # manually, not via Qt's ScrollHandDrag.
    assert viewport.dragMode() == QGraphicsView.DragMode.NoDrag

    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(20, 50),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewport.mouseReleaseEvent(release)
    assert viewport._pan_last_pos is None


def test_ctrl_wheel_zooms_instead_of_scrolling_slices():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((400, 400), dtype=np.float64))
    viewport.resize(200, 200)
    initial_scale = viewport.transform().m11()

    slice_scroll_calls = []
    viewport.slice_scroll.connect(slice_scroll_calls.append)

    event = _make_wheel_event(viewport, angle_delta_y=120, ctrl=True)
    viewport.wheelEvent(event)

    assert viewport.transform().m11() > initial_scale
    assert viewport._is_zoomed is True
    assert slice_scroll_calls == []


def test_plain_wheel_still_scrolls_slices_and_does_not_zoom():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    viewport.resize(200, 200)
    initial_scale = viewport.transform().m11()

    slice_scroll_calls = []
    viewport.slice_scroll.connect(slice_scroll_calls.append)

    event = _make_wheel_event(viewport, angle_delta_y=120, ctrl=False)
    viewport.wheelEvent(event)

    assert viewport.transform().m11() == initial_scale
    assert slice_scroll_calls == [-1]


def test_double_click_resets_zoom():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    viewport.resize(200, 200)
    viewport.scale(3.0, 3.0)
    viewport._is_zoomed = True

    double_click = QMouseEvent(
        QMouseEvent.Type.MouseButtonDblClick,
        QPointF(100, 100),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewport.mouseDoubleClickEvent(double_click)

    assert viewport._is_zoomed is False


def test_set_image_preserves_zoom_across_slice_change():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    viewport.resize(200, 200)
    viewport.scale(3.0, 3.0)
    viewport._is_zoomed = True
    zoomed_scale = viewport.transform().m11()

    viewport.set_image(np.ones((20, 20), dtype=np.float64) * 5.0)

    assert viewport.transform().m11() == zoomed_scale


def test_set_overlay_adds_a_layer_above_the_base_image():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True

    viewport.set_overlay("skull", mask, color="#ff0000")

    assert "skull" in viewport._overlay_items
    overlay_item = viewport._overlay_items["skull"]
    assert overlay_item.zValue() > viewport._pixmap_item.zValue()


def test_set_overlay_rejects_mismatched_mask_shape():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    wrong_shape_mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="shape"):
        viewport.set_overlay("skull", wrong_shape_mask)


def test_clear_overlay_removes_it():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    mask = np.ones((20, 20), dtype=bool)
    viewport.set_overlay("skull", mask)

    viewport.clear_overlay("skull")

    assert "skull" not in viewport._overlay_items


def test_clear_overlay_on_unknown_key_is_a_noop():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    viewport.clear_overlay("does-not-exist")  # must not raise


def test_set_image_clears_overlays_from_the_previous_slice():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    mask = np.ones((20, 20), dtype=bool)
    viewport.set_overlay("skull", mask)

    viewport.set_image(np.ones((20, 20), dtype=np.float64) * 5.0)

    assert viewport._overlay_items == {}


def test_multiple_overlays_coexist_independently():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    mask_a = np.zeros((20, 20), dtype=bool)
    mask_a[:10, :10] = True
    mask_b = np.zeros((20, 20), dtype=bool)
    mask_b[10:, 10:] = True

    viewport.set_overlay("skull", mask_a, color="#ff0000")
    viewport.set_overlay("brain", mask_b, color="#00ff00")

    assert set(viewport._overlay_items.keys()) == {"skull", "brain"}

    viewport.clear_overlay("skull")

    assert set(viewport._overlay_items.keys()) == {"brain"}


def test_set_image_fits_view_when_not_zoomed():
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    fit_scale = viewport.transform().m11()

    viewport.set_image(np.ones((20, 20), dtype=np.float64) * 5.0)

    assert viewport.transform().m11() == fit_scale


def _move_event(pos: tuple[int, int]) -> QMouseEvent:
    return QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(*pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,  # buttons held during the move
        Qt.KeyboardModifier.NoModifier,
    )


def _press_event(pos: tuple[int, int]) -> QMouseEvent:
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(*pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def test_mouse_move_during_drag_emits_live_preview_and_draws_rect():
    # No arming needed - any left-drag draws a rectangle by default.
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))

    viewport.mousePressEvent(_press_event((20, 20)))
    live_rects = []
    viewport.draft_changed.connect(live_rects.append)

    viewport.mouseMoveEvent(_move_event((60, 60)))

    assert len(live_rects) == 1
    rect = live_rects[0]
    assert rect.width > 0 and rect.height > 0
    assert rect.width == rect.height  # live preview is already squared
    assert viewport._current_rect_item is not None  # drawn live, not only on release


def test_mouse_move_without_a_preceding_press_does_not_draft_but_reads_cursor():
    # No mousePressEvent happened - a stray move (e.g. plain hovering) must
    # not be treated as resizing a draft rectangle; it's a hover reading.
    # Viewport sized to match the image 1:1 (and shown - exact viewport()
    # geometry needs a real show/layout pass, see other tests in this file)
    # so view/scene coordinates agree with no scale-factor rounding.
    viewport = RoiViewport()
    viewport.show()
    viewport.resize(100, 100)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))

    drafts = []
    viewport.draft_changed.connect(drafts.append)
    readings = []
    viewport.cursor_moved.connect(readings.append)
    viewport.mouseMoveEvent(_move_event((60, 60)))

    assert drafts == []
    assert readings == [(60, 60, 0.0)]


def test_mouse_move_tiny_drag_does_not_emit_empty_rect():
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    viewport.mousePressEvent(_press_event((20, 20)))

    live_rects = []
    viewport.draft_changed.connect(live_rects.append)
    viewport.mouseMoveEvent(_move_event((20, 20)))  # no movement yet

    assert live_rects == []


def test_live_preview_rect_matches_final_release_rect():
    # The whole point of squaring during the live preview too: what the
    # user sees while dragging must be exactly what gets used on release,
    # with no jarring resize the instant they let go.
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    viewport.mousePressEvent(_press_event((20, 20)))

    live_rects = []
    viewport.draft_changed.connect(live_rects.append)
    viewport.mouseMoveEvent(_move_event((60, 45)))

    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(60, 45),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewport.mouseReleaseEvent(release)

    assert live_rects[-2] == live_rects[-1]  # mid-drag preview, then release - same rect


def test_commit_current_as_returns_none_when_nothing_drawn():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))

    assert viewport.commit_current_as("roi1") is None


def test_commit_current_as_saves_the_draft_and_draws_it_in_that_slots_color():
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    viewport.mousePressEvent(_press_event((20, 20)))
    viewport.mouseMoveEvent(_move_event((60, 45)))
    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(60, 45),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewport.mouseReleaseEvent(release)

    saved = viewport.commit_current_as("roi1")

    assert saved is not None
    assert "roi1" in viewport._roi_items
    assert viewport._current_rect_item is None  # draft preview removed once committed


def test_commit_current_as_can_save_the_same_draft_to_both_slots():
    # "draw once, decide which slot(s) after" - committing shouldn't
    # prevent also committing the same draft to the other slot.
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    viewport.mousePressEvent(_press_event((20, 20)))
    viewport.mouseMoveEvent(_move_event((60, 45)))

    first = viewport.commit_current_as("roi1")
    second = viewport.commit_current_as("roi2")

    assert first == second
    assert "roi1" in viewport._roi_items and "roi2" in viewport._roi_items


def test_set_image_clears_the_draft_and_emits_none():
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    viewport.mousePressEvent(_press_event((20, 20)))
    viewport.mouseMoveEvent(_move_event((60, 45)))
    assert viewport._current_rect is not None

    drafts = []
    viewport.draft_changed.connect(drafts.append)
    viewport.set_image(np.ones((100, 100), dtype=np.float64))

    assert viewport._current_rect is None
    assert viewport._current_rect_item is None
    assert drafts == [None]


def test_hover_emits_pixel_coordinates_and_hu_value():
    # Viewport matches the image 1:1 (and shown) so view/scene coordinates
    # agree exactly with no scale-factor rounding to account for.
    viewport = RoiViewport()
    viewport.show()
    viewport.resize(100, 100)
    pixels = np.zeros((100, 100), dtype=np.float64)
    pixels[28:33, 58:63] = 42.0  # a small block, not one pixel - tolerant of the 1px rounding noted below
    viewport.set_image(pixels)

    readings = []
    viewport.cursor_moved.connect(readings.append)
    viewport.mouseMoveEvent(_move_event((60, 30)))

    assert len(readings) == 1
    x, y, value = readings[0]
    # `fitInView`'s scale is only /approximately/ 1:1 in practice (sub-pixel
    # rounding in the real viewport geometry) - a 1px tolerance avoids this
    # test being brittle to that, while still catching a real axis-swap or
    # scaling bug (which would be off by far more than 1px).
    assert x == pytest.approx(60, abs=1)
    assert y == pytest.approx(30, abs=1)
    assert value == pytest.approx(42.0)


def test_hover_off_image_emits_none():
    viewport = RoiViewport()
    viewport.resize(200, 200)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))

    readings = []
    viewport.cursor_moved.connect(readings.append)
    viewport.mouseMoveEvent(_move_event((-5, -5)))

    assert readings == [None]


def test_leave_event_emits_none():
    from PySide6.QtCore import QEvent

    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))

    readings = []
    viewport.cursor_moved.connect(readings.append)
    viewport.leaveEvent(QEvent(QEvent.Type.Leave))

    assert readings == [None]


def test_set_overlay_text_shows_and_positions_label_in_its_corner():
    viewport = RoiViewport()
    viewport.resize(300, 200)

    viewport.set_overlay_text("top_left", ["Series A", "Slice 1 / 5"])

    label = viewport._overlay_labels["top_left"]
    # `isVisible()` is unreliable on a widget whose top-level window was
    # never `.show()`n (always False regardless of `setVisible` calls) -
    # `isHidden()` reflects the explicit setVisible(True/False) state.
    assert not label.isHidden()
    assert label.text() == "Series A\nSlice 1 / 5"
    assert label.x() == 10 and label.y() == 10  # pinned to the top-left corner, margin included


def test_set_overlay_text_empty_list_hides_the_label():
    viewport = RoiViewport()
    viewport.resize(300, 200)
    viewport.set_overlay_text("bottom_right", ["CTDIvol: 12.0 mGy"])
    assert not viewport._overlay_labels["bottom_right"].isHidden()

    viewport.set_overlay_text("bottom_right", [])

    assert viewport._overlay_labels["bottom_right"].isHidden()


def test_right_and_bottom_corner_labels_stay_pinned_to_the_far_edge_after_resize():
    viewport = RoiViewport()
    viewport.show()  # exact viewport() geometry needs a real show/layout pass, unlike other tests here
    viewport.resize(300, 200)
    viewport.set_overlay_text("bottom_right", ["CTDIvol: 12.0 mGy", "DLP: 210 mGy·cm"])
    label = viewport._overlay_labels["bottom_right"]
    vp_width, vp_height = viewport.viewport().width(), viewport.viewport().height()
    right_edge = label.x() + label.width()
    bottom_edge = label.y() + label.height()
    assert right_edge == pytest.approx(vp_width - 10, abs=1)
    assert bottom_edge == pytest.approx(vp_height - 10, abs=1)

    viewport.resize(500, 400)  # resizeEvent must reposition, not just resizeEvent-on-set

    vp_width, vp_height = viewport.viewport().width(), viewport.viewport().height()
    right_edge = label.x() + label.width()
    bottom_edge = label.y() + label.height()
    assert right_edge == pytest.approx(vp_width - 10, abs=1)
    assert bottom_edge == pytest.approx(vp_height - 10, abs=1)


def test_overlay_labels_are_transparent_to_mouse_events_so_roi_dragging_still_works_underneath():
    viewport = RoiViewport()
    viewport.set_overlay_text("top_left", ["Series A"])
    label = viewport._overlay_labels["top_left"]
    assert label.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_clear_removes_the_image_and_all_roi_and_draft_state():
    viewport = RoiViewport()
    viewport.set_image(np.zeros((20, 20), dtype=np.float64))
    viewport.mousePressEvent(_press_event((2, 2)))
    viewport.mouseMoveEvent(_move_event((15, 15)))
    viewport.commit_current_as("roi1")
    viewport.set_overlay_text("top_left", ["Series A"])

    drafts = []
    viewport.draft_changed.connect(drafts.append)
    viewport.clear()

    assert viewport._pixmap_item is None
    assert viewport._roi_items == {}
    assert viewport._current_rect is None
    assert viewport._current_pixels_hu is None
    assert drafts == [None]
    for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
        assert viewport._overlay_labels[corner].isHidden()


def test_clear_on_a_never_shown_viewport_does_not_raise():
    # Close Study can be clicked before any image was ever loaded in some
    # test setups - clear() must be a safe no-op-ish call either way.
    viewport = RoiViewport()
    viewport.clear()  # must not raise
