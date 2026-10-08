import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from bictool.presentation.viewmodels.roi_selection import RoiRect
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
        QPointF(50, 50),  # global position (Qt 6 constructor)
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewport.mousePressEvent(press)
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(20, 50),
        QPointF(20, 50),  # global position (Qt 6 constructor)
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
        QPointF(20, 50),  # global position (Qt 6 constructor)
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
        QPointF(100, 100),  # global position (Qt 6 constructor)
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
        QPointF(*pos),  # global position (Qt 6 constructor)
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,  # buttons held during the move
        Qt.KeyboardModifier.NoModifier,
    )


def _press_event(pos: tuple[int, int]) -> QMouseEvent:
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(*pos),
        QPointF(*pos),  # global position (Qt 6 constructor)
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
        QPointF(60, 45),  # global position (Qt 6 constructor)
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
        QPointF(60, 45),  # global position (Qt 6 constructor)
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


def test_set_overlay_stores_the_mask_and_display_name_for_hover_lookup():
    viewport = RoiViewport()
    viewport.resize(100, 100)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True

    viewport.set_overlay("seg_Ventricular System", mask, label="Ventricular System")

    assert viewport._overlay_masks["seg_Ventricular System"][15, 15]
    assert viewport._overlay_display_names["seg_Ventricular System"] == "Ventricular System"


def test_set_overlay_label_defaults_to_the_key():
    viewport = RoiViewport()
    viewport.resize(100, 100)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))

    viewport.set_overlay("stroke_lesion", np.zeros((100, 100), dtype=bool))

    assert viewport._overlay_display_names["stroke_lesion"] == "stroke_lesion"


def test_clear_overlay_removes_the_mask_and_display_name():
    viewport = RoiViewport()
    viewport.resize(100, 100)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    viewport.set_overlay("tissue_White Matter", np.ones((100, 100), dtype=bool), label="White Matter")

    viewport.clear_overlay("tissue_White Matter")

    assert "tissue_White Matter" not in viewport._overlay_masks
    assert "tissue_White Matter" not in viewport._overlay_display_names


def test_hovering_a_masked_pixel_shows_a_tooltip_with_the_overlay_name(monkeypatch):
    viewport = RoiViewport()
    viewport.show()
    viewport.resize(100, 100)
    pixels = np.zeros((100, 100), dtype=np.float64)
    viewport.set_image(pixels)
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:35, 55:65] = True
    viewport.set_overlay("seg_Ventricular System", mask, label="Ventricular System")

    shown = []
    monkeypatch.setattr("bictool.presentation.views.roi_viewport.QToolTip.showText", lambda pos, text, w: shown.append(text))
    hidden = []
    monkeypatch.setattr("bictool.presentation.views.roi_viewport.QToolTip.hideText", lambda: hidden.append(True))

    viewport.mouseMoveEvent(_move_event((60, 30)))

    assert shown == ["Ventricular System"]
    assert hidden == []


def test_hovering_outside_any_overlay_hides_the_tooltip(monkeypatch):
    viewport = RoiViewport()
    viewport.show()
    viewport.resize(100, 100)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:35, 55:65] = True
    viewport.set_overlay("seg_Ventricular System", mask, label="Ventricular System")

    shown = []
    monkeypatch.setattr("bictool.presentation.views.roi_viewport.QToolTip.showText", lambda pos, text, w: shown.append(text))
    hidden = []
    monkeypatch.setattr("bictool.presentation.views.roi_viewport.QToolTip.hideText", lambda: hidden.append(True))

    viewport.mouseMoveEvent(_move_event((5, 5)))  # well outside the mask

    assert shown == []
    assert hidden == [True]


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


def _release_event(pos: tuple[int, int]) -> QMouseEvent:
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(*pos),
        QPointF(*pos),  # global position (Qt 6 constructor)
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _one_to_one_viewport() -> RoiViewport:
    # Shown and sized to match the image so view and scene coordinates agree
    # (see test_mouse_move_without_a_preceding_press_does_not_draft_but_reads_cursor).
    viewport = RoiViewport()
    viewport.show()
    viewport.resize(100, 100)
    viewport.set_image(np.zeros((100, 100), dtype=np.float64))
    return viewport


def test_set_draft_draws_and_emits_like_a_drawn_rect():
    viewport = _one_to_one_viewport()
    drafts = []
    viewport.draft_changed.connect(drafts.append)

    viewport.set_draft(RoiRect(10, 10, 30, 30))

    assert drafts == [RoiRect(10, 10, 30, 30)]
    assert viewport._current_rect_item is not None
    assert viewport.commit_current_as("roi1") == RoiRect(10, 10, 30, 30)


def test_clear_draft_removes_the_draft():
    viewport = _one_to_one_viewport()
    viewport.set_draft(RoiRect(10, 10, 30, 30))
    drafts = []
    viewport.draft_changed.connect(drafts.append)

    viewport.clear_draft()

    assert drafts == [None]
    assert viewport._current_rect_item is None
    assert viewport.commit_current_as("roi1") is None


def test_dragging_inside_the_draft_moves_it_and_keeps_its_size():
    viewport = _one_to_one_viewport()
    viewport.set_draft(RoiRect(10, 10, 30, 30))

    viewport.mousePressEvent(_press_event((20, 20)))
    viewport.mouseMoveEvent(_move_event((35, 25)))
    viewport.mouseReleaseEvent(_release_event((35, 25)))

    rect = viewport._current_rect
    assert (rect.width, rect.height) == (20, 20)
    # Moved by the drag delta (15, 5); +/-1 px for view-to-scene rounding.
    assert abs(rect.x0 - 25) <= 1 and abs(rect.y0 - 15) <= 1


def test_moving_the_draft_emits_draft_moved():
    viewport = _one_to_one_viewport()
    viewport.set_draft(RoiRect(10, 10, 30, 30))
    moved = []
    viewport.draft_moved.connect(lambda key, rect: moved.append((key, rect)))

    viewport.mousePressEvent(_press_event((20, 20)))
    viewport.mouseMoveEvent(_move_event((35, 25)))

    assert moved == [("", viewport._current_rect)]


def test_dragging_outside_the_draft_still_draws_a_new_one():
    viewport = _one_to_one_viewport()
    viewport.set_draft(RoiRect(10, 10, 30, 30))
    moved = []
    viewport.draft_moved.connect(lambda key, rect: moved.append((key, rect)))

    viewport.mousePressEvent(_press_event((60, 60)))
    viewport.mouseMoveEvent(_move_event((80, 80)))
    viewport.mouseReleaseEvent(_release_event((80, 80)))

    assert viewport._current_rect.x0 >= 60 and viewport._current_rect.y0 >= 60
    assert moved == []  # a new drawing is not a move


def test_moving_the_draft_is_clamped_to_the_image():
    viewport = _one_to_one_viewport()
    viewport.set_draft(RoiRect(10, 10, 30, 30))

    viewport.mousePressEvent(_press_event((20, 20)))
    viewport.mouseMoveEvent(_move_event((99, 99)))

    assert viewport._current_rect == RoiRect(80, 80, 100, 100)


def test_after_committing_a_draft_pressing_inside_grabs_the_committed_roi():
    # After Set ROI1 the dashed draft is gone - pressing where it was grabs
    # the now-committed ROI1 (movable until confirmed), not an invisible draft.
    viewport = _one_to_one_viewport()
    viewport.set_draft(RoiRect(10, 10, 30, 30))
    viewport.commit_current_as("roi1")

    viewport.mousePressEvent(_press_event((20, 20)))

    assert viewport._move_key == "roi1"
    assert viewport._drag_start is None


def test_set_roi_shows_committed_rois_for_both_slots():
    viewport = _one_to_one_viewport()

    viewport.set_roi("roi1", RoiRect(10, 10, 30, 30))
    viewport.set_roi("roi2", RoiRect(50, 50, 70, 70))

    assert viewport.roi("roi1") == RoiRect(10, 10, 30, 30)
    assert viewport.roi("roi2") == RoiRect(50, 50, 70, 70)
    assert set(viewport._roi_items) == {"roi1", "roi2"}


def test_clear_rois_removes_them():
    viewport = _one_to_one_viewport()
    viewport.set_roi("roi1", RoiRect(10, 10, 30, 30))

    viewport.clear_rois()

    assert viewport.roi("roi1") is None
    assert viewport._roi_items == {}


def test_dragging_a_committed_roi_moves_it_and_emits_its_key():
    viewport = _one_to_one_viewport()
    viewport.set_roi("roi2", RoiRect(50, 50, 70, 70))
    moved = []
    viewport.draft_moved.connect(lambda key, rect: moved.append((key, rect)))

    viewport.mousePressEvent(_press_event((60, 60)))
    viewport.mouseMoveEvent(_move_event((70, 60)))
    viewport.mouseReleaseEvent(_release_event((70, 60)))

    rect = viewport.roi("roi2")
    assert (rect.width, rect.height) == (20, 20)
    assert abs(rect.x0 - 60) <= 1 and abs(rect.y0 - 50) <= 1
    assert moved and moved[-1] == ("roi2", rect)


def test_locked_viewport_neither_moves_rois_nor_draws():
    viewport = _one_to_one_viewport()
    viewport.set_roi("roi1", RoiRect(10, 10, 30, 30))
    viewport.set_locked(True)
    drafts = []
    viewport.draft_changed.connect(drafts.append)

    viewport.mousePressEvent(_press_event((20, 20)))  # inside ROI1
    viewport.mouseMoveEvent(_move_event((40, 40)))
    viewport.mousePressEvent(_press_event((60, 60)))  # empty area
    viewport.mouseMoveEvent(_move_event((80, 80)))

    assert viewport.roi("roi1") == RoiRect(10, 10, 30, 30)
    assert drafts == []


def test_unlocking_makes_rois_movable_again():
    viewport = _one_to_one_viewport()
    viewport.set_roi("roi1", RoiRect(10, 10, 30, 30))
    viewport.set_locked(True)
    viewport.set_locked(False)

    viewport.mousePressEvent(_press_event((20, 20)))
    viewport.mouseMoveEvent(_move_event((30, 20)))

    assert viewport.roi("roi1") != RoiRect(10, 10, 30, 30)


def test_new_image_clears_committed_rois():
    viewport = _one_to_one_viewport()
    viewport.set_roi("roi1", RoiRect(10, 10, 30, 30))

    viewport.set_image(np.zeros((100, 100), dtype=np.float64))

    assert viewport.roi("roi1") is None


def _key_event(key):
    from PySide6.QtGui import QKeyEvent

    return QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


def test_falx_and_septum_pairs_are_drawn_side_by_side_with_labels():
    from PySide6.QtWidgets import QGraphicsSimpleTextItem

    viewport = _one_to_one_viewport()
    for key, rect in (
        ("falx.roi1", RoiRect(10, 10, 30, 30)),
        ("falx.roi2", RoiRect(40, 10, 60, 30)),
        ("septum.roi1", RoiRect(10, 50, 30, 70)),
        ("septum.roi2", RoiRect(40, 50, 60, 70)),
    ):
        viewport.set_roi(key, rect)

    labels = sorted(
        item.text() for item in viewport.scene().items() if isinstance(item, QGraphicsSimpleTextItem)
    )
    assert labels == ["F1", "F2", "S1", "S2"]
    assert viewport.roi("septum.roi2") == RoiRect(40, 50, 60, 70)


def test_clicking_an_roi_selects_it_and_delete_asks_to_remove_it():
    viewport = _one_to_one_viewport()
    viewport.set_roi("septum.roi1", RoiRect(50, 50, 70, 70))
    selected, deleted = [], []
    viewport.roi_selected.connect(selected.append)
    viewport.roi_delete_requested.connect(deleted.append)

    viewport.mousePressEvent(_press_event((60, 60)))
    viewport.mouseReleaseEvent(_release_event((60, 60)))
    viewport.keyPressEvent(_key_event(Qt.Key.Key_Delete))

    assert selected == ["septum.roi1"]
    assert viewport.selected_roi() == "septum.roi1"
    assert deleted == ["septum.roi1"]


def test_clicking_empty_image_clears_the_selection():
    viewport = _one_to_one_viewport()
    viewport.set_roi("falx.roi1", RoiRect(50, 50, 70, 70))
    viewport.select_roi("falx.roi1")

    viewport.mousePressEvent(_press_event((5, 5)))
    viewport.mouseReleaseEvent(_release_event((5, 5)))

    assert viewport.selected_roi() is None


def test_delete_does_nothing_without_selection_or_while_locked():
    viewport = _one_to_one_viewport()
    viewport.set_roi("falx.roi1", RoiRect(50, 50, 70, 70))
    deleted = []
    viewport.roi_delete_requested.connect(deleted.append)

    viewport.keyPressEvent(_key_event(Qt.Key.Key_Delete))  # nothing selected
    viewport.select_roi("falx.roi1")
    viewport.set_locked(True)
    viewport.keyPressEvent(_key_event(Qt.Key.Key_Delete))  # confirmed -> locked

    assert deleted == []


def test_selection_is_cleared_when_another_slice_is_shown():
    viewport = _one_to_one_viewport()
    viewport.set_roi("falx.roi1", RoiRect(50, 50, 70, 70))
    viewport.select_roi("falx.roi1")

    viewport.set_image(np.zeros((100, 100)))

    assert viewport.selected_roi() is None


def test_selecting_an_roi_that_is_not_shown_selects_nothing():
    viewport = _one_to_one_viewport()
    viewport.select_roi("septum.roi2")
    assert viewport.selected_roi() is None


def _context_event(pos):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent

    return QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(*pos), QPoint(*pos))


def test_right_click_on_an_roi_or_the_drawn_rectangle_asks_for_its_menu():
    viewport = _one_to_one_viewport()
    viewport.set_roi("falx.roi1", RoiRect(50, 50, 70, 70))
    viewport.set_draft(RoiRect(100, 100, 120, 120))
    requested = []
    viewport.context_menu_requested.connect(lambda key, pos: requested.append(key))

    viewport.contextMenuEvent(_context_event((60, 60)))
    viewport.contextMenuEvent(_context_event((110, 110)))
    viewport.contextMenuEvent(_context_event((5, 5)))  # empty image: no menu

    assert requested == ["falx.roi1", ""]


def test_delete_key_without_a_selected_roi_deletes_the_drawn_rectangle():
    viewport = _one_to_one_viewport()
    viewport.set_draft(RoiRect(100, 100, 120, 120))
    deleted = []
    viewport.roi_delete_requested.connect(deleted.append)

    viewport.keyPressEvent(_key_event(Qt.Key.Key_Delete))

    assert deleted == [""]
