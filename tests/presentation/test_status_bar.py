"""Tests the bottom bar: state follows the text the workflow writes."""

from bictool.presentation.views.status_bar import StatusPanel


def test_idle_tasks_show_their_last_state_and_the_study_message_stays():
    panel = StatusPanel()
    segmentation = panel.add_item("Segmentation:", busy=r"^Segmenting")
    panel.message.label.setText("Loaded: study (44 slice(s))")

    segmentation.label.setText("—")
    assert segmentation.isHidden()
    segmentation.label.setText("Done")
    assert not segmentation.isHidden() and not segmentation.is_busy()
    assert not panel.message.isHidden() and panel.progress_bar().isHidden()


def test_a_running_task_replaces_the_study_message_with_a_full_width_bar():
    panel = StatusPanel()
    segmentation = panel.add_item("Segmentation:", busy=r"^Segmenting")
    panel.message.label.setText("Loaded: study (44 slice(s))")

    segmentation.label.setText("Segmenting volume… 12/44")

    bar = panel.progress_bar()
    assert not bar.isHidden() and panel.message.isHidden() and segmentation.isHidden()
    assert (bar.minimum(), bar.maximum(), bar.value()) == (0, 44, 12)
    assert panel._task_label.text() == "Segmentation: Segmenting volume… 12/44"

    segmentation.label.setText("Done (volume)")
    assert bar.isHidden() and not panel.message.isHidden() and not segmentation.isHidden()


def test_loading_without_counts_is_indeterminate_and_the_last_started_task_is_shown():
    panel = StatusPanel()
    qa = panel.add_item("", busy=r"^QA \d+/\d+")

    panel.message.label.setText("Loading DICOM data… (120 files scanned)")
    assert panel.progress_bar().maximum() == 0  # indeterminate animation
    qa.label.setText("QA 3/33")
    assert panel.active_item() is qa and panel.progress_bar().maximum() == 33

    qa.label.setText("QA done: 33 ROI pair(s)")
    assert panel.active_item() is panel.message  # loading still running


def test_cancel_button_sits_in_the_bar_and_calls_back():
    panel = StatusPanel()
    qa = panel.add_item("", busy=r"^QA")
    clicked = []
    button = qa.add_cancel(lambda: clicked.append(True))

    button.click()

    assert clicked == [True] and panel.isAncestorOf(button) and button.isHidden()
