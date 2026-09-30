from bictool.presentation.views.export_dialog import ExportDialog


def test_accept_disabled_with_no_destination():
    dialog = ExportDialog()
    assert dialog._dest_edit.text() == ""
    assert dialog._accept_btn.isEnabled() is False


def test_accept_disabled_when_nothing_checked(tmp_path):
    dialog = ExportDialog(default_dir=str(tmp_path))
    dialog._excel_cb.setChecked(False)

    assert dialog._accept_btn.isEnabled() is False


def test_accept_enabled_once_a_format_and_destination_are_set(tmp_path):
    dialog = ExportDialog(default_dir=str(tmp_path))
    # Excel is checked by default (see ExportDialog docstring/default state).

    assert dialog._accept_btn.isEnabled() is True


def test_selected_items_reflects_checked_boxes(tmp_path):
    dialog = ExportDialog(default_dir=str(tmp_path))
    dialog._excel_cb.setChecked(False)
    dialog._csv_cb.setChecked(True)
    dialog._jpg_cb.setChecked(True)

    assert dialog.selected_items() == {"csv", "jpg"}


def test_destination_returns_the_typed_path(tmp_path):
    dialog = ExportDialog()
    dialog._dest_edit.setText(str(tmp_path) + " ")  # trailing whitespace, as a user might leave it

    assert dialog.destination() == tmp_path


def test_scope_defaults_to_the_current_slice_and_study_needs_results():
    dialog = ExportDialog()

    assert dialog.scope() == "slice"
    assert dialog._study_rb.isEnabled() is False


def test_study_scope_can_be_preselected_and_disables_chart_images(tmp_path):
    dialog = ExportDialog(default_dir=str(tmp_path), study_available=True, default_scope="study")

    assert dialog.scope() == "study"
    assert dialog._jpg_cb.isEnabled() is False

    dialog._slice_rb.setChecked(True)
    assert dialog._jpg_cb.isEnabled() is True
