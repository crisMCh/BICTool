from bictool.presentation.views.export_dialog import ExportDialog


def test_accept_disabled_when_nothing_checked():
    dialog = ExportDialog()
    dialog._excel_cb.setChecked(False)

    assert dialog._accept_btn.isEnabled() is False


def test_accept_enabled_with_a_format_and_leads_to_the_save_dialog():
    dialog = ExportDialog()
    # Excel is checked by default; the file name and folder are chosen next.

    assert dialog._accept_btn.isEnabled() is True
    assert dialog._accept_btn.text() == "Save as…"


def test_selected_items_reflects_checked_boxes():
    dialog = ExportDialog()
    dialog._excel_cb.setChecked(False)
    dialog._csv_cb.setChecked(True)
    dialog._jpg_cb.setChecked(True)

    assert dialog.selected_items() == {"csv", "jpg"}


def test_scope_defaults_to_the_current_slice_and_study_needs_results():
    dialog = ExportDialog()

    assert dialog.scope() == "slice"
    assert dialog._study_rb.isEnabled() is False


def test_study_scope_can_be_preselected_and_disables_chart_images():
    dialog = ExportDialog(study_available=True, default_scope="study")

    assert dialog.scope() == "study"
    assert dialog._jpg_cb.isEnabled() is False

    dialog._slice_rb.setChecked(True)
    assert dialog._jpg_cb.isEnabled() is True
