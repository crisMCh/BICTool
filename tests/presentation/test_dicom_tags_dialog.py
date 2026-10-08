from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from bictool.presentation.views.dicom_tags_dialog import DicomTagsDialog


def _fake_dataset() -> Dataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = file_meta
    ds.PatientID = "SYNTH-0001"
    ds.Modality = "CT"
    ds.Rows = 512
    ds.Columns = 512
    ds.PixelData = b"\x00" * 100  # never dumped as text - see _format_value
    return ds


def test_populates_one_row_per_top_level_element():
    dialog = DicomTagsDialog(_fake_dataset(), title="DICOM Tags")

    assert dialog._table.rowCount() == 5  # PatientID, Modality, Rows, Columns, PixelData
    keywords = {dialog._table.item(row, 2).text() for row in range(dialog._table.rowCount())}
    assert keywords == {"PatientID", "Modality", "Rows", "Columns", "PixelData"}


def test_pixel_data_is_never_dumped_as_raw_bytes():
    dialog = DicomTagsDialog(_fake_dataset(), title="DICOM Tags")

    row = next(r for r in range(dialog._table.rowCount()) if dialog._table.item(r, 2).text() == "PixelData")
    value_text = dialog._table.item(row, 3).text()

    assert "byte(s)" in value_text
    assert "\\x00" not in value_text


def test_filter_hides_non_matching_rows():
    dialog = DicomTagsDialog(_fake_dataset(), title="DICOM Tags")

    dialog._filter_edit.setText("modality")

    hidden_keywords = set()
    visible_keywords = set()
    for row in range(dialog._table.rowCount()):
        keyword = dialog._table.item(row, 2).text()
        (hidden_keywords if dialog._table.isRowHidden(row) else visible_keywords).add(keyword)

    assert "Modality" in visible_keywords
    assert "PatientID" in hidden_keywords


def test_clearing_the_filter_shows_every_row_again():
    dialog = DicomTagsDialog(_fake_dataset(), title="DICOM Tags")
    dialog._filter_edit.setText("modality")

    dialog._filter_edit.setText("")

    assert all(not dialog._table.isRowHidden(row) for row in range(dialog._table.rowCount()))
