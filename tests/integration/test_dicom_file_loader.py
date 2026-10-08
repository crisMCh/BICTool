"""Tests use a synthetic, hand-built DICOM dataset - never real patient data."""

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from bictool.integration.dicom_file_loader import (
    _parse_decimal,
    discover_series,
    load_dicom_slice,
    load_dicom_volume,
    load_series,
)


def _make_synthetic_dicom(
    directory,
    *,
    filename="synthetic_slice.dcm",
    rescale_slope=1.0,
    rescale_intercept=-1024.0,
    pixel_spacing=("0.5", "0.5"),
    instance_number=None,
    fill_value=None,
    series_instance_uid=None,
    study_instance_uid=None,
    patient_id="SYNTH-0001",
    modality=None,
    patient_name=None,
    patient_sex=None,
    study_description=None,
    series_description=None,
    series_number=None,
    institution_name=None,
    manufacturer=None,
    manufacturer_model_name=None,
    acquisition_time=None,
    kvp=None,
    xray_tube_current=None,
    image_position_patient=None,
    slice_thickness=None,
    image_orientation_patient=None,
):
    rows, cols = 8, 8
    if fill_value is None:
        pixels = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
    else:
        pixels = np.full((rows, cols), fill_value, dtype=np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.PixelSpacing = list(pixel_spacing)
    ds.PatientID = patient_id
    ds.StudyID = "SYNTH-STUDY"
    ds.AcquisitionDate = "20260101"
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    if series_instance_uid is not None:
        ds.SeriesInstanceUID = series_instance_uid
    if study_instance_uid is not None:
        ds.StudyInstanceUID = study_instance_uid
    if modality is not None:
        ds.Modality = modality
    if patient_name is not None:
        ds.PatientName = patient_name
    if patient_sex is not None:
        ds.PatientSex = patient_sex
    if study_description is not None:
        ds.StudyDescription = study_description
    if series_description is not None:
        ds.SeriesDescription = series_description
    if series_number is not None:
        ds.SeriesNumber = series_number
    if institution_name is not None:
        ds.InstitutionName = institution_name
    if manufacturer is not None:
        ds.Manufacturer = manufacturer
    if manufacturer_model_name is not None:
        ds.ManufacturerModelName = manufacturer_model_name
    if acquisition_time is not None:
        ds.AcquisitionTime = acquisition_time
    if kvp is not None:
        ds.KVP = kvp
    if xray_tube_current is not None:
        ds.XRayTubeCurrent = xray_tube_current
    if image_position_patient is not None:
        ds.ImagePositionPatient = list(image_position_patient)
    if slice_thickness is not None:
        ds.SliceThickness = slice_thickness
    if image_orientation_patient is not None:
        ds.ImageOrientationPatient = list(image_orientation_patient)

    ds.save_as(str(path), enforce_file_format=True)
    return path, pixels


def test_load_dicom_slice_converts_to_hounsfield_units(tmp_path):
    path, raw_pixels = _make_synthetic_dicom(tmp_path, rescale_slope=1.0, rescale_intercept=-1024.0)

    result = load_dicom_slice(path)

    expected_hu = raw_pixels.astype(np.float64) * 1.0 + (-1024.0)
    np.testing.assert_array_equal(result.pixels_hu, expected_hu)


def test_load_dicom_slice_reads_pixel_spacing_and_header_fields(tmp_path):
    path, _ = _make_synthetic_dicom(tmp_path, pixel_spacing=("0.488", "0.488"))

    result = load_dicom_slice(path)

    assert result.pixel_spacing_mm == pytest.approx(0.488)
    assert result.patient_id == "SYNTH-0001"
    assert result.study_id == "SYNTH-STUDY"
    assert result.acquisition_date == "20260101"


def test_load_dicom_slice_defaults_missing_optional_fields_to_none(tmp_path):
    path, _ = _make_synthetic_dicom(tmp_path)

    result = load_dicom_slice(path)

    assert result.ctdi_vol_mgy is None
    assert result.dlp_mgy_cm is None
    assert result.reconstruction is None
    assert result.patient_name is None
    assert result.kvp is None
    assert result.slice_location_mm is None


def test_load_dicom_slice_reads_the_corner_overlay_metadata_fields(tmp_path):
    # These are the fields the main window's corner overlay shows (see
    # `main_window._update_viewer_overlay`'s docstring) - covers the
    # research-driven extraction added alongside it: identity, institution/
    # scanner, exposure technique, transfer syntax, and slice position.
    path, _ = _make_synthetic_dicom(
        tmp_path,
        modality="CT",
        patient_name="Doe^Jane",
        patient_sex="F",
        study_description="CT Head Spiral",
        series_description="Head 0.5 Vol",
        series_number="4",
        institution_name="Test Hospital",
        manufacturer="TestCorp",
        manufacturer_model_name="Scanner X",
        acquisition_time="104500.123",
        kvp=120.0,
        xray_tube_current=170,
        image_position_patient=(0.0, 0.0, -459.34),
    )

    result = load_dicom_slice(path)

    assert result.patient_name == "Doe Jane"
    assert result.patient_sex == "F"
    assert result.study_description == "CT Head Spiral"
    assert result.series_description == "Head 0.5 Vol"
    assert result.series_number == "4"
    assert result.institution_name == "Test Hospital"
    assert result.manufacturer_model == "TestCorp Scanner X"
    assert result.acquisition_time == "10:45:00"
    assert result.kvp == pytest.approx(120.0)
    assert result.tube_current_ma == pytest.approx(170.0)
    assert result.modality == "CT"
    assert result.transfer_syntax_name  # exact text depends on pydicom's UID dictionary - just non-empty
    assert result.slice_location_mm == pytest.approx(-459.34)


def test_load_dicom_volume_from_single_file_returns_one_slice_volume(tmp_path):
    path, raw_pixels = _make_synthetic_dicom(tmp_path, rescale_slope=1.0, rescale_intercept=-1024.0)

    volume = load_dicom_volume(path)

    assert volume.num_slices == 1
    expected_hu = raw_pixels.astype(np.float64) * 1.0 + (-1024.0)
    np.testing.assert_array_equal(volume.get_slice_hu(0), expected_hu)
    assert volume.patient_id == "SYNTH-0001"


def test_load_dicom_volume_from_directory_stacks_slices_in_instance_order(tmp_path):
    # Save files out of instance order to verify sorting is by InstanceNumber,
    # not filename/directory-listing order.
    _make_synthetic_dicom(tmp_path, filename="c.dcm", instance_number=3, fill_value=30)
    _make_synthetic_dicom(tmp_path, filename="a.dcm", instance_number=1, fill_value=10)
    _make_synthetic_dicom(tmp_path, filename="b.dcm", instance_number=2, fill_value=20)

    volume = load_dicom_volume(tmp_path)

    assert volume.num_slices == 3
    expected_fill_values = [10.0 - 1024.0, 20.0 - 1024.0, 30.0 - 1024.0]
    for i, expected_fill in enumerate(expected_fill_values):
        assert np.all(volume.get_slice_hu(i) == expected_fill)


def test_load_dicom_volume_carries_each_slices_own_location(tmp_path):
    # Slice location genuinely varies per slice (unlike the rest of this
    # project's metadata, taken from just the first slice) - the overlay
    # (`main_window._update_viewer_overlay`) needs the value for whichever
    # slice is actually on screen, indexed the same as `get_slice_hu`.
    _make_synthetic_dicom(tmp_path, filename="a.dcm", instance_number=1, image_position_patient=(0, 0, -10.0))
    _make_synthetic_dicom(tmp_path, filename="b.dcm", instance_number=2, image_position_patient=(0, 0, -8.0))
    _make_synthetic_dicom(tmp_path, filename="c.dcm", instance_number=3, image_position_patient=(0, 0, -6.0))

    volume = load_dicom_volume(tmp_path)

    assert volume.slice_locations_mm == pytest.approx([-10.0, -8.0, -6.0])


def test_load_dicom_volume_finds_series_nested_in_subfolders(tmp_path):
    # Real DICOM exports are commonly nested (patient/study/series/*.dcm) -
    # the user should be able to point at the top-level folder without
    # navigating down to the exact subfolder holding the images.
    _make_synthetic_dicom(tmp_path, filename="Patient1/Study1/Series1/a.dcm", instance_number=1)
    _make_synthetic_dicom(tmp_path, filename="Patient1/Study1/Series1/b.dcm", instance_number=2)
    _make_synthetic_dicom(tmp_path, filename="Patient1/Study1/Series1/c.dcm", instance_number=3)

    volume = load_dicom_volume(tmp_path)

    assert volume.num_slices == 3
    assert len(discover_series(tmp_path)) == 1


def test_load_dicom_volume_skips_non_dicom_clutter(tmp_path):
    _make_synthetic_dicom(tmp_path, filename="a.dcm", instance_number=1)
    _make_synthetic_dicom(tmp_path, filename="b.dcm", instance_number=2)
    (tmp_path / "DICOMDIR").write_bytes(b"not a real DICOMDIR")
    (tmp_path / "readme.txt").write_text("not DICOM at all")

    volume = load_dicom_volume(tmp_path)

    assert volume.num_slices == 2


def test_load_dicom_volume_skips_ihe_pdi_style_jpeg_previews_without_opening_them(tmp_path, monkeypatch):
    # Reproduces a real "burn to CD" export layout (IHE PDI profile): a
    # duplicate JPEG rendering of every DICOM image, sitting right alongside
    # it - these must never even be opened, let alone force-parsed as DICOM.
    _make_synthetic_dicom(tmp_path, filename="IMAGE/DCM/I0000000", instance_number=1)
    _make_synthetic_dicom(tmp_path, filename="IMAGE/DCM/I0000001", instance_number=2)
    for i in range(2):
        (tmp_path / "IHE_PDI" / "JPG").mkdir(parents=True, exist_ok=True)
        (tmp_path / "IHE_PDI" / "JPG" / f"I000000{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    (tmp_path / "burn.xml").write_text("<manifest/>")

    opened_paths = []
    import bictool.integration.dicom_file_loader as loader_module

    original_try_read_header = loader_module._try_read_header

    def _tracking_try_read_header(path):
        opened_paths.append(path)
        return original_try_read_header(path)

    monkeypatch.setattr(loader_module, "_try_read_header", _tracking_try_read_header)

    volume = load_dicom_volume(tmp_path)

    assert volume.num_slices == 2
    assert all(f.suffix.lower() != ".jpg" for f in opened_paths)
    assert all(f.name != "burn.xml" for f in opened_paths)


def test_load_dicom_volume_picks_largest_ct_series_when_multiple_present(tmp_path):
    # A localizer/scout series (single image, non-CT) alongside the real
    # axial CT series - the CT series should win regardless of file order.
    _make_synthetic_dicom(
        tmp_path, filename="scout.dcm", series_instance_uid=generate_uid(), modality="OT", instance_number=1
    )
    ct_series_uid = generate_uid()
    for i in range(1, 4):
        _make_synthetic_dicom(
            tmp_path,
            filename=f"ct_{i}.dcm",
            series_instance_uid=ct_series_uid,
            modality="CT",
            instance_number=i,
        )

    volume = load_dicom_volume(tmp_path)

    assert volume.num_slices == 3
    candidates = discover_series(tmp_path)
    assert len(candidates) == 2
    assert candidates[0].label.startswith("CT")  # best-first: CT preferred over the scout


def test_load_dicom_volume_prefers_axial_4mm_series_as_the_default(tmp_path):
    # The routine head-CT reading series (per WP2's target protocol) is
    # axial, 4mm slices - when a study also has a thin axial series (e.g. a
    # bone-algorithm recon) and a sagittal reformat, the 4mm axial series
    # should be the one auto-selected, not just whichever has the most
    # slices or came first alphabetically.
    axial_orientation = (1, 0, 0, 0, 1, 0)  # row=X, col=Y -> normal=Z (axial)
    sagittal_orientation = (0, 1, 0, 0, 0, -1)  # row=Y, col=-Z -> normal=X (sagittal)

    sagittal_uid = generate_uid()
    for i in range(1, 4):
        _make_synthetic_dicom(
            tmp_path, filename=f"sag_{i}.dcm", series_instance_uid=sagittal_uid, modality="CT",
            instance_number=i, slice_thickness=1.0, image_orientation_patient=sagittal_orientation,
        )
    thin_axial_uid = generate_uid()
    for i in range(1, 6):
        _make_synthetic_dicom(
            tmp_path, filename=f"thin_{i}.dcm", series_instance_uid=thin_axial_uid, modality="CT",
            instance_number=i, slice_thickness=1.0, image_orientation_patient=axial_orientation,
        )
    target_axial_uid = generate_uid()
    for i in range(1, 3):
        _make_synthetic_dicom(
            tmp_path, filename=f"axial4_{i}.dcm", series_instance_uid=target_axial_uid, modality="CT",
            instance_number=i, slice_thickness=4.0, image_orientation_patient=axial_orientation,
        )

    candidates = discover_series(tmp_path)

    assert candidates[0].label.endswith("axial")
    assert "4mm" in candidates[0].label
    volume = load_dicom_volume(tmp_path)
    assert volume.slice_thickness_mm == pytest.approx(4.0)
    assert volume.num_slices == 2


def test_load_dicom_volume_splits_mismatched_dimensions_into_separate_candidates_instead_of_erroring(tmp_path):
    # Real-world cause: a stray localizer/secondary-capture image sharing
    # the main series' SeriesInstanceUID but a different image matrix. Real
    # DICOM toolkits (GDCM/SimpleITK's `SetUseSeriesDetails`, dcm2niix) treat
    # this as "more than one stack under one nominal series," not an error -
    # loading must never fail just because a folder's files aren't all the
    # same shape.
    shared_series_uid = generate_uid()
    _make_synthetic_dicom(
        tmp_path, filename="a.dcm", instance_number=1, series_instance_uid=shared_series_uid
    )

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    path = tmp_path / "b.dcm"
    pixels = np.zeros((4, 4), dtype=np.uint16)
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Rows, ds.Columns = 4, 4
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds.InstanceNumber = 2
    ds.SeriesInstanceUID = shared_series_uid  # same series UID, different shape
    ds.save_as(str(path), enforce_file_format=True)

    candidates = discover_series(tmp_path)

    assert len(candidates) == 2
    assert {c.num_slices for c in candidates} == {1, 1}
    # Each candidate is internally consistent by construction - loading
    # either one must succeed without raising.
    for candidate in candidates:
        volume = load_series(candidate)
        assert volume.num_slices == 1

    # load_dicom_volume (auto-select) must also just work, not raise.
    volume = load_dicom_volume(tmp_path)
    assert volume.num_slices == 1


def test_load_dicom_volume_rejects_folder_spanning_multiple_studies(tmp_path):
    # Reproduces a real bug: a folder holding several unrelated patients'
    # single reference files (each its own study) must not be silently
    # narrowed down to "the largest series" - that discards everyone else
    # without any indication to the user.
    _make_synthetic_dicom(
        tmp_path,
        filename="patient_a.dcm",
        patient_id="PATIENT-A",
        study_instance_uid=generate_uid(),
        instance_number=1,
    )
    _make_synthetic_dicom(
        tmp_path,
        filename="patient_b.dcm",
        patient_id="PATIENT-B",
        study_instance_uid=generate_uid(),
        instance_number=1,
    )

    with pytest.raises(ValueError, match="different studies"):
        load_dicom_volume(tmp_path)


def test_load_dicom_volume_reports_scan_progress(tmp_path):
    for i in range(1, 31):
        _make_synthetic_dicom(tmp_path, filename=f"slice_{i:02d}.dcm", instance_number=i)

    progress_calls = []
    load_dicom_volume(tmp_path, on_progress=progress_calls.append)

    assert progress_calls == [25]


# ---- comma decimal separator (non-conformant DS values) ----
# Confirmed 2026-09-24 on a real Siemens NAEOTOM Alpha export: some DS
# (decimal string) tag values were written with a comma instead of the
# DICOM-standard-required period (e.g. ImagePositionPatient's z coordinate
# as the literal text "-748,312") - evidently a European-locale bug in the
# scanner's own export tool. Reproduced here by writing a normal, valid
# DICOM file and then patching its raw bytes, since pydicom's own Python-
# level attribute assignment (`ds.ImagePositionPatient = [...]`) always
# validates via a real `float()` call and can't be used to construct this
# malformed value directly - only reading an already-non-conformant file
# (as a real scanner can produce) hits the code path this fix is for.


def _corrupt_decimal_separator(path, decimal_text: str) -> None:
    """Rewrites `path`'s raw bytes, replacing the period in `decimal_text`
    (e.g. "-748.312", as written by a normal `_make_synthetic_dicom` call)
    with a comma - same byte length, so no DICOM length field needs fixing.
    """
    corrupted_text = decimal_text.replace(".", ",")
    raw = path.read_bytes()
    assert decimal_text.encode() in raw, f"{decimal_text!r} not found in {path} - adjust the test's input value"
    path.write_bytes(raw.replace(decimal_text.encode(), corrupted_text.encode()))


class TestCommaDecimalSeparator:
    def test_parse_decimal_accepts_a_comma_separator(self):
        assert _parse_decimal("-748,312") == pytest.approx(-748.312)

    def test_parse_decimal_still_accepts_a_normal_period(self):
        assert _parse_decimal("-748.312") == pytest.approx(-748.312)

    def test_parse_decimal_leaves_a_real_thousands_style_value_alone(self):
        # Not a realistic DICOM DS value, but confirms the comma-only-when-
        # no-period heuristic doesn't mangle a string that already has one.
        assert _parse_decimal("1.234") == pytest.approx(1.234)

    def test_a_comma_separated_image_position_patient_does_not_crash_volume_loading(self, tmp_path):
        path, _ = _make_synthetic_dicom(
            tmp_path, filename="a.dcm", instance_number=1, image_position_patient=(0, 0, -748.312)
        )
        _corrupt_decimal_separator(path, "-748.312")

        volume = load_dicom_volume(tmp_path)  # must not raise

        assert volume.slice_locations_mm == pytest.approx([-748.312])

    def test_a_comma_separated_image_position_patient_does_not_crash_series_discovery(self, tmp_path):
        # `discover_series` sorts every candidate's slices by
        # ImagePositionPatient during discovery itself (`_slice_sort_key`),
        # before any single volume is chosen/loaded - this is the exact
        # code path that actually crashed on the real scan (`_build_
        # candidate` -> `_slice_sort_key` -> `float(position[2])`).
        path_a, _ = _make_synthetic_dicom(
            tmp_path, filename="a.dcm", instance_number=1, image_position_patient=(0, 0, -748.312)
        )
        _corrupt_decimal_separator(path_a, "-748.312")
        _make_synthetic_dicom(tmp_path, filename="b.dcm", instance_number=2, image_position_patient=(0, 0, -750.0))

        candidates = discover_series(tmp_path)  # must not raise

        assert candidates[0].num_slices == 2


def test_reformats_mislabelled_as_axial_are_recognised_by_their_slice_positions(tmp_path):
    # Real Siemens photon-counting export: the coronal MPR ("kor") carries the
    # source volume's axial ImageOrientationPatient, but its slices step
    # along Y. It has more slices than the real axial series and used to win.
    axial_orientation = (1, 0, 0, 0, 1, 0)
    coronal_uid = generate_uid()
    for i in range(6):
        _make_synthetic_dicom(
            tmp_path, filename=f"kor_{i}.dcm", series_instance_uid=coronal_uid, modality="CT",
            instance_number=i + 1, slice_thickness=4.0, image_orientation_patient=axial_orientation,
            image_position_patient=(-70.0, -350.0 + 4 * i, -594.5), series_description="CCT 4,00 kor",
        )
    axial_uid = generate_uid()
    for i in range(4):
        _make_synthetic_dicom(
            tmp_path, filename=f"ax_{i}.dcm", series_instance_uid=axial_uid, modality="CT",
            instance_number=i + 1, slice_thickness=4.0, image_orientation_patient=axial_orientation,
            image_position_patient=(-75.0, -345.0, -746.0 + 4 * i), series_description="CCT 4,00 ax",
        )

    candidates = discover_series(tmp_path)

    by_description = {c._reference_header.SeriesDescription: c for c in candidates}
    assert by_description["CCT 4,00 kor"].axial is False
    assert by_description["CCT 4,00 ax"].axial is True
    assert load_dicom_volume(tmp_path).num_slices == 4  # the real axial series


def test_a_strongly_tilted_axial_series_still_counts_as_axial(tmp_path):
    # Real Siemens ER series: gantry tilt + angulation put the slice normal
    # ~47 deg off the Z axis (Y component slightly larger than Z).
    normal = np.array([0.078, 0.73, 0.679])
    normal /= np.linalg.norm(normal)
    row = np.array([0.99, -0.1, 0.0])
    row -= row.dot(normal) * normal
    row /= np.linalg.norm(row)
    col = np.cross(normal, row)
    for i in range(4):
        _make_synthetic_dicom(
            tmp_path, filename=f"tilt_{i}.dcm", series_instance_uid="1.2.3.4", modality="CT",
            instance_number=i + 1, slice_thickness=4.0,
            image_orientation_patient=tuple(float(v) for v in (*row, *col)),
            image_position_patient=tuple(float(v) for v in 4 * i * normal),
        )

    (candidate,) = discover_series(tmp_path)

    assert candidate.axial is True
    assert candidate.label.endswith("axial")
