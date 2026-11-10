"""Synthetic helpers for the API tests - hand-built label maps, images and
DICOM files, a stand-in for the CNN. No patient data."""

from types import SimpleNamespace

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from bictool.domain.results.study_quality import SliceQualityResult
from bictool.domain.roi_selection.structured_roi import SEPTUM_PELLUCIDUM

SHAPE = (160, 160)
PIXEL_SPACING = 0.5


def septum_slice() -> dict[str, np.ndarray]:
    ventricle = np.zeros(SHAPE, dtype=bool)
    ventricle[40:120, 40:120] = True
    septum = np.zeros(SHAPE, dtype=bool)
    septum[45:115, 79:81] = True
    ventricle &= ~septum
    parenchyma = np.zeros(SHAPE, dtype=bool)
    parenchyma[10:150, 10:150] = True
    parenchyma &= ~(ventricle | septum)
    return {"Brain Parenchyma": parenchyma, "Ventricular System": ventricle, SEPTUM_PELLUCIDUM: septum}


def image(masks) -> np.ndarray:
    out = np.full(SHAPE, 30.0)
    out[masks["Ventricular System"]] = 5.0
    out[masks[SEPTUM_PELLUCIDUM]] = 35.0
    return out


def fake_volume(masks_by_slice, num_slices=3):
    return SimpleNamespace(
        num_slices=num_slices,
        pixel_spacing_mm=PIXEL_SPACING,
        slice_thickness_mm=4.0,
        series_description="Head 4.0 Hr40",
        acquisition_date="20000101",
        slice_paths=[f"IM{i:04d}" for i in range(num_slices)],
        slice_locations_mm=[4.0 * i for i in range(num_slices)],
        sop_instance_uids=[f"1.2.3.{i}" for i in range(num_slices)],
        series_instance_uid="1.2.3",
        get_slice_hu=lambda i: image(masks_by_slice[i]) if i in masks_by_slice else np.zeros(SHAPE),
    )


def fake_quality(**overrides) -> SliceQualityResult:
    nps = SimpleNamespace(f_avg=np.linspace(0, 1, 5), nps_measured_radagv=np.ones(5))
    values = dict(
        mtf_xaxis=np.linspace(0, 1, 4), mtf=np.ones((4, 2)), mtf_area_per_edge=np.ones(2), mtf_area=0.5,
        nps_roi1=nps, nps_roi2=nps, contrast_roi1=0.1, contrast_roi2=0.02,
        ratio=SimpleNamespace(f_int=np.linspace(0, 1, 3), ratio_nps_mtf=np.ones(3)),
        feature_vector=np.array([0.5, 2.0, 3.0, 4.0, 1.5, 0.0, 0.0]), predicted_label=1, decision_value=0.3,
        recipe="physical", nps_f_av=0.2, nps_f_peak=float("nan"),
    )
    values.update(overrides)
    return SliceQualityResult(**values)


def write_synthetic_study(folder, num_slices=3):
    """A CT series of `num_slices` synthetic 160x160 axial slices (4 mm apart)."""
    study_uid, series_uid = generate_uid(), generate_uid()
    pixels = (image(septum_slice()) + 1024).astype(np.uint16)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(num_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = folder / f"IM{i:04d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.Modality, ds.SeriesDescription, ds.PatientID = "CT", "Head 4.0 Hr40", "SYNTH-0001"
        ds.Rows, ds.Columns = SHAPE
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = pixels.tobytes()
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelSpacing = [str(PIXEL_SPACING), str(PIXEL_SPACING)]
        ds.SliceThickness = "4.0"
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0, 0, 4.0 * i]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceLocation = 4.0 * i
        ds.save_as(str(path), enforce_file_format=True)
    return folder
