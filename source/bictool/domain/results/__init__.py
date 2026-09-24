"""Results sub-module of the Domain Layer.

Aggregates per-slice/per-structure metrics into quality scores and
study-level ratings (FR-04) for the Application Layer to consume via
the Core API. `classification` is a translation of the SVM classification
tail of `reference/octave/funcImageQualityOctave.m`; the trained model
(`_svm_model.npz`) is a frozen, portable extraction of
`reference/octave/svmModelOctave.mat` + `OctaveFeatSet.mat`. `study_quality`
wires the individual metrics (MTF/NPS/contrast/PS-MTF-ratio) together and
assembles the 7-element feature vector `classification` consumes - it
captures that computational recipe from `funcImageQualityOctave.m` without
translating the script itself (no DICOM I/O, ROI cropping, plotting, or
file export - see its module docstring). Exposed to the Application Layer
via `bictool.domain.core_api`.
"""
