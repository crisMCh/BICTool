# Reference: Result types

What a QA run returns. All types are frozen dataclasses of the API, independent of BICTool's internal classes.
Field names are also the keys in the [JSON format](json-format.md). Slice indices are 0-based; values that could
not be computed are NaN (JSON: `null`); a "can be empty" field is `None` (JSON: `null`) when not available.

| Type | What it is |
|---|---|
| [`StudyResult`](#studyresult) | Everything one QA run found for one series. |
| [`SeriesInfo`](#seriesinfo), [`SeriesChoice`](#serieschoice), [`SliceRef`](#sliceref) | Which series, which image. |
| [`Measurement`](#measurement), [`Curves`](#curves), [`Failure`](#failure) | One ROI pair on one slice. |
| [`Verdict`](#verdict), [`FeatureFinding`](#featurefinding), [`NoiseScatter`](#noisescatter), [`SeptumCheck`](#septumcheck) | The study verdict and its explanation. |
| [`ArtefactHints`](#artefacthints), [`BasalGangliaSlice`](#basalgangliaslice), [`SegmentationCheck`](#segmentationcheck) | The hints of the program's Study summary. |
| `PluginValues`, `PluginHints`, `PluginMarker` | What plug-ins reported - see [Plug-in types](plugin-types.md). |

## `StudyResult`

*Since 0.1; plug-in fields since 0.2 / 0.3*

| Field | Type | Description |
|---|---|---|
| `label` | `str` | The study's name: the caller's `label`, else `"study - <series description>"`. |
| `series` | [`SeriesInfo`](#seriesinfo) | The measured series. |
| `settings` | [`Settings`](settings.md#settings) | Recipe and WM/GM method used. In JSON: `{"recipe", "tissue_method", "tissue_parameters"}`. |
| `rois_reviewed` | `bool` | The caller said a person checked the ROIs (`measure(reviewed=True)`; always `True` from the program). |
| `rois_edited` | `bool` | Some ROI was moved, drawn or deleted. |
| `measurements` | tuple of [`Measurement`](#measurement) | One per measured ROI pair and slice, in slice order. |
| `failed` | tuple of [`Failure`](#failure) | Pairs that could not be measured, and pairs whose ROI2 was not found. |
| `skipped` | `dict[str, tuple[int, ...]]` | Per pair: slices where the structure was found but the brain is too small for automatic ROIs. |
| `segmented_slices` | `int` | Number of segmented slices. |
| `verdict` | [`Verdict`](#verdict), can be empty | The head-CT verdict. Empty for the legacy recipe, with fewer than 5 measured falx slices, or without a model. |
| `artefacts` | [`ArtefactHints`](#artefacthints), can be empty | Possible artefacts. |
| `basal_ganglia` | `dict[int, BasalGangliaSlice]` | Basal-ganglia visibility per slice. |
| `segmentation_check` | [`SegmentationCheck`](#segmentationcheck), can be empty | Study-level check of the segmentation. |
| `plugins` | `dict[str, str]` | The active plug-ins: name -> version. *Since 0.2.* |
| `plugin_values` | tuple of `PluginValues` | Each plug-in's values for the whole series (`bictool_study_metrics`). *Since 0.2.* |
| `slice_plugin_values` | `dict[int, tuple[PluginValues, ...]]` | Slice -> each plug-in's slice values (`bictool_slice_metrics`). *Since 0.3.* |
| `plugin_hints` | tuple of `PluginHints` | Possible artefacts plug-ins noticed (`bictool_artefact_hints`). *Since 0.3.* |
| `api_version` | `str` | The API version that produced the result. |
| `schema_version` | `str` | The result layout version (`api.SCHEMA_VERSION`). |

| Method | Returns |
|---|---|
| `measurements_for(pair)` | `{slice index: Measurement}` of `"falx"` or `"septum"`. |
| `plugin_markers(slice_index=None)` | Every `PluginMarker` the plug-ins returned (of one slice, if given). *Since 0.3.* |
| `to_dict(include_curves=False, include_dicom_uids=False)` | Plain, JSON-ready values; curves and DICOM UIDs left out unless asked for. |
| `to_json(include_curves=False, include_dicom_uids=False, indent=2)` | The same as a JSON string. |
| `StudyResult.from_dict(data)`, `StudyResult.from_json(text)` | A result back from `to_dict`/`to_json` (same schema major version, else `ValueError`). It has no internal objects: `to_volume_qa` is not available on it. |
| `to_volume_qa(group="")` | Internal: the container BICTool's Excel/CSV exports read. |

## `SeriesInfo`

*Since 0.1.* The measured series: acquisition parameters and DICOM references, no patient identifiers.

| Field | Type | Description |
|---|---|---|
| `description` | `str`, can be empty | DICOM Series Description. |
| `slices` | `int` | Number of slices. |
| `slice_thickness_mm` | `float`, can be empty | Slice thickness (mm). |
| `pixel_spacing_mm` | `float` | In-plane pixel size (mm). |
| `manufacturer_model` | `str`, can be empty | Scanner model. |
| `reconstruction` | `str`, can be empty | Reconstruction kernel. |
| `kvp` | `float`, can be empty | Tube voltage (kV). |
| `tube_current_ma` | `float`, can be empty | Tube current (mA). |
| `ctdi_vol_mgy` | `float`, can be empty | CTDIvol (mGy). |
| `dlp_mgy_cm` | `float`, can be empty | DLP (mGy·cm). |
| `acquisition_date` | `str`, can be empty | DICOM date (`YYYYMMDD`). |
| `study_instance_uid` | `str`, can be empty | DICOM Study Instance UID - in JSON only with `include_dicom_uids`. |
| `series_instance_uid` | `str`, can be empty | DICOM Series Instance UID - likewise. |
| `label` | `str` (property) | `"Head 4.0 Hr40 (37 slices, 4.0 mm)"` - the program's and exports' series label. |

## `SeriesChoice`

*Since 0.1.* One series of a study folder ([`list_series`](api-functions.md#list_series)).

| Field | Type | Description |
|---|---|---|
| `index` | `int` | What `open_study(series=...)` and `analyse_study(series=...)` take. |
| `label` | `str` | Modality, slices, matrix, kernel, thickness, "axial". |
| `slices` | `int` | Number of slices. |
| `axial` | `bool`, can be empty | Whether the slices are axial (empty if unknown). |

## `SliceRef`

*Since 0.1.* Which image.

| Field | Type | Description |
|---|---|---|
| `index` | `int` | 0-based index in the series (the program shows index + 1). |
| `location_mm` | `float`, can be empty | DICOM slice location (mm). |
| `sop_instance_uid` | `str`, can be empty | DICOM SOP Instance UID - in JSON only with `include_dicom_uids`. |

## `Measurement`

*Since 0.1; `plugin_values` since 0.2.* One ROI pair measured on one slice. The first five values are the verdict
model's inputs and keep the model's names.

| Field | Type | Unit | Description |
|---|---|---|---|
| `slice` | [`SliceRef`](#sliceref) | | The slice. |
| `pair` | `str` | | `"falx"` or `"septum"`. |
| `roi1`, `roi2` | [`RoiRect`](rois.md#roirect) | px | ROI1 on the structure, ROI2 in homogeneous tissue. JSON: `[x0, y0, x1, y1]`. |
| `roi1_manual`, `roi2_manual` | `bool` | | Whether each ROI was drawn or moved by hand. |
| `roi2_region` | `str`, can be empty | | `"ventricle"`, `"white_matter"` or `"fallback"` for an automatic ROI2; empty for a manual one. |
| `recipe` | `str` | | `"physical"` or `"octave"`. |
| `mtf_area` | `float` | 1/mm | Area under the MTF measured at the structure's edges. |
| `nps_homo` | `float` | HU²·mm² | Noise power (NPS variance) in ROI2. |
| `nps_struct` | `float` | HU²·mm² | Noise power in ROI1. |
| `var_struct` | `float` | HU² | Pixel variance in ROI1. |
| `ratio_lf_hf` | `float` | 1 | Low- to high-frequency ratio of PS/MTF. |
| `nps_f_av` | `float` | 1/mm | Average frequency of the NPS. |
| `nps_f_peak` | `float` | 1/mm | Peak frequency of the NPS. |
| `contrast_roi1`, `contrast_roi2` | `float` | 1 | RMS contrast (SD / mean) of each ROI. |
| `curves` | [`Curves`](#curves), can be empty | | Only with `curves=True`. |
| `plugin_values` | tuple of `PluginValues` | | Each plug-in's values for this pair (`bictool_pair_metrics`). |

How each value is computed: *Technical Reference* (physical QA recipe).

## `Curves`

*Since 0.1.* The curves behind one measurement (only with `curves=True`).

| Field | Type | Description |
|---|---|---|
| `mtf_frequency` | tuple of `float` | Frequencies of the MTF (1/mm). |
| `mtf_per_edge` | tuple of tuples of `float` | MTF per edge: `[edge][frequency]`. |
| `nps_frequency` | tuple of `float` | Frequencies of the NPS (1/mm). |
| `nps_roi1`, `nps_roi2` | tuple of `float` | Radially averaged NPS of each ROI (HU²·mm²). |
| `ratio_frequency` | tuple of `float` | Frequencies of the PS/MTF ratio (1/mm). |
| `ps_mtf_ratio` | tuple of `float` | PS/MTF ratio. |

## `Failure`

*Since 0.1.* A pair that could not be measured.

| Field | Type | Description |
|---|---|---|
| `slice` | [`SliceRef`](#sliceref) | The slice. |
| `pair` | `str` | `"falx"` or `"septum"`. |
| `message` | `str` | Why, e.g. "No ROI2 found" or the measurement's reason. |

## `Verdict`

*Since 0.1.* The study verdict of the physical recipe - a provisional model, decision support only.

| Field | Type | Description |
|---|---|---|
| `verdict` | `str` | `"acceptable"`, `"not_acceptable"`, `"uncertain"` or `"not_assessable"` (`api.ACCEPTABLE`, ...). |
| `probability` | `float` | The model's probability of "acceptable". |
| `probability_low`, `probability_high` | `float` | Its 95 % interval. |
| `slices_used` | `int` | Falx slices that entered the verdict. |
| `features` | tuple of [`FeatureFinding`](#featurefinding) | Each model feature over the study. |
| `reasons` | tuple of `str` | Feature names behind the verdict, strongest first. |
| `noise_scatter` | [`NoiseScatter`](#noisescatter), can be empty | Spread of the noise across slices (from 10 slices on). |
| `outlier_slices` | tuple of `int` | Slices whose ROI2 noise is more than 3 times the study median. |
| `septum_checks` | tuple of [`SeptumCheck`](#septumcheck) | Septum against falx, per value. |
| `slice_scores` | `dict[int, float]` | Indicative per-slice probabilities - the model is fitted on study medians. |
| `model_version` | `str` | The verdict model's version. |
| `metal_slices` | tuple of `int` | Slices with metal streaks in the brain (metal rule). |
| `model_verdict` | `str`, can be empty | The model's own verdict when the metal rule replaced it with "not_assessable". |

## `FeatureFinding`

*Since 0.1.* One model feature over the study.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | The feature (e.g. `"mtf_area"`). |
| `label` | `str` | Its readable name. |
| `unit` | `str` | Its unit. |
| `value` | `float` | Median over the study's slices. |
| `low`, `high` | `float` | 95 % interval of the median. |
| `good_low`, `good_high` | `float` | Range of the rated-good training studies. |
| `contribution` | `float` | Contribution to the model's logit against the training mean; below 0 = towards "not acceptable". |
| `outside` | `str`, can be empty | `"above"` / `"below"` when the whole interval lies outside the good range. |

## `NoiseScatter`

*Since 0.1.* Spread of the ROI2 noise across slices (IQR / median) - a possible-artefact indicator.

| Field | Type | Description |
|---|---|---|
| `value` | `float` | IQR / median. |
| `low`, `high` | `float` | Its 95 % interval. |
| `good_high` | `float` | Highest value of the rated-good training studies. |
| `flagged` | `bool` | The whole interval lies above every good training study. |

## `SeptumCheck`

*Since 0.1.* Septum median minus falx median of one value, against the range seen in training.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | The value. |
| `difference` | `float` | Septum median minus falx median. |
| `low`, `high` | `float` | Range seen in training. |
| `consistent` | `bool` | The difference lies within that range. |

## `ArtefactHints`

*Since 0.1.* Possible artefacts - descriptive; only metal streaks in the brain change the verdict.

| Field | Type | Description |
|---|---|---|
| `metal_slices` | tuple of `int` | Slices with metal (> 2800 HU). |
| `metal_max_hu` | `float`, can be empty | Highest HU found. |
| `metal_streak_slices` | tuple of `int` | Metal slices with streaks in the brain. |
| `metal_streak_share` | `dict[int, float]` | Metal slice -> share of nearby brain disturbed. |
| `brainstem_band_slices` | tuple of `int` | Slices with a dark band across the brainstem. |
| `brainstem_band_depth_hu` | `dict[int, float]` | Slice -> depth of the band (HU). |

## `BasalGangliaSlice`

*Since 0.1.* Border contrast of the basal ganglia against the internal capsule (provisional).

| Field | Type | Description |
|---|---|---|
| `border_hu` | `float` | Border contrast (HU). |
| `differences_hu` | `dict[str, float]` | Nucleus -> HU difference. |
| `plausible` | `bool` | Within the plausible range (2-15 HU). |

## `SegmentationCheck`

*Since 0.1.* Study-level check of the anatomy segmentation (label fragmentation, provisional).

| Field | Type | Description |
|---|---|---|
| `fragmentation` | `float` | Median number of extra label pieces per slice. |
| `slices` | `int` | Slices checked. |
| `worst_slices` | tuple of `int` | The most fragmented slices, most fragmented first. |
| `lobe_islands` | `int` | Pieces of a lobe surrounded by another lobe, over all checked slices. |
| `unreliable` | `bool` | The segmentation looks unreliable. |

The methods and thresholds behind the hints are described in `docs/qa_method/` and the *Technical Reference*.

## Units

`api.UNITS` (also in every JSON result under `"units"`):

| Key | Unit |
|---|---|
| `mtf_area` | 1/mm |
| `nps_homo`, `nps_struct` | HU^2*mm^2 |
| `var_struct` | HU^2 |
| `ratio_lf_hf`, `contrast_roi1`, `contrast_roi2`, `probability` | 1 |
| `nps_f_av`, `nps_f_peak`, `frequency` | 1/mm |
| `location_mm` | mm |
| `hu` | HU |
