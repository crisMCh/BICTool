# BICTool Python API

`bictool.api` runs BICTool's image-quality assessment of a head-CT series without the user interface: from a
script, a plug-in or an integration adapter (PACS/HIS). It runs the same pipeline as the program and the batch
tool, so a series gives the same numbers whichever way it is run.

> Decision support only. BICTool is not a medical device. The study verdict comes from a provisional model and
> must be checked by a qualified person.

**Status: provisional (API 0.x).** Names and the result layout may still change between minor versions; every
change is listed under [Changes](#changes). Only the names in `bictool.api.__all__` are the API - everything
else in `bictool` is internal.

## Installation

```bash
pip install -e ".[dicom,ml]"
```

The anatomy segmentation needs the CNN weights, configured as described in `reference/README.md`.
`import bictool.api` loads neither TensorFlow nor Qt; the first segmentation loads TensorFlow.

## One call

```python
from bictool import api

result = api.analyse_study("D:/studies/case1", label="case 1")

if result.verdict is not None:
    print(result.verdict.verdict, round(result.verdict.probability, 2))   # e.g. "acceptable 0.83"
with open("case1.json", "w", encoding="utf-8") as f:
    f.write(result.to_json())
```

`analyse_study` takes the series the program takes (CT, axial, closest to 4 mm), segments it (about 1-2 s per
slice), places the falx and septum ROI pairs on every suitable slice, measures them and computes the verdict
and the hints.

| Argument | Meaning |
|---|---|
| `series` | Which series of the folder (index from `list_series`; default 0 = the program's choice). |
| `label` | Name of the study in the result. Default `"study - <series description>"`, never the folder name (it often carries the scan date and time). |
| `settings` | `api.Settings` - QA recipe and WM/GM method (default: the program's defaults). |
| `model` | A loaded U-Net (`api.load_anatomy_model()`), to reuse it across studies. |
| `cache_file` | A file for the segmentation: read if it exists (it must belong to this series, else `InvalidSegmentation`), written otherwise. Derived from patient data - keep it out of version control. |
| `curves` | Keep the MTF/NPS curves of every measurement. |
| `progress`, `cancel` | `progress(stage, done, total)` with stage `"segmentation"`, `"rois"`, `"measurement"`; `cancel()` returning True stops between two slices (`api.Cancelled`). |

## Command line

```bash
python -m bictool                                   # starts the program
python -m bictool series D:/studies/case1           # lists the series
python -m bictool analyse D:/studies/case1 --label "case 1" --out case1.json
```

`analyse` also takes `--series N`, `--curves`, `--uids`, `--weights FILE` and `--cache FILE`. Exit code 0 =
done, 2 = the study could not be analysed (the reason is printed).

## In steps: check or change the ROIs before measuring

```python
for choice in api.list_series("D:/studies/case1"):
    print(choice.index, choice.label)                 # "0: CT · 37 slice(s) · 512×512 · Hr40 · 4mm · axial"

study = api.open_study("D:/studies/case1", series=0, label="case 1")
study.segment(progress=lambda stage, done, total: print(stage, done, total))

rois = study.place_rois()                         # api.RoiPairStore
falx_roi1 = api.roi_key(api.FALX, "roi1")         # slots: falx.roi1, falx.roi2, septum.roi1, septum.roi2
print(rois.slices(), rois.effective(12))          # slice indices are 0-based

rois.set_manual(12, falx_roi1, api.RoiRect(x0=240, y0=150, x1=268, y1=178))  # wins over the automatic ROI
rois.delete(13, api.roi_key(api.SEPTUM, "roi2"))  # no septum pair on slice 13
rois.restore_slice(14)                             # back to the automatic ROIs

result = study.measure(rois, reviewed=True)
```

- `Study.segment(model=None, cache_file=None, progress=None, cancel=None)`: slices already segmented are skipped,
  so after `Cancelled` a second call continues where the first stopped.
- `Study.set_segmentation(masks_by_slice)`: use a segmentation you already have - `{slice index: {label: mask}}`
  in DICOM pixel space, label names as the anatomy CNN's. Masks must match the images (`InvalidSegmentation`).
- `Study.place_rois()`: the automatic ROIs. ROI1 sits on the structure, ROI2 next to it (white matter for the
  falx, ventricle for the septum). No automatic ROIs on slices with less than 20 % of the largest brain area
  (`rois.skipped`).
- `Study.measure(rois=None, reviewed=False, curves=False)`:
  - Like the program, a **manual ROI1 gets a new automatic ROI2** next to it (unless ROI2 is manual too);
    `rois` is updated.
  - ROIs are checked first: whole pixels, inside the image, **square** (the noise power spectrum needs it), on
    this series. Any problem raises `InvalidRois` before anything is measured.
  - `reviewed` records that a person checked the ROIs; `result.rois_edited` records whether any ROI was moved,
    drawn or deleted.
  - A slice whose measurement fails (e.g. no usable edge for the MTF) is listed in `result.failed`; any other
    error is raised.
- `RoiPairStore.set_manual`, `delete`, `set_auto` refuse unknown slot names (`ValueError`).

## Settings

```python
settings = api.Settings(
    recipe=api.RECIPE_PHYSICAL,                  # or api.RECIPE_OCTAVE (legacy)
    tissue=api.TissueSettings(method=api.METHOD_FCM),
)
```

The WM/GM method only decides where the falx pair's ROI2 goes. Methods are short ids: `METHOD_ADAPTIVE`
(`"adaptive"`, default), `METHOD_FCM` (`"fcm"`), `METHOD_THRESHOLD` (`"threshold"`), `METHOD_ANCHORED`
(`"anchored"`, prototype); `METHOD_LABELS` gives the names the program shows. Each method's parameters are in
`TissueSettings.adaptive` / `.fcm` / `.threshold` (`AdaptiveSettings`, `FcmSettings`, `BrainTissueSettings`).
All settings are immutable and checked when created (`ValueError`); change one with `dataclasses.replace`.

## The result

`api.StudyResult` - the API's own frozen dataclasses:

| Field | Content |
|---|---|
| `label`, `series` | The study's label; `SeriesInfo`: description, slices, thickness, pixel spacing, scanner model, reconstruction kernel, kVp, mA, CTDIvol, DLP, acquisition date, Study/Series Instance UID. No patient name, ID or birth date. |
| `settings`, `rois_reviewed`, `rois_edited`, `api_version`, `schema_version` | How the result was produced. |
| `measurements` | One `Measurement` per (slice, pair): `slice` (`SliceRef`: index, position in mm, SOP Instance UID), `pair`, `roi1`, `roi2`, whether each was manual, `roi2_region`, and the values `mtf_area`, `nps_homo`, `nps_struct`, `var_struct`, `ratio_lf_hf` (the model's inputs), `nps_f_av`, `nps_f_peak`, `contrast_roi1`, `contrast_roi2`; `curves` if asked for. |
| `failed`, `skipped` | `Failure(slice, pair, message)`; per pair, slices with the structure but too little brain. |
| `verdict` | `Verdict` (physical recipe): `verdict` (`"acceptable"`, `"not_acceptable"`, `"uncertain"`, `"not_assessable"`), probability of acceptable and its 95 % interval, `features` (`FeatureFinding`: median, interval, good range, contribution), `reasons`, `noise_scatter`, `outlier_slices`, `septum_checks`, indicative `slice_scores`, metal rule (`metal_slices`, `model_verdict`), `model_version`. None for the legacy recipe or too few slices. |
| `artefacts`, `basal_ganglia`, `segmentation_check` | The hints of the program's Study summary (descriptive, provisional). |

Units are in `api.UNITS` (also written into every JSON result); values that could not be computed are NaN in
Python and `null` in JSON; slice indices are 0-based (the program shows index + 1).

- `result.measurements_for(api.FALX)` gives `{slice index: Measurement}`.
- `result.to_dict(include_curves=False, include_dicom_uids=False)` / `to_json(...)` - the DICOM UIDs are left
  out unless asked for, as they link the result to the patient in the PACS.
- `api.StudyResult.from_dict(data)` / `from_json(text)` read a result back (same schema major version).

## Versions

- `api.API_VERSION` - semantic versioning. In 0.x, minor versions may change names; from 1.0 on, names in
  `__all__` keep working within a major version.
- `api.SCHEMA_VERSION` - the layout of `to_dict()`: the minor number grows when keys are added, the major number
  when a key is renamed or removed. A test pins the layout. `from_dict` reads results of the same major version.

## Errors

Every error the API raises on purpose derives from `api.BictoolError` - and from the matching built-in error, so
`except ValueError` keeps working:

| Error | When |
|---|---|
| `SeriesError` (`ValueError`) | Folder missing, no DICOM series, several studies, or no such series number. |
| `ModelNotAvailable` (`FileNotFoundError`) | No weights configured, weights file missing, or TensorFlow not installed. |
| `NotSegmented` (`RuntimeError`) | ROIs requested before the segmentation. |
| `InvalidSegmentation` (`ValueError`) | A segmentation or cache that does not fit the series. |
| `InvalidRois` (`ValueError`) | ROIs that can't be measured or belong to another series. |
| `Cancelled` | `cancel()` returned True. |

## Data protection

The API reads DICOM files and returns numbers. It writes nothing except the optional segmentation cache.
Results contain no patient name, ID or birth date; the acquisition date is included (as in the exports), the
DICOM UIDs only on request. Keep caches and results derived from patient data out of version control.

## Plug-ins

Plug-ins add metrics, WM/GM methods and later other extensions - see [plugins.md](plugins.md). The API runs
the same plug-ins as the program: on first use it loads those in the plug-in folders and the installed ones.

```python
api.load_plugins(["D:/my-plugins"], installed=False)   # choose; returns each plug-in's status
api.load_plugins(folders=[], installed=False)           # run without plug-ins
print(api.loaded_plugins())
```

A result lists the active plug-ins (`StudyResult.plugins`, name -> version) and their values
(`Measurement.plugin_values`, `StudyResult.plugin_values`); plug-in values never enter the verdict. The
program's Settings > Plug-ins choice (switched-off plug-ins) applies to the program only.

## Changes

- **0.2.0** (2026-11-10) - plug-ins: `bictool.api.plugins` (hooks `bictool_pair_metrics`,
  `bictool_study_metrics`, `bictool_tissue_methods`), `load_plugins`, `loaded_plugins`; plug-in WM/GM methods
  as `"<plug-in>.<id>"` with `TissueSettings.plugin`; schema 1.1 adds `plugins` and `plugin_values`.
- **0.1.0** (2026-11-10) - first provisional version: `analyse_study`, `list_series`, `open_study`, `Study`,
  `load_anatomy_model`, the result types (schema 1.0), settings, ROI store, errors, `python -m bictool`.
