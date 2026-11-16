# Reference: Functions and the Study class

All names are in `bictool.api` (`from bictool import api`). Arguments after `*` are keyword-only. The status
line under each signature says since which API version the name exists, what it reads and what it loads.

| Name | Purpose |
|---|---|
| [`analyse_study`](#analyse_study) | The whole assessment of one series in one call. |
| [`list_series`](#list_series) | The series in a study folder. |
| [`open_study`](#open_study) | Open one series for step-by-step work: returns a `Study`. |
| [`Study`](#study) | One series and what has been computed on it: `segment`, `place_rois`, `measure`. |
| [`load_anatomy_model`](#load_anatomy_model) | Load the anatomy CNN once, to reuse it. |
| [`load_plugins`](#load_plugins) | Choose which plug-ins the API uses. |
| [`loaded_plugins`](#loaded_plugins) | The plug-ins in use and their status. |
| [`export_formats`](#export_formats) | The file formats plug-ins add. |
| [`write_export`](#write_export) | Write a result in a plug-in's file format. |

Progress and cancel callbacks have the same form everywhere:

* `progress(stage, done, total)` - `stage` is `"segmentation"`, `"rois"` or `"measurement"`; called after each
  slice or ROI pair.
* `cancel()` - return `True` to stop; the step stops between two slices and raises
  [`Cancelled`](errors.md#cancelled).

## Running an assessment

### `analyse_study`

```text
api.analyse_study(path, *, series=0, label=None, settings=None, model=None, cache_file=None,
                  curves=False, progress=None, cancel=None) -> StudyResult
```

*Since 0.1 · reads the study folder · runs the anatomy CNN (loads TensorFlow) · uses the plug-ins in use*

The whole assessment of one series: chooses the series, segments it, places the falx and septum ROI pairs on
every slice with enough brain, measures them, and computes the verdict and the hints. Equivalent to
`open_study(...)`, `Study.segment(...)`, `Study.place_rois()` and `Study.measure(...)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` or `Path` | | Folder of one study (or one DICOM file). |
| `series` | `int` | `0` | Which series, as numbered by [`list_series`](#list_series); 0 = the program's choice (CT, axial, closest to 4 mm). |
| `label` | `str` or `None` | `None` | Name of the study in the result. `None`: `"study - <series description>"`. Never the folder name, which often carries the scan date. |
| `settings` | [`Settings`](settings.md#settings) or `None` | `None` | QA recipe and WM/GM method; `None`: the program's defaults. |
| `model` | anatomy model or `None` | `None` | A model from [`load_anatomy_model`](#load_anatomy_model), to reuse it across studies; `None`: loaded from the configured weights. |
| `cache_file` | `str`, `Path` or `None` | `None` | Segmentation cache: read if it exists (it must belong to this series), else written when the segmentation is complete. Derived from patient data. |
| `curves` | `bool` | `False` | Keep the MTF/NPS curves of every measurement (`Measurement.curves`). |
| `progress` | callable or `None` | `None` | `progress(stage, done, total)`, see above. |
| `cancel` | callable or `None` | `None` | `cancel() -> bool`, see above. |

**Returns** a [`StudyResult`](results.md#studyresult) with `rois_reviewed=False`.

**Raises** [`SeriesError`](errors.md#serieserror) (folder missing, no usable series, several studies, no such
series number), [`ModelNotAvailable`](errors.md#modelnotavailable), [`InvalidSegmentation`](errors.md#invalidsegmentation)
(cache of another series), [`Cancelled`](errors.md#cancelled).

**Example**

```python
result = api.analyse_study("D:/studies/case1", label="case 1")
print(result.verdict.verdict if result.verdict else "no verdict")
```

**See also** [Tutorial: Your first analysis](../tutorials/first-analysis.md) ·
[Analyse many studies](../how-to/run-many-studies.md)

### `list_series`

```text
api.list_series(path) -> list[SeriesChoice]
```

*Since 0.1 · reads the study folder*

The series in a study folder, best first: CT, axial, closest to 4 mm. The first one is the program's default
choice.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` or `Path` | Folder of one study (or one DICOM file). |

**Returns** a list of [`SeriesChoice`](results.md#serieschoice) (`index`, `label`, `slices`, `axial`).

**Raises** [`SeriesError`](errors.md#serieserror).

**Example**

```python
for choice in api.list_series("D:/studies/case1"):
    print(choice.index, choice.label)       # 0 CT · 37 slice(s) · 512×512 · Hr40 · 4mm · axial
```

### `open_study`

```text
api.open_study(path, *, series=0, label=None, settings=None) -> Study
```

*Since 0.1 · reads the study folder*

Opens one series for step-by-step work. Nothing is computed yet.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` or `Path` | | Folder of one study. |
| `series` | `int` | `0` | As in [`analyse_study`](#analyse_study). |
| `label` | `str` or `None` | `None` | As in [`analyse_study`](#analyse_study). |
| `settings` | [`Settings`](settings.md#settings) or `None` | `None` | As in [`analyse_study`](#analyse_study). |

**Returns** a [`Study`](#study).

**Raises** [`SeriesError`](errors.md#serieserror).

### `load_anatomy_model`

```text
api.load_anatomy_model(weights_path=None) -> model
```

*Since 0.1 · reads the weights file · loads TensorFlow*

The anatomy CNN (U-Net) with its weights. Load it once and pass it as `model` to analyse several studies
without reloading.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `weights_path` | `str`, `Path` or `None` | `None` | The weights file; `None`: the configured weights (`BICTOOL_HEADCT_WEIGHTS` or `~/.bictool/config.json`). |

**Returns** an opaque model object for `model=` arguments.

**Raises** [`ModelNotAvailable`](errors.md#modelnotavailable) (no weights configured, file missing, or
TensorFlow not installed).

## The Study class

### `Study`

```text
api.Study(volume, *, label=None, settings=None)
```

*Since 0.1*

One series of a study and what has been computed on it so far. Create it with [`open_study`](#open_study);
the constructor takes BICTool's internal volume object and is not meant to be called directly.

| Attribute | Type | Description |
|---|---|---|
| `label` | `str` | The study's name in results. |
| `series` | [`SeriesInfo`](results.md#seriesinfo) | The series' acquisition parameters. |
| `settings` | [`Settings`](settings.md#settings) | Used by `place_rois` and `measure`. |
| `series_key` | `str` | Identifies the series (from its file names); stored in ROI stores and caches. |
| `masks_by_slice` | `dict[int, dict[str, ndarray]]` | The segmentation: slice index -> CNN label name -> bool mask. Empty before `segment`. |
| `confidence_by_slice` | `dict` | The CNN's confidence per slice (recorded, not used). |

| Method | Purpose |
|---|---|
| [`segment`](#studysegment) | Run (or continue) the anatomy CNN. |
| [`set_segmentation`](#studyset_segmentation) | Use a segmentation you already have. |
| `is_segmented()` | `True` once at least one slice is segmented. |
| [`place_rois`](#studyplace_rois) | The automatic ROI pairs. |
| [`measure`](#studymeasure) | QA on the ROI pairs, then verdict and hints. |

### `Study.segment`

```text
study.segment(model=None, *, cache_file=None, progress=None, cancel=None) -> int
```

*Since 0.1 · runs the anatomy CNN (about 1-2 s per slice) · reads/writes `cache_file`*

Segments every slice not segmented yet. Slices already segmented are skipped, so after
[`Cancelled`](errors.md#cancelled) a second call continues where the first stopped.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | anatomy model or `None` | `None` | From [`load_anatomy_model`](#load_anatomy_model); `None`: loaded when needed. |
| `cache_file` | `str`, `Path` or `None` | `None` | Read if it exists (must belong to this series), written when complete. |
| `progress`, `cancel` | callables or `None` | `None` | Stage `"segmentation"`. |

**Returns** the number of segmented slices.

**Raises** [`ModelNotAvailable`](errors.md#modelnotavailable), [`InvalidSegmentation`](errors.md#invalidsegmentation),
[`Cancelled`](errors.md#cancelled).

### `Study.set_segmentation`

```text
study.set_segmentation(masks_by_slice, confidence_by_slice=None) -> None
```

*Since 0.1*

Use a segmentation from elsewhere instead of the CNN: `{slice index: {label name: bool mask}}` in the DICOM
pixel grid, label names as the anatomy CNN's ([`LABELS`](plugin-types.md#constants)).

**Raises** [`InvalidSegmentation`](errors.md#invalidsegmentation) (unknown slice, or a mask whose shape differs
from the image).

### `Study.place_rois`

```text
study.place_rois(progress=None) -> RoiPairStore
```

*Since 0.1*

The automatic falx and septum ROI pairs on every segmented slice with at least 20 % of the largest brain area.
ROI1 lies on the structure (falx or septum), ROI2 next to it in homogeneous tissue (white matter for the falx,
ventricle for the septum). Change them before measuring, see
[Check or change the ROIs](../how-to/check-rois-before-measuring.md).

**Returns** a [`RoiPairStore`](rois.md#roipairstore), with `skipped` and `unplaced` filled in.

**Raises** [`NotSegmented`](errors.md#notsegmented).

### `Study.measure`

```text
study.measure(rois=None, *, reviewed=False, curves=False, head_ct_model=<shipped model>,
              progress=None, cancel=None) -> StudyResult
```

*Since 0.1 · calls the plug-ins in use*

QA on every complete ROI pair of `rois`, then the hints, the verdict and the plug-ins' slice, series and hint
hooks.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rois` | [`RoiPairStore`](rois.md#roipairstore) or `None` | `None` | The ROIs; `None`: `place_rois()`. Updated in place: a manual ROI1 gets a new automatic ROI2 next to it (unless its ROI2 is manual too), as in the program. |
| `reviewed` | `bool` | `False` | A person checked the ROIs; recorded in `StudyResult.rois_reviewed`. |
| `curves` | `bool` | `False` | Keep the curves of every measurement. |
| `head_ct_model` | model or `None` | the shipped model | The verdict model; `None`: no verdict. |
| `progress`, `cancel` | callables or `None` | `None` | Stage `"measurement"`. |

**Returns** a [`StudyResult`](results.md#studyresult). A pair whose measurement fails (e.g. no usable edge for
the MTF) is listed in `failed`; the others are measured.

**Raises** [`NotSegmented`](errors.md#notsegmented), [`InvalidRois`](errors.md#invalidrois) (before anything
is measured: ROIs of another series, outside the image, not square, not whole pixels),
[`Cancelled`](errors.md#cancelled).

## Plug-ins

### `load_plugins`

```text
api.load_plugins(folders=None, *, installed=True, disabled=()) -> list[LoadedPlugin]
```

*Since 0.2 · imports the plug-in files*

(Re)loads the plug-ins every later API call uses.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `folders` | list of `str`/`Path` or `None` | `None` | Folders with `.py` files or package folders; `None`: the plug-in folders (`~/.bictool/plugins`, `BICTOOL_PLUGIN_PATH`). |
| `installed` | `bool` | `True` | Also load installed packages (entry point group `bictool.plugins`). |
| `disabled` | iterable of `str` | `()` | Plug-in names to skip. |

**Returns** every plug-in found as [`LoadedPlugin`](plugin-types.md#loadedplugin), with status and message.

Without a call to `load_plugins`, the API loads the plug-in folders and installed packages on first use.
`api.load_plugins(folders=[], installed=False)` runs without plug-ins.

### `loaded_plugins`

```text
api.loaded_plugins() -> list[LoadedPlugin]
```

*Since 0.2*

The plug-ins in use (loading them on first use), with status and message.

### `export_formats`

```text
api.export_formats() -> dict[str, str]
```

*Since 0.3*

The file formats the plug-ins in use add: `{"<plug-in>.<id>": label}`.

### `write_export`

```text
api.write_export(result, path, format_id) -> Path
```

*Since 0.3 · writes a file (by the plug-in)*

Writes `result` in a plug-in's file format.

| Parameter | Type | Description |
|---|---|---|
| `result` | [`StudyResult`](results.md#studyresult) | The result to write. |
| `path` | `str` or `Path` | The file; the format's extension is added if missing, the folder is created. |
| `format_id` | `str` | `"<plug-in>.<id>"` from [`export_formats`](#export_formats). |

**Returns** the path written.

**Raises** [`PluginFailed`](errors.md#pluginfailed) (format not available, or the plug-in failed).

## Constants

| Name | Value | Meaning |
|---|---|---|
| `API_VERSION` | `"0.3.0"` | This API's version ([Changelog and versioning](changelog.md)). |
| `SCHEMA_VERSION` | `"1.2"` | The layout of `StudyResult.to_dict()` ([JSON format](json-format.md)). |
| `UNITS` | dict | Unit of each value, e.g. `UNITS["mtf_area"] == "1/mm"` ([Result types](results.md#units)). |
| `FALX`, `SEPTUM`, `PAIRS`, `SLOTS` | `"falx"`, `"septum"`, both, `("roi1", "roi2")` | ROI pairs and slots ([ROIs](rois.md)). |
| `ACCEPTABLE`, `NOT_ACCEPTABLE`, `UNCERTAIN`, `NOT_ASSESSABLE` | `"acceptable"`, ... | Verdict ids ([`Verdict`](results.md#verdict)). |
| `VENTRICLE`, `WHITE_MATTER`, `FALLBACK` | `"ventricle"`, ... | ROI2 regions ([`Measurement`](results.md#measurement)). |
| `RECIPE_PHYSICAL`, `RECIPE_OCTAVE` | `"physical"`, `"octave"` | QA recipes ([Settings](settings.md)). |
| `METHOD_ADAPTIVE`, `METHOD_FCM`, `METHOD_THRESHOLD`, `METHOD_ANCHORED`, `METHOD_LABELS` | `"adaptive"`, ... | WM/GM methods ([Settings](settings.md)). |
| `plugins` | module | What plug-ins import ([Plug-in types](plugin-types.md)). |
