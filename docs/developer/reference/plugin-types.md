# Reference: Plug-in types

Everything a plug-in imports comes from one module:

```python
from bictool.api import plugins
```

All types are frozen dataclasses. Arrays handed to a plug-in are read-only.

| Kind | Names |
|---|---|
| Describing the plug-in | [`PluginInfo`](#plugininfo), [`hookimpl`](#hookimpl) |
| What hooks receive | [`PairContext`](#paircontext), [`SliceContext`](#slicecontext), [`StudyContext`](#studycontext), [`TissueContext`](#tissuecontext), [`RoiRect`](#roirect) |
| What hooks return | [`MetricValue`](#metricvalue), [`RoiMarker`](#roimarker), [`ArtefactHint`](#artefacthint), [`TissueMethod`](#tissuemethod), [`ExportFormat`](#exportformat) |
| What BICTool keeps | [`PluginValues`](#pluginvalues), [`PluginMarker`](#pluginmarker), [`PluginHints`](#pluginhints), [`LoadedPlugin`](#loadedplugin) |
| Constants | [`LABELS`, `WHITE_MATTER`, `GRAY_MATTER`, `CSF`, `HOOKS`, ...](#constants) |

## Describing the plug-in

### `PluginInfo`

```text
BICTOOL_PLUGIN = plugins.PluginInfo(name, version, requires_api, description="", author="")
```

*Since 0.2.* Every plug-in module needs this description as the module attribute `BICTOOL_PLUGIN`.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Unique name, no "." (e.g. `"roi2-sd"`). Shown in the program and results; prefixes method and format ids. |
| `version` | `str` | The plug-in's version, recorded in every result (`StudyResult.plugins`). |
| `requires_api` | `str` | The `bictool.api` versions it works with, as a version range: `">=0.3,<1"`. Outside the range the plug-in is *incompatible* and not loaded. |
| `description` | `str` | One line, shown in Settings > Plug-ins. |
| `author` | `str` | Who to ask. |

### `hookimpl`

```text
@plugins.hookimpl
def bictool_pair_metrics(pair): ...
```

*Since 0.2.* Marks a function as a hook implementation. Without it the function is never called (the status
message says so). The hooks are described in [Hooks](hooks.md).

## What hooks receive

### `PairContext`

*Since 0.2.* One measured ROI pair - input of [`bictool_pair_metrics`](hooks.md#bictool_pair_metrics); also in
`SliceContext.pairs` and `StudyContext.pairs`.

| Field | Type | Description |
|---|---|---|
| `slice_index` | `int` | 0-based slice. |
| `pair` | `str` | `"falx"` or `"septum"`. |
| `image` | `ndarray` (rows, columns) | The slice in HU. |
| `pixel_spacing_mm` | `float` | mm per pixel. |
| `roi1` | [`RoiRect`](#roirect) | ROI1, on the structure. |
| `roi2` | `RoiRect` | ROI2, in homogeneous tissue next to it. |
| `labels` | `Mapping[str, ndarray]` or `None` | The anatomy CNN's masks by name ([`LABELS`](#constants)); `None` if the slice is not segmented. |
| `bictool_values` | `Mapping[str, float]` | BICTool's own measurement: `mtf_area`, `nps_homo`, `nps_struct`, `var_struct`, `ratio_lf_hf`, `nps_f_av`, `nps_f_peak`, `contrast_roi1`, `contrast_roi2` (as in [`Measurement`](results.md#measurement)). |
| `roi1_pixels` | `ndarray` (property) | `image[roi1.y0:roi1.y1, roi1.x0:roi1.x1]`. |
| `roi2_pixels` | `ndarray` (property) | Likewise for ROI2. |

### `SliceContext`

*Since 0.3.* One measured slice - input of [`bictool_slice_metrics`](hooks.md#bictool_slice_metrics).

| Field | Type | Description |
|---|---|---|
| `slice_index` | `int` | 0-based slice. |
| `image` | `ndarray` | The slice in HU. |
| `pixel_spacing_mm` | `float` | mm per pixel. |
| `labels` | `Mapping[str, ndarray]` or `None` | The CNN's masks, as in `PairContext`. |
| `pairs` | tuple of `PairContext` | BICTool's ROI pairs measured on this slice. |
| `brain_mask` | `() -> ndarray` | BICTool's brain mask (from the segmentation; else the HU skull-strip). Computed on first call. |
| `tissue` | `() -> Mapping[str, ndarray]` | BICTool's WM/GM/CSF masks: `{WHITE_MATTER: mask, GRAY_MATTER: mask, CSF: mask}`, from the method in `tissue_method`. Computed on first call (milliseconds to about a second). |
| `tissue_method` | `str` | The WM/GM method `tissue()` uses: chosen in the program or `api.Settings`. Recorded with the values when `tissue()` was called. |

### `StudyContext`

*Since 0.2; `brain_mask`, `tissue`, `tissue_method`, `measured_slices` since 0.3.* The whole series after the
measurement - input of [`bictool_study_metrics`](hooks.md#bictool_study_metrics) and
[`bictool_artefact_hints`](hooks.md#bictool_artefact_hints).

| Field | Type | Description |
|---|---|---|
| `num_slices` | `int` | Slices in the series. |
| `pixel_spacing_mm` | `float` | mm per pixel. |
| `slice_thickness_mm` | `float` or `None` | Slice thickness. |
| `segmented_slices` | tuple of `int` | Slices with a segmentation. |
| `measured_slices` | tuple of `int` | Slices with at least one measured ROI pair. |
| `pairs` | tuple of `PairContext` | Every measured ROI pair. |
| `image` | `(index) -> ndarray` | Slice `index` in HU. |
| `labels` | `(index) -> Mapping or None` | The CNN's masks of slice `index`. |
| `brain_mask` | `(index) -> ndarray` | As in `SliceContext`. |
| `tissue` | `(index) -> Mapping` | As in `SliceContext`. |
| `tissue_method` | `str` | As in `SliceContext`. |

### `TissueContext`

*Since 0.2.* Extra input of `TissueMethod.classify`.

| Field | Type | Description |
|---|---|---|
| `slice_index` | `int` or `None` | 0-based slice (`None` if not known). |
| `pixel_spacing_mm` | `float` or `None` | mm per pixel. |
| `labels` | `Mapping[str, ndarray]` or `None` | The CNN's masks of the slice, `None` if not segmented. |

### `RoiRect`

*Since 0.3 in `plugins` (0.1 in `bictool.api`).* An ROI rectangle `RoiRect(x0, y0, x1, y1)`; pixels
`image[y0:y1, x0:x1]`. See [ROIs](rois.md#roirect).

## What hooks return

### `MetricValue`

```text
plugins.MetricValue(name, value, unit="", description="")
```

*Since 0.2.*

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Unique within the plug-in's return value; shown and exported as "<plug-in>: <name>". |
| `value` | `float` | A number; NaN for "could not compute". Infinite values become NaN; a non-number is dropped with a note. |
| `unit` | `str` | E.g. `"HU"`, `"1/mm"`, `""` for none. |
| `description` | `str` | One line, for exports and tooltips. |

### `RoiMarker`

```text
plugins.RoiMarker(name, mask, slice_index=None, description="")
```

*Since 0.3.* Where a plug-in measured; BICTool draws it on the image. Allowed in the return value of every
metric hook.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Shown when hovering over the marker. |
| `mask` | bool `ndarray` | The slice's shape; any shape of ROI. Not empty. |
| `slice_index` | `int` or `None` | Needed in `bictool_study_metrics`; the other hooks use their own slice. |
| `description` | `str` | One line. |

### `ArtefactHint`

```text
plugins.ArtefactHint(text, slices=())
```

*Since 0.3.* A possible artefact, from [`bictool_artefact_hints`](hooks.md#bictool_artefact_hints).

| Field | Type | Description |
|---|---|---|
| `text` | `str` | One neutral line, at most 200 characters. |
| `slices` | tuple of `int` | The 0-based slices it concerns (may be empty). |

### `TissueMethod`

```text
plugins.TissueMethod(id, label, classify, settings=None, description="")
```

*Since 0.2.* A WM/GM method, from [`bictool_tissue_methods`](hooks.md#bictool_tissue_methods).

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique within the plug-in, no "."; selected as `"<plug-in>.<id>"`. |
| `label` | `str` | Shown in the Method drop-down. |
| `classify` | callable | `classify(slice_hu, brain_mask, settings, context) -> {WHITE_MATTER: mask, GRAY_MATTER: mask, CSF: mask}`: bool masks of the slice's shape; a missing mask counts as empty; anything outside `brain_mask` is dropped. `context` is a [`TissueContext`](#tissuecontext). |
| `settings` | frozen dataclass instance or `None` | The defaults. The program builds a settings dialog from its `int`, `float` and `bool` fields. |
| `description` | `str` | One line. |

### `ExportFormat`

```text
plugins.ExportFormat(id, label, extension, write, description="")
```

*Since 0.3.* A file format for study results, from [`bictool_export_formats`](hooks.md#bictool_export_formats).

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique within the plug-in, no "."; used as `"<plug-in>.<id>"`. |
| `label` | `str` | Shown in the Export dialog. |
| `extension` | `str` | E.g. `".json"`: a dot and up to 16 letters, digits, `_` or `-`. |
| `write` | callable | `write(result, path)`: writes the [`StudyResult`](results.md#studyresult) to `path` (a `pathlib.Path` ending with `extension`). |
| `description` | `str` | One line. |

## What BICTool keeps

### `PluginValues`

*Since 0.2; `markers` since 0.3.* What one plug-in reported for one ROI pair, slice or study.

| Field | Type | Description |
|---|---|---|
| `plugin` | `str` | The plug-in's name. |
| `values` | tuple of `MetricValue` | The valid values. |
| `error` | `str` or `None` | What went wrong: the hook's exception, or what was dropped from its return value. |
| `markers` | tuple of [`PluginMarker`](#pluginmarker) | The valid markers. |

### `PluginMarker`

*Since 0.3.* A checked `RoiMarker`, the mask cropped to its bounding box.

| Field | Type | Description |
|---|---|---|
| `plugin` | `str` | The plug-in's name. |
| `name` | `str` | The marker's name. |
| `slice_index` | `int` | 0-based slice. |
| `box` | `RoiRect` | Bounding box of the mask in the slice. |
| `mask` | bool `ndarray` | The mask inside `box`, shape `(box.height, box.width)`. |
| `description` | `str` | One line. |

| Method | Returns |
|---|---|
| `full_mask(shape)` | The marker as a bool array of the slice's `shape`. |

### `PluginHints`

*Since 0.3.* The possible artefacts one plug-in noticed in a study.

| Field | Type | Description |
|---|---|---|
| `plugin` | `str` | The plug-in's name. |
| `hints` | tuple of `ArtefactHint` | The valid hints. |
| `error` | `str` or `None` | What went wrong or was dropped. |

### `LoadedPlugin`

*Since 0.2.* A plug-in BICTool found (`api.load_plugins`, `api.loaded_plugins`, Settings > Plug-ins).

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Its name (the file or package if it has no valid `PluginInfo`). |
| `version` | `str` | Its version. |
| `source` | `str` | Where it came from: a file, a folder or a package. |
| `status` | `str` | `"active"`, `"disabled"`, `"incompatible"` or `"error"`. |
| `message` | `str` | Why it is not active, and notes (e.g. functions that will never be called). |
| `description` | `str` | From `PluginInfo`. |
| `hooks` | tuple of `str` | The hooks it implements. |

## Constants

| Name | Value |
|---|---|
| `LABELS` | The anatomy CNN's 16 structure names (keys of `labels`): "Brain Parenchyma", "Subarachnoid Space", "Dural Folds", "Septum Pellucidum", "Cerebellum", "Caudate Nucleus", "Lentiform Nucleus", "Insular Cortex", "Internal Capsule", "Ventricular System", "Central Sulcus", "Frontal Lobe", "Parietal Lobe", "Occipital Lobe", "Temporal Lobe", "Thalamus". |
| `WHITE_MATTER`, `GRAY_MATTER`, `CSF` | `"White Matter"`, `"Gray Matter"`, `"CSF"` - keys of WM/GM masks. |
| `HOOKS` | The hook names of the running BICTool. |
| `HookSpecs` | The hook specifications (for tools; plug-ins do not need it). |
| `hookspec` | pluggy's specification marker (for BICTool itself). |
| `PROJECT` | `"bictool"` - pluggy's project name. |
| `ENTRY_POINT_GROUP` | `"bictool.plugins"` - for installed packages. |
