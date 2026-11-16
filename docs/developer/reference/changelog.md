# Reference: Changelog and versioning

## Versioning policy

BICTool has three version numbers:

| Version | Where | Rule |
|---|---|---|
| Program version | `bictool.__version__`, About dialog | The program as a whole. |
| API version | `bictool.api.API_VERSION` | Semantic versioning of the Python API **and** the plug-in interface. |
| Result schema | `bictool.api.SCHEMA_VERSION`, `schema_version` in every result | The layout of `StudyResult.to_dict()` / JSON. |

**API version (major.minor.patch):**

* **0.x - provisional (now).** A minor version may still change or remove names; every change is listed below.
  Plug-ins should accept a range they have tested, e.g. `requires_api=">=0.3,<0.4"`, or `">=0.3,<1"` to accept
  compatible changes.
* **From 1.0 on**, names in `bictool.api.__all__` and `bictool.api.plugins.__all__` keep working within a major
  version: a minor version only adds (new functions, hooks, fields), a major version may remove or rename.
* New hooks and new fields are added in minor versions; existing plug-ins keep working.

**Result schema (major.minor):** the minor number grows when keys are added, the major number when a key is
renamed or removed. `StudyResult.from_dict` reads results of the same major version. A test pins the layout.

## Changes

### 0.3.0 (2026-11-11) - schema 1.2

Plug-ins:

* New hook [`bictool_slice_metrics`](hooks.md#bictool_slice_metrics) with [`SliceContext`](plugin-types.md#slicecontext):
  values per measured slice.
* New [`RoiMarker`](plugin-types.md#roimarker) in the return value of every metric hook: where a plug-in
  measured, drawn on the image (*Plug-in ROIs*), kept as [`PluginMarker`](plugin-types.md#pluginmarker) in
  `PluginValues.markers`.
* New hook [`bictool_artefact_hints`](hooks.md#bictool_artefact_hints) with [`ArtefactHint`](plugin-types.md#artefacthint)
  and [`PluginHints`](plugin-types.md#pluginhints).
* New hook [`bictool_export_formats`](hooks.md#bictool_export_formats) with [`ExportFormat`](plugin-types.md#exportformat).
* `StudyContext` gains `brain_mask`, `tissue`, `tissue_method` and `measured_slices`.
* `RoiRect` is also in `bictool.api.plugins`.
* A plug-in's status message names `bictool_*` functions that are never called (unknown hook, missing
  `@hookimpl`).
* Plug-in WM/GM methods get the CNN masks and pixel spacing in the program too (as in the API).

API:

* New [`export_formats`](api-functions.md#export_formats), [`write_export`](api-functions.md#write_export) and
  the error [`PluginFailed`](errors.md#pluginfailed).
* `StudyResult` gains `slice_plugin_values`, `plugin_hints` and `plugin_markers()`.
* `TissueSettings.plugin` may be a dict of the fields that differ from the plug-in's defaults.
* Fixed: `StudyResult.from_dict` failed on a result whose WM/GM method came from a plug-in; its settings now
  come back as a dict.

Schema 1.2: adds `slice_plugin_values`, `plugin_hints`, and `markers` in plug-in values.

### 0.2.0 (2026-11-10) - schema 1.1

* Plug-ins: `bictool.api.plugins` with the hooks `bictool_pair_metrics`, `bictool_study_metrics` and
  `bictool_tissue_methods`; `load_plugins`, `loaded_plugins`.
* Plug-in WM/GM methods as `"<plug-in>.<id>"` with `TissueSettings.plugin`.
* Schema 1.1: adds `plugins` and `plugin_values`.

### 0.1.0 (2026-11-10) - schema 1.0

* First provisional version: `analyse_study`, `list_series`, `open_study`, `Study`, `load_anatomy_model`, the
  result types, settings, the ROI store, the errors and `python -m bictool`.
