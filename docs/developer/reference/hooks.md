# Reference: Hooks

A hook is a module-level function of a plug-in, named like the hook and marked `@plugins.hookimpl`. BICTool
calls it at a defined moment. A plug-in implements only the hooks it needs, and may leave out arguments it does
not use. The hooks are listed here in the order BICTool calls them.

```text
Program start / api.load_plugins
  └─ bictool_tissue_methods()      once per plug-in: WM/GM methods it adds
  └─ bictool_export_formats()      once per plug-in: file formats it adds

QA run (program: Run QA · API: Study.measure / analyse_study · batch tool)
  ├─ for each ROI pair:      BICTool measures the pair, then
  │    └─ bictool_pair_metrics(pair)
  ├─ for each measured slice (after all pairs):
  │    └─ bictool_slice_metrics(ct_slice)
  ├─ bictool_study_metrics(study)
  └─ bictool_artefact_hints(study)

Export (program: Export > Whole study · api.write_export · batch --plugin-export)
  └─ ExportFormat.write(result, path)      of the chosen format

Whenever BICTool needs WM/GM masks with a plug-in's method selected
  └─ TissueMethod.classify(slice_hu, brain_mask, settings, context)
```

Rules for every hook:

| Rule | |
|---|---|
| **Isolation** | Each plug-in is called separately. If a hook raises, the error is recorded (`PluginValues.error`, `PluginHints.error`, the plug-in's status) and BICTool goes on - with the other plug-ins and the study. |
| **Checked output** | BICTool checks every return value; invalid entries are dropped with a note in `error`, valid ones kept. |
| **Read-only input** | Images and masks are read-only NumPy arrays; writing to them raises `ValueError`. Copy first. |
| **No verdict** | Nothing a hook returns enters the study verdict. |
| **Threads** | Hooks may run on a background thread (program) or the caller's thread (API, batch), one call at a time per study. Do not rely on a Qt event loop; protect shared state. |
| **Order** | The order in which plug-ins are called is not guaranteed; results are listed sorted by plug-in name. |
| **Arguments** | Passed by name; the argument names below are part of the contract. |

## When a plug-in is loaded

### `bictool_tissue_methods`

```text
@plugins.hookimpl
def bictool_tissue_methods()
        -> list[plugins.TissueMethod] | None
```

*Since 0.2 · called once, when the plug-in is loaded*

The WM/GM methods this plug-in adds.

**Arguments** none.

**Returns** a list of [`TissueMethod`](plugin-types.md#tissuemethod), or `None`.

**Checked** `id` must be non-empty without "."; `classify` callable; `settings` a dataclass instance or `None`.
Invalid methods are left out and named in the plug-in's status message.

**Appears** in the program's Brain Tissue > Method drop-down as *"<label> (plug-in)"*, with a settings dialog
built from `settings`; in the API as `TissueSettings(method="<plug-in>.<id>")`.

**Example**

```python
@plugins.hookimpl
def bictool_tissue_methods():
    return [plugins.TissueMethod("level", "Level split", classify, LevelSettings())]
```

**See also** [Add a WM/GM method](../how-to/wmgm-method.md) · [`TissueContext`](plugin-types.md#tissuecontext)

### `bictool_export_formats`

```text
@plugins.hookimpl
def bictool_export_formats()
        -> list[plugins.ExportFormat] | None
```

*Since 0.3 · called once, when the plug-in is loaded*

The file formats this plug-in writes study results in.

**Arguments** none.

**Returns** a list of [`ExportFormat`](plugin-types.md#exportformat), or `None`.

**Checked** `id` non-empty without "."; `label` non-empty; `write` callable; `extension` a dot and up to 16
letters, digits, `_` or `-`. Invalid formats are left out and named in the plug-in's status message.

**Appears** in the program's Export dialog (scope *Whole study*), in `api.export_formats()` /
`api.write_export(...)` and in the batch tool's `--plugin-export`.

**Example**

```python
@plugins.hookimpl
def bictool_export_formats():
    return [plugins.ExportFormat("short", "Registry JSON", ".json", write)]
```

**See also** [Add an export format](../how-to/export-format.md)

## During the measurement

### `bictool_pair_metrics`

```text
@plugins.hookimpl
def bictool_pair_metrics(pair: plugins.PairContext)
        -> list[plugins.MetricValue | plugins.RoiMarker] | None
```

*Since 0.2 · called once per measured ROI pair, right after BICTool measured it · program: background thread*

Extra values for one of BICTool's ROI pairs (falx or septum, on one slice).

| Argument | Type | Description |
|---|---|---|
| `pair` | [`PairContext`](plugin-types.md#paircontext) | The pair: slice image, ROIs and their pixels, CNN masks, BICTool's own values. |

**Returns** a list of [`MetricValue`](plugin-types.md#metricvalue) and [`RoiMarker`](plugin-types.md#roimarker),
or `None`.

**Checked** value names unique within the plug-in, values numbers (infinite values become NaN); markers of the
pair's slice and shape, not empty.

**Appears** in the slice note of the Per slice tab, the export's Plug-ins sheet (ROI pair column *Falx* /
*Septum*) and `Measurement.plugin_values`.

**Example**

```python
@plugins.hookimpl
def bictool_pair_metrics(pair):
    return [plugins.MetricValue("roi2_sd", float(pair.roi2_pixels.std(ddof=1)), "HU")]
```

**See also** [Tutorial: Your first plug-in](../tutorials/first-plugin.md)

### `bictool_slice_metrics`

```text
@plugins.hookimpl
def bictool_slice_metrics(ct_slice: plugins.SliceContext)
        -> list[plugins.MetricValue | plugins.RoiMarker] | None
```

*Since 0.3 · called once per slice with at least one measured ROI pair, after all pairs of the run · program:
background thread*

Extra values for one slice - typically from ROIs of the plug-in's own.

| Argument | Type | Description |
|---|---|---|
| `ct_slice` | [`SliceContext`](plugin-types.md#slicecontext) | The slice: image, CNN masks, BICTool's pairs on it, brain and WM/GM masks on demand. |

**Returns** a list of `MetricValue` and `RoiMarker`, or `None`.

**Checked** as `bictool_pair_metrics`.

**Appears** in the slice note, the export's Plug-ins sheet (ROI pair column *slice*), `StudyResult.slice_plugin_values`;
markers on the image with *Plug-in ROIs* (Run QA box) and in `StudyResult.plugin_markers(slice)`. When the
plug-in used `tissue()`, the WM/GM method is recorded with the values.

**Example**

```python
@plugins.hookimpl
def bictool_slice_metrics(ct_slice):
    white = ct_slice.tissue()[plugins.WHITE_MATTER]
    return [plugins.MetricValue("wm_mean", float(ct_slice.image[white].mean()), "HU")]
```

**See also** [Add a metric with ROIs of its own](../how-to/metric-with-own-roi.md)

## After the measurement

### `bictool_study_metrics`

```text
@plugins.hookimpl
def bictool_study_metrics(study: plugins.StudyContext)
        -> list[plugins.MetricValue | plugins.RoiMarker] | None
```

*Since 0.2 · called once per run, after the slice hooks · program: background thread*

Extra values for the whole series.

| Argument | Type | Description |
|---|---|---|
| `study` | [`StudyContext`](plugin-types.md#studycontext) | All measured pairs, and every slice's image, CNN masks, brain and WM/GM masks on demand. |

**Returns** a list of `MetricValue` and `RoiMarker`, or `None`. A marker needs its `slice_index`.

**Checked** as `bictool_pair_metrics`; a marker without `slice_index` or of a slice outside the series is
dropped.

**Appears** under *Details* on the Study summary, the export's Plug-ins sheet (*whole series*) and
`StudyResult.plugin_values`.

**Example**

```python
@plugins.hookimpl
def bictool_study_metrics(study):
    return [plugins.MetricValue("measured_slices", float(len(study.measured_slices)))]
```

### `bictool_artefact_hints`

```text
@plugins.hookimpl
def bictool_artefact_hints(study: plugins.StudyContext)
        -> list[plugins.ArtefactHint] | None
```

*Since 0.3 · called once per run, after `bictool_study_metrics` · program: background thread*

Possible artefacts the plug-in noticed: neutral, descriptive, no score.

| Argument | Type | Description |
|---|---|---|
| `study` | [`StudyContext`](plugin-types.md#studycontext) | As in `bictool_study_metrics`. |

**Returns** a list of [`ArtefactHint`](plugin-types.md#artefacthint), or `None`.

**Checked** text non-empty and at most 200 characters (white space collapsed to single spaces); slices whole
numbers within the series. Invalid hints are dropped with a note in `PluginHints.error`.

**Appears** on the Study summary with BICTool's own possible artefacts ("Possible artefact (plug-in name):
text (slice ...)"), in the notes of the named slices, in the export (ROI pair column *possible artefact*) and in
`StudyResult.plugin_hints`.

**Example**

```python
@plugins.hookimpl
def bictool_artefact_hints(study):
    return [plugins.ArtefactHint("Air-like values inside the brain mask", (11, 12))]
```

**See also** [Add possible-artefact hints](../how-to/artefact-hints.md)

## Names of the hooks

`plugins.HOOKS` lists every hook name of the running BICTool. A function named `bictool_...` that is not in
`HOOKS`, or a hook function without `@plugins.hookimpl`, is never called; the plug-in's status message names it.
