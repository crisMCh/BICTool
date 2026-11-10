# Writing BICTool plug-ins

A plug-in adds to what BICTool measures or how it works, without changing BICTool itself (FR-08). It is
ordinary Python code that BICTool calls at defined points ("hooks"). The same plug-ins run in the program, in the
Python API (`bictool.api`, [api.md](api.md)) and in the batch tool.

> Plug-in values are descriptive. They never change the study verdict, which comes from BICTool's own provisional
> model. Like everything in BICTool, they are decision support only.

## A first plug-in

```python
# roi_stats.py
from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(
    name="roi-stats",            # unique, no "."
    version="1.0",
    requires_api=">=0.2,<1",     # the bictool.api versions this plug-in works with
    description="Mean and SD of both ROIs",
)

@plugins.hookimpl
def bictool_pair_metrics(pair):
    return [plugins.MetricValue("roi2_sd", float(pair.roi2_pixels.std(ddof=1)), "HU", "Pixel SD of ROI2")]
```

Two complete examples are in [examples/plugins](../examples/plugins): `roi_stats.py` (metrics) and `otsu_wmgm.py`
(a WM/GM method). They are examples, not validated measures.

## Where BICTool finds plug-ins

| How | Where |
|---|---|
| Copied file or folder | A `.py` file or a package folder (with `__init__.py`) in a plug-in folder: `plugins/` next to `BICTool.exe` (portable program), `~/.bictool/plugins` (run from source), and every folder in the environment variable `BICTOOL_PLUGIN_PATH`. Names starting with `_` or `.` are ignored. |
| Installed package | A Python package with an entry point in the group `bictool.plugins`, e.g. in its `pyproject.toml`: `[project.entry-points."bictool.plugins"]` `roi-stats = "roi_stats"`. |

Plug-ins are loaded when the program starts (Settings > Plug-ins lists them and switches them on or off; changes
take effect after a restart) and when the API is first used (`api.load_plugins(...)` to choose). A plug-in in
the portable program can use only what the program contains: Python's standard library, NumPy, SciPy and
BICTool itself.

Each plug-in gets a status: **active**; **disabled** (switched off); **incompatible** (its `requires_api` does not
include the running `bictool.api.API_VERSION`); **error** (it could not be imported, has no `BICTOOL_PLUGIN`, its
name is taken, a hook has unknown arguments, or it implements no hook - the message says which).

## The hooks

A hook is a module-level function with the hook's name, marked `@plugins.hookimpl`. A plug-in implements only the
hooks it needs, and may leave out arguments it does not use. New hooks may come in later versions; existing
plug-ins keep working.

### `bictool_pair_metrics(pair)` - values for one ROI pair

Called once for every measured ROI pair (falx and septum, every slice). `pair` is a `plugins.PairContext`:

| Field | Content |
|---|---|
| `slice_index`, `pair` | 0-based slice; `"falx"` or `"septum"`. |
| `image`, `pixel_spacing_mm` | The slice in HU (rows × columns), mm per pixel. |
| `roi1`, `roi2`, `roi1_pixels`, `roi2_pixels` | The ROIs (`RoiRect`: x0, y0, x1, y1) and their pixels. ROI1 lies on the structure, ROI2 in homogeneous tissue next to it. |
| `labels` | The anatomy CNN's masks by name (`plugins.LABELS`), None for a slice without segmentation. |
| `bictool_values` | BICTool's own measurement: `mtf_area`, `nps_homo`, `nps_struct`, `var_struct`, `ratio_lf_hf`, `nps_f_av`, `nps_f_peak`, `contrast_roi1`, `contrast_roi2` (units in `bictool.api.UNITS`). |

Return a list of `plugins.MetricValue(name, value, unit="", description="")`, or None. Names must be unique within
the plug-in; values must be numbers (NaN for "could not compute").

### `bictool_study_metrics(study)` - values for the whole series

Called once after all pairs are measured. `study` is a `plugins.StudyContext`: `num_slices`, `pixel_spacing_mm`,
`slice_thickness_mm`, `segmented_slices`, `pairs` (every `PairContext` above), `image(index)` and
`labels(index)` (decoded on demand). Returns `MetricValue`s like the pair hook.

### `bictool_tissue_methods()` - WM/GM methods

Called once when the plug-in is loaded. Returns a list of `plugins.TissueMethod`:

| Field | Content |
|---|---|
| `id` | Unique within the plug-in, no "." - selected as `"<plug-in>.<id>"` (`api.TissueSettings(method="otsu-wmgm.otsu")`). |
| `label` | Shown in the program's Brain Tissue > Method drop-down. |
| `classify` | `classify(slice_hu, brain_mask, settings, context)` returning `{plugins.WHITE_MATTER: mask, plugins.GRAY_MATTER: mask, plugins.CSF: mask}`: bool arrays of the slice's shape (anything outside `brain_mask` is dropped). `context` is a `plugins.TissueContext` (slice index, pixel spacing, CNN masks or None). |
| `settings` | A frozen dataclass instance with the defaults. The program builds its settings dialog from the fields (a number box per `int`/`float`, a tick box per `bool`); the API passes `api.TissueSettings(plugin=...)`. |

The method only decides where the falx pair's ROI2 goes (white matter next to the falx), as the built-in methods
do. It runs for every slice with automatic ROIs, possibly on a background thread.

## Rules

- **Read-only input.** Images and masks are read-only; writing to them raises an error (copy first).
- **Isolation.** If a hook raises, or returns something invalid, the error is recorded with the plug-in's values
  (`PluginValues.error`), shown in the program and exported; the study goes on. A failing WM/GM method stops the
  ROI placement with a clear message.
- **Threads.** Hooks may run on a background thread, one call at a time per study. Keep plug-ins free of shared
  mutable state, or protect it.
- **Speed.** Pair hooks run for every measured pair (about 50 per study): keep them to milliseconds.
- **No verdict.** Plug-in values are shown and exported, never fed into the study verdict.
- **Trust.** Plug-ins run inside BICTool with full access to the images and the computer. Only install or copy in
  plug-ins from people you trust. BICTool never downloads plug-ins.
- **Data protection.** A plug-in that writes files or uses the network is responsible for doing so lawfully
  (patient data, GDPR/BDSG).

## Where plug-in values appear

- **Program:** the Results page (per slice and on the Study summary), and the Excel/CSV exports.
- **API:** `Measurement.plugin_values` and `StudyResult.plugin_values` (`PluginValues(plugin, values, error)`),
  `StudyResult.plugins` (the active plug-ins and their versions); in the JSON result under the same names.

## Testing a plug-in

```python
from bictool import api

print(api.load_plugins(["path/to/my/plugins"], installed=False))   # status and message of each
result = api.analyse_study("D:/studies/case1")
for m in result.measurements:
    print(m.slice.index, m.pair, m.plugin_values)
```

## Versions

`requires_api` is checked against `bictool.api.API_VERSION`. The API is provisional (0.x): hooks and contexts may
still change between minor versions - use a range like `">=0.2,<0.3"` to be safe, or `">=0.2,<1"` to accept
compatible changes. Changes are listed in [api.md](api.md#changes).
