# How to add a metric with ROIs of its own

Use this when your metric needs ROIs other than BICTool's falx and septum pairs - a different place, shape or
size. BICTool's own ROIs stay as they are; your plug-in places its ROIs, measures in them, and shows them on the
image.

## The plug-in

```python
import numpy as np
from scipy import ndimage

from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(name="wm-disc", version="1.0", requires_api=">=0.3,<1",
                                    description="Mean and SD in a white-matter disc")

RADIUS_MM = 5.0


@plugins.hookimpl
def bictool_slice_metrics(ct_slice):
    white = ct_slice.tissue()[plugins.WHITE_MATTER] & ct_slice.brain_mask()
    if not white.any():
        return [plugins.MetricValue("wm_sd", float("nan"), "HU")]
    # The white-matter pixel farthest from any other tissue is the centre of the disc.
    depth = ndimage.distance_transform_edt(white) * ct_slice.pixel_spacing_mm
    row, col = np.unravel_index(int(np.argmax(depth)), depth.shape)
    rows, cols = np.ogrid[: white.shape[0], : white.shape[1]]
    disc = (rows - row) ** 2 + (cols - col) ** 2 <= (RADIUS_MM / ct_slice.pixel_spacing_mm) ** 2
    disc &= white
    pixels = ct_slice.image[disc]
    return [
        plugins.MetricValue("wm_mean", float(pixels.mean()), "HU", "Mean in the white-matter disc"),
        plugins.MetricValue("wm_sd", float(pixels.std(ddof=1)), "HU", "Pixel SD in the white-matter disc"),
        plugins.RoiMarker("white-matter disc", disc),
    ]
```

## Step by step

1. **Choose the hook.** [`bictool_slice_metrics`](../reference/hooks.md#bictool_slice_metrics) is called once
   per measured slice. For a value over the whole series use
   [`bictool_study_metrics`](../reference/hooks.md#bictool_study_metrics) (it can look at every slice).
2. **Use BICTool's masks.** `ct_slice.brain_mask()` is BICTool's brain mask, `ct_slice.tissue()` its
   white-matter, grey-matter and CSF masks, `ct_slice.labels` the anatomy CNN's 16 structures. They are
   computed on first use; the WM/GM masks come from the method chosen in the program or in `api.Settings`
   (`ct_slice.tissue_method` says which). BICTool records that method next to your values.
3. **Return values and markers.** A `RoiMarker` is a name and a bool mask of the slice's shape - any shape. In
   the program, the tick box **Plug-in ROIs** (Run QA box) shows it in yellow; the API keeps it in
   `PluginValues.markers` and `StudyResult.plugin_markers(slice)`.
4. **Return NaN when the value cannot be computed** - not `None`, not 0.

## Where the values appear

| Where | What |
|---|---|
| Program, Per slice | slice note: `wm-disc: wm_mean 28.4 HU, wm_sd 3.9 HU` |
| Program, image | the disc, when **Plug-in ROIs** is ticked |
| Export | sheet/CSV Plug-ins, ROI pair column `slice`, and a row "plug-in input / WM/GM method" |
| API | `result.slice_plugin_values[slice]`, `result.plugin_markers(slice)` |

## Notes

* Images and masks are read-only; copy them before changing (`mask = ct_slice.brain_mask().copy()`).
* Keep the hook fast (it runs for every measured slice, about 30 per study).
* A complete example with a square ROI, a series value and an artefact hint is
  `examples/plugins/wm_box.py`.

## See also

* [`SliceContext`](../reference/plugin-types.md#slicecontext), [`RoiMarker`](../reference/plugin-types.md#roimarker)
* [Test a plug-in](test-a-plugin.md)
