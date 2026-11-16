# How to add a WM/GM method

A WM/GM method separates white matter, grey matter and CSF inside the brain mask of one slice. BICTool uses the
white matter to place the falx pair's ROI2. A plug-in's method appears in the program's **Brain Tissue >
Method** drop-down, with a settings dialog built from its settings.

## The plug-in

```python
from dataclasses import dataclass

import numpy as np

from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(name="level-split", version="1.0", requires_api=">=0.3,<1",
                                    description="White matter below a fixed HU level")


@dataclass(frozen=True)
class LevelSettings:
    level_hu: float = 30.0          # white matter below, grey matter above
    csf_below_hu: float = 15.0


def classify(slice_hu, brain_mask, settings, context):
    csf = brain_mask & (slice_hu < settings.csf_below_hu)
    white = brain_mask & ~csf & (slice_hu < settings.level_hu)
    grey = brain_mask & ~csf & ~white
    return {plugins.WHITE_MATTER: white, plugins.GRAY_MATTER: grey, plugins.CSF: csf}


@plugins.hookimpl
def bictool_tissue_methods():
    return [plugins.TissueMethod("level", "Level split", classify, LevelSettings())]
```

## Step by step

1. **`classify(slice_hu, brain_mask, settings, context)`** gets one slice in HU, BICTool's brain mask, the
   settings and a [`TissueContext`](../reference/plugin-types.md#tissuecontext) (slice index, pixel spacing,
   the CNN masks or `None`). It returns a dict with the three masks; a missing one counts as empty, anything
   outside `brain_mask` is dropped.
2. **Settings** are a frozen dataclass; its instance in `TissueMethod` holds the defaults. The program builds
   the settings dialog from the fields: a number box for each `int` or `float`, a tick box for each `bool`.
3. **Select it** in the program (Brain Tissue > Method, shown as *Level split (plug-in)*) or in the API:
   `api.TissueSettings(method="level-split.level")` - the id is `"<plug-in name>.<method id>"`.

## Speed

The program runs the method on its main thread whenever a new slice is shown. It must take milliseconds: above
half a second per slice, the program shows a warning under the drop-down, because the window waits for it.

## When it fails

If `classify` raises or returns masks of the wrong shape, ROI placement stops with a clear message naming the
method - choose another method or fix the plug-in.

## See also

* [`bictool_tissue_methods`](../reference/hooks.md#bictool_tissue_methods), [`TissueMethod`](../reference/plugin-types.md#tissuemethod)
* Complete example: `examples/plugins/otsu_wmgm.py`
* [Choose the QA recipe and the WM/GM method](choose-settings.md)
