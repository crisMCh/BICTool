# How to test a plug-in

## Quick check: does it load?

```python
from bictool import api

for plugin in api.load_plugins(["D:/my-plugins"], installed=False):
    print(plugin.name, plugin.status, plugin.message)
```

`load_plugins` loads only the plug-ins in the given folders (with `installed=False`, no installed packages) and
uses them for every later API call. `status` is `active`, `disabled`, `incompatible` or `error`; `message` says
why, and also names `bictool_*` functions BICTool will not call (a typo in a hook name, or a missing
`@plugins.hookimpl`).

## Run it on a real study

```python
result = api.analyse_study("D:/studies/case1", label="case 1")
print(result.plugins)                                  # {"my-plugin": "1.0"}
for m in result.measurements:
    print(m.slice.index, m.pair, m.plugin_values)      # pair values
print(result.slice_plugin_values)                      # slice values
print(result.plugin_values, result.plugin_hints)       # series values, artefact hints
```

Look at `PluginValues.error`: it holds what went wrong in your plug-in, or what BICTool dropped from its
return value.

## Automated tests with synthetic data (pytest)

Call your hook functions directly with a context you build yourself - no study, no CNN needed:

```python
import numpy as np
from types import MappingProxyType

from bictool.api import plugins

import my_plugin  # the plug-in module


def test_roi2_sd():
    image = np.full((64, 64), 30.0)
    image[40:50, 40:50] += np.tile([0.0, 2.0], 50).reshape(10, 10)
    pair = plugins.PairContext(
        slice_index=0, pair="falx", image=image, pixel_spacing_mm=0.5,
        roi1=plugins.RoiRect(10, 10, 20, 20), roi2=plugins.RoiRect(40, 40, 50, 50),
        labels=None, bictool_values=MappingProxyType({}),
    )

    (value,) = my_plugin.bictool_pair_metrics(pair)

    assert value.name == "roi2_sd" and value.unit == "HU"
    assert value.value == np.std(image[40:50, 40:50], ddof=1)
```

For the slice and study hooks, build a `SliceContext` or `StudyContext` the same way; `brain_mask` and `tissue`
are functions, e.g. `brain_mask=lambda: np.ones((64, 64), bool)`.

To test that BICTool loads and calls the plug-in as a whole, run `api.load_plugins([folder], installed=False)`
in the test and analyse a small synthetic or anonymised study.

## Test it in the portable program

Copy the file into the `plugins` folder next to `BICTool.exe` and restart. Settings > Plug-ins shows the status;
the log file `%LOCALAPPDATA%\BICTool\logs\bictool.log` has one line per plug-in and the details of any error.
The portable program contains only Python's standard library, NumPy, SciPy and BICTool - test there if your
users run it.

## See also

* [Troubleshooting and FAQ](troubleshooting.md)
* [`load_plugins`](../reference/api-functions.md#load_plugins), [Plug-in types](../reference/plugin-types.md)
