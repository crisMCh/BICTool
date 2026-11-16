# Tutorial: Your first plug-in

In this tutorial you write a plug-in that adds one value to every measured ROI pair - the pixel standard
deviation of ROI2 - and see it in the program, in the export and in the Python API. It takes about 20 minutes.

**You need:** BICTool (the portable program or the source code) and a study you can run QA on. No Python
knowledge beyond functions and lists is needed.

## 1. Find the plug-in folder

BICTool looks for plug-ins in a folder:

| You run | Plug-in folder |
|---|---|
| the portable program (`BICTool.exe`) | `plugins` next to `BICTool.exe` |
| BICTool from the source code | `.bictool/plugins` in your home folder (create it) |

Settings > Plug-ins in the program shows the folder.

## 2. Write the plug-in

Create the file `roi2_sd.py` in the plug-in folder:

```python
from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(
    name="roi2-sd",
    version="1.0",
    requires_api=">=0.3,<1",
    description="Pixel SD of ROI2",
)


@plugins.hookimpl
def bictool_pair_metrics(pair):
    sd = float(pair.roi2_pixels.std(ddof=1))
    return [plugins.MetricValue("roi2_sd", sd, "HU", "Pixel SD of ROI2")]
```

What the parts do:

* `BICTOOL_PLUGIN` describes the plug-in. `name` must be unique and must not contain a dot; `requires_api`
  says which versions of BICTool's API it works with (here: 0.3 and later 0.x versions).
* `@plugins.hookimpl` marks a function that BICTool calls. Its name says when: `bictool_pair_metrics` is
  called once for every measured ROI pair.
* `pair` holds the pair: the slice image in HU, both ROIs, BICTool's own values and more
  ([`PairContext`](../reference/plugin-types.md#paircontext)). `pair.roi2_pixels` are the pixels inside ROI2.
* The function returns a list of `MetricValue`s: a name, a number, a unit and a description.

## 3. Check that BICTool loads it

Start BICTool (restart it if it was running) and open **Settings > Plug-ins**. The list shows `roi2-sd` with
the status **Active**. If it shows **Error**, the message says why - most often a typing error; see
[Troubleshooting](../how-to/troubleshooting.md).

## 4. See the value

Open a study, segment it, place and confirm the ROIs, and run QA as usual. Then:

* the **Per slice** tab of the Results page shows `roi2-sd: roi2_sd 4.21 HU` in the slice note;
* the study export (Excel) has a sheet **Plug-ins** with the value of every ROI pair, and the CSV export a file
  `... - plug-ins.csv`.

The study verdict is the same as without the plug-in: plug-in values never change it.

## 5. Use it from Python

The API loads the same plug-ins:

```python
from bictool import api

print(api.loaded_plugins())            # roi2-sd ... status 'active'
result = api.analyse_study("D:/studies/case1", label="case 1")
for m in result.measurements[:3]:
    print(m.slice.index, m.pair, m.plugin_values)
```

```text
8 falx (PluginValues(plugin='roi2-sd', values=(MetricValue(name='roi2_sd', value=4.21, unit='HU', ...),), ...),)
```

## 6. Make it fail on purpose

Change the return line to `return [plugins.MetricValue("roi2_sd", "a lot", "HU")]` and run QA again. The study
still finishes; the slice note shows `roi2-sd: problem: roi2_sd: value 'a lot' is not a number`. BICTool checks
everything a plug-in returns and keeps going when a plug-in fails. Undo the change.

## What you have learnt

* A plug-in is one Python file with a `BICTOOL_PLUGIN` description and hook functions marked `@hookimpl`.
* BICTool finds it in the plug-in folder, shows its status, calls it, checks what it returns and shows the
  values next to its own - without changing the verdict.

## Next steps

* Measure in ROIs of your own and show them on the image: [Add a metric with ROIs of its own](../how-to/metric-with-own-roi.md)
* All hooks: [Hooks](../reference/hooks.md)
* Give the plug-in to colleagues: [Install and share a plug-in](../how-to/install-and-share-a-plugin.md)
* Test it automatically: [Test a plug-in](../how-to/test-a-plugin.md)
