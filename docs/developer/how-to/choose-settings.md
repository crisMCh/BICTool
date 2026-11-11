# How to choose the QA recipe and the WM/GM method

Without settings, the API uses the program's defaults: the physical QA recipe and Adaptive Thresholding for
the WM/GM separation. Pass `api.Settings` to change them.

## Use the legacy QA recipe

```python
from bictool import api

settings = api.Settings(recipe=api.RECIPE_OCTAVE)
result = api.analyse_study("D:/studies/case1", settings=settings)
```

The legacy (Octave) recipe gives no head-CT verdict (`result.verdict` is `None`); its numbers are on a
different scale than the physical recipe's - never mix them.

## Use another WM/GM method

The WM/GM method decides where the falx pair's ROI2 goes (white matter next to the falx).

```python
settings = api.Settings(tissue=api.TissueSettings(method=api.METHOD_FCM))
```

| Id | Method |
|---|---|
| `api.METHOD_ADAPTIVE` (`"adaptive"`) | Adaptive Thresholding - the default |
| `api.METHOD_FCM` (`"fcm"`) | Fuzzy C-Means |
| `api.METHOD_THRESHOLD` (`"threshold"`) | HU Thresholding |
| `api.METHOD_ANCHORED` (`"anchored"`) | Anchored (prototype) |
| `"<plug-in>.<id>"` | a plug-in's method, e.g. `"otsu-wmgm.otsu"` |

`api.METHOD_LABELS` gives the names the program shows.

## Change one parameter

Settings are frozen: change a copy with `dataclasses.replace`.

```python
import dataclasses

tissue = api.TissueSettings()
tissue = dataclasses.replace(tissue, adaptive=dataclasses.replace(tissue.adaptive, block_size=41))
settings = api.Settings(tissue=tissue)
```

Invalid values raise `ValueError` when the settings are created (e.g. `block_size` below 3).

## Use a plug-in's method with its own settings

```python
settings = api.Settings(tissue=api.TissueSettings(method="my-wmgm.level", plugin={"level": 25.0}))
```

`plugin` takes a dict of the fields that differ from the plug-in's defaults (BICTool fills them into the
plug-in's settings dataclass; a field it does not have raises `ValueError`), or an instance of that dataclass.
Without `plugin` the plug-in's defaults are used. A result read back from JSON holds the plug-in's settings as
such a dict, so it can be run again with the same settings.

## See also

* [Settings reference](../reference/settings.md) - every parameter, its default and its limits
* [Add a WM/GM method](wmgm-method.md)
