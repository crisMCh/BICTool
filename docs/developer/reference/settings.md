# Reference: Settings

All settings are frozen dataclasses, checked when created: an invalid value raises `ValueError`. Change a copy
with `dataclasses.replace`. The defaults are the program's.

## `Settings`

```text
api.Settings(recipe="physical", tissue=TissueSettings())
```

*Since 0.1*

Everything a QA run can be tuned with.

| Field | Type | Default | Description |
|---|---|---|---|
| `recipe` | `str` | `"physical"` | The QA recipe: `api.RECIPE_PHYSICAL` (`"physical"`, gives the head-CT verdict) or `api.RECIPE_OCTAVE` (`"octave"`, the legacy recipe, no verdict). Another value raises `ValueError`. |
| `tissue` | [`TissueSettings`](#tissuesettings) | defaults | The WM/GM method and its parameters. |

| Method | Returns |
|---|---|
| `to_dict()` | `{"recipe": ..., "tissue_method": ..., "tissue_parameters": {...}}` - the chosen method's parameters only (as in a JSON result). |

## `TissueSettings`

```text
api.TissueSettings(method="adaptive", threshold=..., fcm=..., adaptive=..., plugin=None)
```

*Since 0.1; `plugin` since 0.2*

The WM/GM method and every method's parameters; only the chosen method's are used. The method decides where
the falx pair's ROI2 goes (white matter next to the falx).

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"adaptive"` | `"adaptive"`, `"fcm"`, `"threshold"`, `"anchored"` (prototype; uses `adaptive` where it has no model) or a plug-in's `"<plug-in>.<id>"`. Another value raises `ValueError`. |
| `threshold` | [`BrainTissueSettings`](#braintissuesettings) | defaults | For `"threshold"`; its `air_hu_cutoff` and `bone_hu_cutoff` also define the HU skull-strip used where a slice has no segmentation. |
| `fcm` | [`FcmSettings`](#fcmsettings) | defaults | For `"fcm"`. |
| `adaptive` | [`AdaptiveSettings`](#adaptivesettings) | defaults | For `"adaptive"` and `"anchored"`. |
| `plugin` | object | `None` | For a plug-in's method: an instance of its settings dataclass, or a dict of the fields that differ from its defaults (a field it does not have raises `ValueError` when used); `None`: its defaults. |

| Method | Returns |
|---|---|
| `parameters()` | The chosen method's parameters. |

Method ids and names: `api.METHOD_ADAPTIVE`, `api.METHOD_FCM`, `api.METHOD_THRESHOLD`, `api.METHOD_ANCHORED`;
`api.METHOD_LABELS` maps them to the names the program shows ("Adaptive Thresholding", "Fuzzy C-Means", "HU
Thresholding", "Anchored (prototype)").

## `AdaptiveSettings`

*Since 0.1* - Adaptive Thresholding (the default method).

| Field | Type | Default | Limits | Description |
|---|---|---|---|---|
| `block_size` | `int` | `31` | >= 3 | Neighbourhood size (pixels) of the local threshold. |
| `plausible_low` | `float` | `15.0` | < `plausible_high` | Lower end of the plausible brain-tissue HU range (used without a study range). |
| `plausible_high` | `float` | `50.0` | | Upper end. |
| `min_component_size` | `int` | `20` | >= 0 | Smaller regions (pixels) are dropped. |
| `smoothing_mm` | `float` | `1.0` | >= 0 | Gaussian smoothing before classifying (sigma, mm); 0 = none. |
| `use_study_range` | `bool` | `True` | | Take the plausible range from the study's own grey/white references (basal ganglia) when available. |

## `FcmSettings`

*Since 0.1* - Fuzzy C-Means.

| Field | Type | Default | Limits | Description |
|---|---|---|---|---|
| `fuzziness` | `float` | `2.0` | > 1 | Fuzziness exponent m. |
| `max_iter` | `int` | `50` | >= 1 | Maximum iterations. |
| `spatial_weight` | `float` | `0.3` | >= 0 | Weight of the neighbourhood term. |
| `min_component_size` | `int` | `20` | >= 0 | Smaller regions (pixels) are dropped. |

## `BrainTissueSettings`

*Since 0.1* - HU Thresholding, and the HU skull-strip.

| Field | Type | Default | Limits | Description |
|---|---|---|---|---|
| `air_hu_cutoff` | `float` | `-500.0` | < `bone_hu_cutoff` | Skull-strip: below = air. |
| `bone_hu_cutoff` | `float` | `100.0` | | Skull-strip: above = bone. |
| `csf_low`, `csf_high` | `float` | `0.0`, `20.0` | low < high | CSF HU range. |
| `white_matter_low`, `white_matter_high` | `float` | `20.0`, `32.0` | low < high | White-matter HU range. |
| `gray_matter_low`, `gray_matter_high` | `float` | `32.0`, `45.0` | low < high | Grey-matter HU range. |
| `smoothing_size` | `int` | `3` | >= 1 | Median filter size (pixels). |
| `min_component_size` | `int` | `20` | >= 0 | Smaller regions (pixels) are dropped. |

The algorithms behind the methods are described in the *Technical Reference* (WM/GM methods).

**See also** [Choose the QA recipe and the WM/GM method](../how-to/choose-settings.md)
