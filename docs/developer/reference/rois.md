# Reference: ROIs

Every slice can carry two ROI pairs, measured side by side:

| Pair | ROI1 (on the structure) | ROI2 (homogeneous tissue next to it) |
|---|---|---|
| `"falx"` (`api.FALX`) | on the falx (dural folds) | white matter, found by the WM/GM method |
| `"septum"` (`api.SEPTUM`) | on the septum pellucidum | ventricle |

Each slice therefore has four **slots**: `falx.roi1`, `falx.roi2`, `septum.roi1`, `septum.roi2` - built with
[`roi_key`](#roi_key). Each slot holds an automatic ROI, optionally a manual ROI that wins over it, and
optionally a "deleted" mark.

## `RoiRect`

```text
api.RoiRect(x0, y0, x1, y1)
```

*Since 0.1 · frozen*

An ROI rectangle in image-pixel coordinates. `x` counts columns, `y` rows, from the top-left corner of the
image; `x1, y1` lie just outside the ROI, so the pixels are `image[y0:y1, x0:x1]`.

| Field | Type | Description |
|---|---|---|
| `x0`, `y0` | `int` | Top-left corner (inclusive). |
| `x1`, `y1` | `int` | Bottom-right corner (exclusive). |
| `width`, `height` | `int` (properties) | `x1 - x0`, `y1 - y0`. |

To be measured, an ROI must have whole-pixel coordinates, lie inside the image, have `x0 < x1` and `y0 < y1`,
and be **square** (the noise power spectrum needs it). `Study.measure` checks this and raises
[`InvalidRois`](errors.md#invalidrois).

## `roi_key`

```text
api.roi_key(pair, slot) -> str
```

*Since 0.1*

The slot name: `api.roi_key(api.FALX, "roi1") == "falx.roi1"`. `pair` is one of `api.PAIRS`, `slot` one of
`api.SLOTS` (`"roi1"`, `"roi2"`).

## `RoiPairStore`

```text
api.RoiPairStore(series_key=None)
```

*Since 0.1*

A series' ROIs per slice and slot, and what the automatic placement found. [`Study.place_rois`](api-functions.md#studyplace_rois)
returns one; you can also build one by hand (then `series_key` is `None`). Methods that take a slot name raise
`ValueError` for an unknown name.

| Attribute | Type | Description |
|---|---|---|
| `series_key` | `str` or `None` | The series the automatic ROIs were placed on; `Study.measure` refuses a store of another series. |
| `skipped` | `dict[str, tuple[int, ...]]` | Per pair: slices where the structure was found but the brain is too small (below 20 % of the largest brain area) for automatic ROIs. |
| `unplaced` | `set[tuple[int, str]]` | `(slice, pair)` whose ROI1 was placed but no ROI2 found. |

Reading:

| Method | Returns |
|---|---|
| `effective(slice_index)` | `dict[str, RoiRect]`: the slice's ROIs per slot - manual over automatic, deleted slots left out. |
| `auto(slice_index, key)` | The automatic ROI of a slot, or `None`. |
| `auto_region(slice_index, key)` | Where an automatic ROI2 lies: `"ventricle"`, `"white_matter"`, `"fallback"`, or `None`. |
| `is_manual(slice_index, key)` | `True` if the slot holds a manual ROI. |
| `complete_pairs(slice_index)` | Pairs with both ROI1 and ROI2 on the slice - the ones that can be measured. |
| `slices()` | Slice indices with any ROI, sorted. |
| `ready_jobs()` | Every `(slice, pair)` with a complete pair, in slice order - one measurement each. |
| `has_edits(slice_index)` | `True` if the slice has a manual ROI or a deletion. |
| `is_edited()` | `True` if any slice has. |

Changing:

| Method | Effect |
|---|---|
| `set_manual(slice_index, key, rect)` | A drawn or moved ROI; wins over the automatic one and undoes a deletion of that slot. |
| `delete(slice_index, key)` | Removes the slot's ROI from the slice (manual and automatic). |
| `restore_slice(slice_index)` | Back to the automatic ROIs on the slice: manual ROIs and deletions dropped. |
| `set_auto(slice_index, key, rect, region=None)` | Sets the automatic ROI of a slot (`None` removes it); `region` as in `auto_region`. |
| `clear_auto(pair=None)` | Drops the automatic ROIs of `pair` (or of both) on every slice. |
| `clear()` | Drops everything. |

**Example**

```python
rois = study.place_rois()
key = api.roi_key(api.FALX, "roi1")
rois.set_manual(12, key, api.RoiRect(240, 150, 268, 178))
assert rois.is_manual(12, key) and rois.is_edited()
```

**See also** [Check or change the ROIs before measuring](../how-to/check-rois-before-measuring.md)
