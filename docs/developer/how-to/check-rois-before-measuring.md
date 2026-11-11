# How to check or change the ROIs before measuring

`analyse_study` places and measures the ROIs in one go. To look at the ROIs first - or to move, draw or delete
some, as in the program - run the steps one by one with a [`Study`](../reference/api-functions.md#study).

```python
from bictool import api

study = api.open_study("D:/studies/case1", series=0, label="case 1")
study.segment()                                   # the anatomy CNN; 1-2 s per slice
rois = study.place_rois()                         # the automatic ROI pairs: an api.RoiPairStore
```

## Look at the ROIs

```python
print(rois.slices())                              # slices with any ROI, 0-based
print(rois.effective(12))                         # {"falx.roi1": RoiRect(...), "falx.roi2": ..., ...}
print(rois.auto_region(12, "falx.roi2"))          # "white_matter", "ventricle" or "fallback"
print(rois.skipped)                               # per pair: slices left out (too little brain)
print(rois.unplaced)                              # (slice, pair) whose ROI2 search found nothing
```

Each slice has four slots: `falx.roi1`, `falx.roi2`, `septum.roi1`, `septum.roi2`. Build the names with
`api.roi_key(api.FALX, "roi1")` rather than typing them.

## Change them

```python
roi1 = api.roi_key(api.FALX, "roi1")
rois.set_manual(12, roi1, api.RoiRect(x0=240, y0=150, x1=268, y1=178))   # wins over the automatic ROI
rois.delete(13, api.roi_key(api.SEPTUM, "roi2"))                         # no septum pair on slice 13
rois.restore_slice(14)                                                   # back to the automatic ROIs
```

An ROI must be **square** (the noise power spectrum needs it), in **whole pixels** and **inside the image**.
`x0, y0` is the top-left corner, `x1, y1` the corner just outside the ROI.

## Measure

```python
result = study.measure(rois, reviewed=True)
```

* `reviewed=True` records that a person checked the ROIs; `result.rois_edited` records whether any ROI was
  moved, drawn or deleted.
* As in the program, a **manual ROI1 gets a new automatic ROI2** next to it, unless its ROI2 is manual too;
  `rois` is updated.
* All ROIs are checked before anything is measured. A problem raises
  [`InvalidRois`](../reference/errors.md#invalidrois) with the slice and the reason.

## See also

* [`RoiPairStore` and `RoiRect`](../reference/rois.md)
* [`Study.measure`](../reference/api-functions.md#studymeasure)
