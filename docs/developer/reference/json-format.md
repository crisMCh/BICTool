# Reference: JSON format

`StudyResult.to_json()` / `to_dict()` write a result as plain JSON; `StudyResult.from_json()` / `from_dict()`
read it back. The layout is versioned by `schema_version` (currently `"1.2"`).

## Rules

| Topic | Rule |
|---|---|
| Keys | The field names of the [result types](results.md), nested the same way. |
| Missing values | `null` - both "not available" and "could not be computed" (NaN in Python). |
| Slice indices | 0-based integers. Where they are dictionary keys (`basal_ganglia`, `slice_plugin_values`, `metal_streak_share`, ...) they are strings, as JSON requires: `"12"`. |
| ROIs | `[x0, y0, x1, y1]`. |
| Settings | `{"recipe": ..., "tissue_method": ..., "tissue_parameters": {...}}` - only the chosen method's parameters. |
| Units | Under `"units"`, as `api.UNITS`. |
| Curves | `null` unless written with `include_curves=True` (and measured with `curves=True`). |
| DICOM UIDs | `study_instance_uid`, `series_instance_uid` and `sop_instance_uid` are left out unless written with `include_dicom_uids=True`: they link the result to the patient in the PACS. |
| Patient data | No patient name, ID or birth date - ever. The acquisition date is included. |
| Plug-in markers | Each marker: `{"plugin", "name", "slice_index", "box": [x0, y0, x1, y1], "mask", "description"}`; `mask` is base64 of the mask's bits inside `box`, row by row, packed 8 per byte (`numpy.packbits`). |

## Versions

* The **minor** number of `schema_version` grows when keys are added (1.1: plug-in values; 1.2: plug-in slice
  values, markers, artefact hints). Readers should ignore keys they do not know.
* The **major** number grows when a key is renamed or removed. `from_dict` reads results of the same major
  version and raises `ValueError` for another.
* A test pins the layout, so it cannot change unnoticed.

## Example

A shortened result (one measurement, no verdict - synthetic data):

```json
{
 "label": "study - Head 4.0 Hr40",
 "series": {"description": "Head 4.0 Hr40", "slices": 37, "slice_thickness_mm": 4.0, "pixel_spacing_mm": 0.45,
            "manufacturer_model": null, "reconstruction": "Hr40", "kvp": 120.0, "tube_current_ma": null,
            "ctdi_vol_mgy": null, "dlp_mgy_cm": null, "acquisition_date": "20260131"},
 "settings": {"recipe": "physical", "tissue_method": "adaptive",
              "tissue_parameters": {"block_size": 31, "plausible_low": 15.0, "plausible_high": 50.0,
                                    "min_component_size": 20, "smoothing_mm": 1.0, "use_study_range": true}},
 "rois_reviewed": false,
 "rois_edited": false,
 "measurements": [
  {"slice": {"index": 12, "location_mm": 48.0}, "pair": "septum",
   "roi1": [249, 206, 271, 228], "roi2": [226, 206, 248, 228], "roi1_manual": false, "roi2_manual": false,
   "roi2_region": "ventricle", "recipe": "physical",
   "mtf_area": 0.41, "nps_homo": 38.2, "nps_struct": 61.0, "var_struct": 70.3, "ratio_lf_hf": 21.4,
   "nps_f_av": 0.21, "nps_f_peak": null, "contrast_roi1": 0.19, "contrast_roi2": 0.31,
   "curves": null, "plugin_values": []}
 ],
 "failed": [],
 "skipped": {"falx": [], "septum": [3, 4]},
 "segmented_slices": 37,
 "verdict": null,
 "artefacts": {"metal_slices": [], "metal_max_hu": null, "metal_streak_slices": [], "metal_streak_share": {},
               "brainstem_band_slices": [], "brainstem_band_depth_hu": {}},
 "basal_ganglia": {},
 "segmentation_check": {"fragmentation": 0.0, "slices": 37, "worst_slices": [12], "lobe_islands": 0,
                        "unreliable": false},
 "plugins": {},
 "plugin_values": [],
 "slice_plugin_values": {},
 "plugin_hints": [],
 "api_version": "0.3.0",
 "schema_version": "1.2",
 "units": {"mtf_area": "1/mm", "nps_homo": "HU^2*mm^2", "nps_struct": "HU^2*mm^2", "var_struct": "HU^2",
           "ratio_lf_hf": "1", "nps_f_av": "1/mm", "nps_f_peak": "1/mm", "contrast_roi1": "1",
           "contrast_roi2": "1", "probability": "1", "location_mm": "mm", "frequency": "1/mm", "hu": "HU"}
}
```

A plug-in's values in a measurement, slice or study look like this:

```json
{"plugin": "roi2-sd", "values": [{"name": "roi2_sd", "value": 4.21, "unit": "HU", "description": "Pixel SD of ROI2"}],
 "error": null, "markers": []}
```

## See also

* [Result types](results.md), [Save results and read them back](../how-to/save-and-read-results.md)
