# Tutorial: Your first analysis

In this tutorial you analyse one head-CT study with the Python API: you list its series, run BICTool's
assessment, look at the verdict and the measurements, and save the result as JSON. It takes about 15 minutes.

**You need:** BICTool's source code with its Python environment, the anatomy CNN weights, and one head-CT study
as a folder of DICOM files (any study you are allowed to process - the result contains no patient name or ID).

## 1. Prepare the environment

In the BICTool folder, install BICTool with its DICOM and machine-learning parts (once):

```bash
pip install -e ".[dicom,ml]"
```

Tell BICTool where the CNN weights are, either with the environment variable `BICTOOL_HEADCT_WEIGHTS` or in
`~/.bictool/config.json` (details in `reference/README.md` of the repository):

```json
{"segmentation": {"anatomy_weights_path": "D:/models/weights_mayoiNPH.hdf5"}}
```

Check that the API imports:

```python
from bictool import api
print(api.API_VERSION)          # 0.3.0
```

## 2. Look at the series of the study

A study folder often holds several series (scout, thin and thick slices, other kernels). List them:

```python
for choice in api.list_series("D:/studies/case1"):
    print(choice.index, choice.label)
```

```text
0 CT · 37 slice(s) · 512×512 · Hr40 · 4mm · axial
1 CT · 148 slice(s) · 512×512 · Hr40 · 1mm · axial
```

The first one (index 0) is the series the program would choose: CT, axial, closest to 4 mm.

## 3. Analyse the study

```python
result = api.analyse_study(
    "D:/studies/case1",
    label="case 1",
    progress=lambda stage, done, total: print(stage, done, total),
)
```

You see the three stages pass: `segmentation` (the CNN, about 1-2 s per slice), `rois` (the automatic falx and
septum ROI pairs) and `measurement` (MTF, noise power spectrum and the other values per ROI pair).

The `label` is how the study is named in the result. Leave out the folder name if it contains the scan date
or other details about the patient.

## 4. Read the verdict

```python
v = result.verdict
print(v.verdict, round(v.probability, 2), v.reasons)
```

```text
acceptable 0.83 ('nps_homo', 'mtf_area')
```

`verdict` is one of `"acceptable"`, `"not_acceptable"`, `"uncertain"` and `"not_assessable"`; `probability` is
the model's probability of "acceptable"; `reasons` names the features behind it, strongest first. The verdict
is `None` when there are fewer than 5 measured falx slices or when the legacy QA recipe was used.

## 5. Look at the measurements

Every ROI pair measured on a slice is one `Measurement`:

```python
for m in result.measurements[:3]:
    print(m.slice.index, m.pair, round(m.mtf_area, 3), round(m.nps_homo, 1), m.roi2_region)
```

```text
8 falx 0.452 41.3 white_matter
8 septum 0.398 38.7 ventricle
9 falx 0.461 39.9 white_matter
```

Slice indices are 0-based - the program shows slice 9 for index 8. The units are in `api.UNITS`:

```python
print(api.UNITS["mtf_area"], api.UNITS["nps_homo"])   # 1/mm HU^2*mm^2
```

Slices that could not be measured are in `result.failed`, each with a message.

## 6. Save the result

```python
with open("case1.json", "w", encoding="utf-8") as f:
    f.write(result.to_json())
```

The JSON file holds everything shown above, the units and the versions. The DICOM UIDs are left out: they
link the result to the patient in the PACS (add them with `to_json(include_dicom_uids=True)` if your
integration needs them).

Read it back later:

```python
with open("case1.json", encoding="utf-8") as f:
    again = api.StudyResult.from_json(f.read())
```

## What you have learnt

* `list_series` shows which series BICTool can analyse; `analyse_study` runs the whole assessment.
* The result is a `StudyResult`: a verdict, one `Measurement` per ROI pair and slice, failures, hints.
* Results are saved and read back as JSON.

## Next steps

* Check or move ROIs before measuring: [Check or change the ROIs before measuring](../how-to/check-rois-before-measuring.md)
* Analyse a whole folder of studies: [Analyse many studies](../how-to/run-many-studies.md)
* Every field of the result: [Result types](../reference/results.md)
* The same from the command line: [Command line](../reference/command-line.md)
