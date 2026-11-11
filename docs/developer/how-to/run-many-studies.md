# How to analyse many studies

## From Python

Load the CNN once and reuse it; keep a segmentation cache per study so that a second run skips the CNN.

```python
from pathlib import Path
from bictool import api

model = api.load_anatomy_model()                       # loads TensorFlow and the weights once
cache_dir = Path("D:/qa_cache")                        # derived from patient data - not into version control
cache_dir.mkdir(exist_ok=True)

for number, folder in enumerate(sorted(Path("D:/studies").iterdir()), start=1):
    label = f"study {number}"                          # not the folder name if it identifies the patient
    try:
        result = api.analyse_study(folder, label=label, model=model, cache_file=cache_dir / f"{number}.npz")
    except api.BictoolError as error:                  # one bad study must not stop the others
        print(label, "skipped:", error)
        continue
    (Path("D:/qa_results") / f"{number}.json").write_text(result.to_json(), encoding="utf-8")
```

* `cache_file` is read if it exists (it must belong to the same series, else
  [`InvalidSegmentation`](../reference/errors.md#invalidsegmentation)) and written otherwise.
* Catch [`api.BictoolError`](../reference/errors.md) for problems with a study (no CT series, several studies
  in one folder, ...). Other exceptions are programming errors and should stop the run.
* To stop a long run cleanly, pass `cancel=lambda: stop_requested` - the step stops between two slices and
  raises [`Cancelled`](../reference/errors.md#cancelled); finished slices are kept in the cache.

## With the batch tool

For datasets organised as one folder per scanner with one folder per study, the batch tool writes one Excel
workbook and CSV tables for all studies:

```bash
python -m bictool.application.batch_qa "D:/data/Scanner A" "D:/data/Scanner B" --out D:/qa_out
```

See [Command line](../reference/command-line.md#batch-tool) for its options. The batch tool keeps the folder
names in its outputs - keep those outputs where patient-related data belongs.

## See also

* [`analyse_study`](../reference/api-functions.md#analyse_study), [`load_anatomy_model`](../reference/api-functions.md#load_anatomy_model)
* [Save results and read them back](save-and-read-results.md)
