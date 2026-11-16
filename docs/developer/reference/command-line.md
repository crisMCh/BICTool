# Reference: Command line

## `python -m bictool`

```text
python -m bictool                                 start the program
python -m bictool series FOLDER                   list the series of a study folder
python -m bictool analyse FOLDER [options]        analyse one series, write the result as JSON
```

*Since 0.1.* Runs the [Python API](api-functions.md) without writing code; uses the plug-ins in the plug-in
folders and installed packages.

### `series`

Prints one line per series: `index: label`, best first (the program's choice is `0`).

### `analyse`

| Option | Default | Effect |
|---|---|---|
| `--series N` | `0` | Which series, as listed by `series`. |
| `--label TEXT` | `"study - <series description>"` | Name of the study in the result. |
| `--out FILE` | standard output | JSON file to write. |
| `--curves` | off | Include the MTF/NPS curves. |
| `--uids` | off | Include the DICOM UIDs (they link the result to the patient). |
| `--weights FILE` | configured weights | Anatomy CNN weights. |
| `--cache FILE` | none | Segmentation cache: read if it exists, else written. |

Progress goes to standard error; at the end a summary line: `case 1: acceptable (52 measurement(s), 1 failed)`.

| Exit code | Meaning |
|---|---|
| `0` | Done. |
| `2` | The study could not be analysed ([`BictoolError`](errors.md)); the reason is printed. |

```bash
python -m bictool analyse D:/studies/case1 --label "case 1" --out case1.json
```

## Batch tool

```text
python -m bictool.application.batch_qa DATASET [DATASET ...] --out FOLDER [options]
```

Analyses every study of one or more datasets. A dataset is a folder with one subfolder per study (typically one
dataset per scanner). The batch tool is meant for research on de-identified data: its outputs name the study
folders.

| Option | Default | Effect |
|---|---|---|
| `--out FOLDER` | required | Output folder: `dataset_qa.xlsx`, `dataset_qa_summary.csv`, `dataset_qa_slices.csv`, `dataset_qa_plugins.csv` (with plug-ins), the segmentation cache in `cache/`. |
| `--weights FILE` | configured weights | Anatomy CNN weights. |
| `--exclude TEXT` | none | Skip studies whose folder name contains `TEXT` (repeatable). |
| `--recipe NAME` | `physical` | QA recipe: `physical` or `octave`. |
| `--tissue NAME` | `adaptive` | WM/GM method: `adaptive` or `anchored`. |
| `--plugin-export ID` | none | Also write each study in a plug-in file format (`"<plug-in>.<id>"`) into `OUT/plugin_exports/<dataset> - <study><extension>` (repeatable). *Since 0.3.* |

A study that fails is recorded with its error; the others go on.

## Portable program

```text
BICTool.exe                      start the program
BICTool.exe --self-test FOLDER   analyse one study and report PASSED/FAILED in the log
```

On Windows, run the self-test with `start /wait "" BICTool.exe --self-test FOLDER`; the result is in
`%LOCALAPPDATA%\BICTool\logs\bictool.log`.
