# Data protection

BICTool processes images of patients. The API and the plug-ins are designed so that results can be shared for
quality assurance without identifying the patient; what you do with images and results beyond that is your
responsibility (GDPR, BDSG and your institution's rules).

## What BICTool reads and writes

| | |
|---|---|
| Reads | The DICOM files of the study folder you give it; the CNN weights. |
| Writes (API) | Nothing, except the segmentation cache you ask for (`cache_file`) and files you write yourself (`to_json`, `write_export`). |
| Writes (program) | The exports you ask for; its settings; a log file without patient identifiers. |
| Network | None. BICTool never sends data anywhere and never downloads plug-ins. |

## What a result contains

* **Never** the patient's name, ID or birth date.
* The **acquisition date** and the series' acquisition parameters (kV, mA, CTDIvol, ...), as in the exports.
* The **DICOM UIDs** (Study, Series, SOP Instance) only when asked for: `to_json(include_dicom_uids=True)`,
  `python -m bictool analyse --uids`. They are not names, but the PACS links them to the patient.
* The **label** you give; without one, `"study - <series description>"` - never the folder name, which often
  carries the scan date and time.
* Messages never repeat patient details from the DICOM files (e.g. "Several studies under ..." does not name
  the patients).

## What is derived from patient data

Segmentation caches (`cache_file`), results, exports and the batch tool's outputs (which name the study folders)
are derived from patient data. Keep them where patient-related data belongs - not in version control, not in
shared folders without permission.

## Plug-ins

A plug-in runs inside BICTool with BICTool's access to the images and the computer. BICTool hands plug-ins the
same data as its own steps (images, masks, the result without patient identifiers), but cannot control what a
plug-in does with it. A plug-in that writes files or uses the network is responsible for doing so lawfully.
Install only plug-ins from people you trust.
