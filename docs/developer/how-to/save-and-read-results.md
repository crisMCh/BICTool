# How to save results and read them back

## As JSON

```python
from bictool import api

text = result.to_json()                                  # or result.to_dict() for plain Python values
with open("case1.json", "w", encoding="utf-8") as f:
    f.write(text)

with open("case1.json", encoding="utf-8") as f:
    again = api.StudyResult.from_json(f.read())
```

Options of `to_json` / `to_dict`:

| Option | Default | Effect |
|---|---|---|
| `include_curves` | `False` | Keep the MTF/NPS curves of every measurement (the study must have been measured with `curves=True`). |
| `include_dicom_uids` | `False` | Keep the Study, Series and SOP Instance UIDs. They link the result to the patient in the PACS - add them only when your integration needs them. |
| `indent` (`to_json` only) | `2` | JSON indentation; `None` for one line. |

`from_json` / `from_dict` read results of the same schema major version (1.x). A result read back has no
internal objects: BICTool's own Excel/CSV export (`to_volume_qa`) is not available on it.

## As BICTool's Excel/CSV export

The program's export and the batch tool write Excel and CSV files. From the API, use the batch tool or a plug-in
export format (`api.write_export`), see [Add an export format](export-format.md).

## Reading the JSON without BICTool

The JSON is plain: keys are the field names of the [result types](../reference/results.md), units are in
`"units"`, missing values are `null`. The layout is described in [JSON format](../reference/json-format.md).

```python
import json

data = json.load(open("case1.json", encoding="utf-8"))
print(data["schema_version"], data["verdict"]["verdict"], len(data["measurements"]))
```

## See also

* [`StudyResult`](../reference/results.md#studyresult)
* [Data protection](../explanation/data-protection.md)
