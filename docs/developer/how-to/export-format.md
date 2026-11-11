# How to add an export format

An export format writes a whole study result into a file of your design - for a registry, a report system or
your own analysis. It is offered in the program's Export dialog, in the API and in the batch tool.

## The plug-in

```python
import json

from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(name="registry-json", version="1.0", requires_api=">=0.3,<1",
                                    description="Short JSON for the QA registry")


def write(result, path):
    falx = result.measurements_for("falx")
    data = {
        "series": result.series.description,
        "date": result.series.acquisition_date,
        "verdict": result.verdict.verdict if result.verdict else None,
        "mtf_area_median": sorted(m.mtf_area for m in falx.values())[len(falx) // 2] if falx else None,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@plugins.hookimpl
def bictool_export_formats():
    return [plugins.ExportFormat("short", "Registry JSON", ".json", write)]
```

`write(result, path)` gets the study as an [`api.StudyResult`](../reference/results.md#studyresult) - the same
object in the program, the API and the batch tool - and a `pathlib.Path` ending with the format's extension.

## Where it is offered

| Where | How |
|---|---|
| Program | Export dialog, scope *Whole study*: tick box *Registry JSON (.json) — plug-in*. The file gets the name chosen in the save dialog. |
| API | `api.export_formats()` lists `{"registry-json.short": "Registry JSON"}`; `api.write_export(result, "D:/out/case1", "registry-json.short")` writes `D:/out/case1.json`. |
| Batch tool | `--plugin-export registry-json.short` writes `OUT/plugin_exports/<dataset> - <study>.json` for every study. |

## Rules

* The extension is a dot and up to 16 letters, digits, `_` or `-` (`.json`, `.pdf`, `.hl7`).
* Write only to `path`. If another file of the same export already has that name (e.g. your `.xlsx` next to
  BICTool's report), BICTool adds the format's id to the name.
* If `write` raises, the program reports "Export: plug-in problem" and still writes BICTool's own files; the
  API raises [`PluginFailed`](../reference/errors.md#pluginfailed).
* The result has no patient name, ID or birth date. What your format contains, and where else you send it, is
  your responsibility (GDPR/BDSG) - see [Data protection](../explanation/data-protection.md).

## See also

* [`bictool_export_formats`](../reference/hooks.md#bictool_export_formats), [`ExportFormat`](../reference/plugin-types.md#exportformat)
* [Result types](../reference/results.md)
