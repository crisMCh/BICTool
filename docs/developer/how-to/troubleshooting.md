# Troubleshooting and FAQ

## A plug-in does not work

Settings > Plug-ins (program) or `api.loaded_plugins()` (API) shows each plug-in's status and message.

| Status / symptom | Cause | Fix |
|---|---|---|
| not listed at all | wrong folder, or the file name starts with `_` or `.` | Put it into the folder shown in Settings > Plug-ins; rename it. |
| **error**: "no BICTOOL_PLUGIN = PluginInfo(...)" | the description is missing or misspelt | Add `BICTOOL_PLUGIN = plugins.PluginInfo(...)` at module level. |
| **error**: "ModuleNotFoundError: No module named ..." | the plug-in imports a package the program does not contain | See [packages](install-and-share-a-plugin.md#packages-the-plug-in-needs). |
| **error**: "implements none of the hooks" | no function is marked `@plugins.hookimpl`, or the names are misspelt | Check the message: it names `bictool_*` functions that are not hooks or not marked. |
| **error**: "PluginValidationError: ... Argument(s) {'roi_pair'} are declared in the hookimpl but can not be found in the hookspec" | an argument name differs from the hook's (e.g. `def bictool_pair_metrics(roi_pair)`) | Use the names in [Hooks](../reference/hooks.md): `pair`, `ct_slice`, `study`. |
| **error**: "another plug-in is already called ..." | two plug-ins with the same `name` | Rename one. |
| **incompatible** | `requires_api` does not include this BICTool's API version | Update the plug-in or BICTool; see the message for both versions. |
| **disabled** | switched off in Settings > Plug-ins | Tick it and restart. |
| **active** with a message "... is not a hook of this BICTool ... ignored" | a function named like a hook that this version does not have | Fix the name, or update BICTool (the plug-in may be made for a newer version). |

## A plug-in loads but shows "problem: ..."

The hook raised an exception or returned something BICTool could not use; the text after "problem:" says which
(e.g. "value 'a lot' is not a number", "the mask must be a bool array of shape (512, 512)"). The study is not
affected. Details, with the traceback, are in the log file `%LOCALAPPDATA%\BICTool\logs\bictool.log`.

## FAQ

**Can a plug-in change the study verdict?**
No. Plug-in values, markers and hints are shown and exported, never used for the verdict. See
[The plug-in system](../explanation/plugin-system.md).

**Can a plug-in replace BICTool's ROIs or segmentation?**
No. It can place and measure its own ROIs next to BICTool's
([Add a metric with ROIs of its own](metric-with-own-roi.md)).

**Why are my values NaN?**
NaN means "could not be computed" - returned by your plug-in, or a value that was infinite. In JSON it is
`null`.

**My WM/GM method makes the program slow.**
It runs on the program's main thread for every slice shown; above half a second per slice the program warns.
Make it faster (vectorise with NumPy) or precompute what you can.

**Why does `analyse_study` raise `SeriesError: Several studies under ...`?**
The folder holds images of more than one study. Give it a folder with one study.

**`ModelNotAvailable` - what now?**
The anatomy CNN's weights are not configured or the file is missing; see
[Your first analysis](../tutorials/first-analysis.md#1-prepare-the-environment).

**Are slice numbers 0- or 1-based?**
0-based in the API and in JSON; the program and its exports show index + 1.
