# How to install and share a plug-in

## As a file (simplest)

A plug-in is a `.py` file, or a folder with an `__init__.py` for a larger one. Give it to colleagues; they copy
it into their plug-in folder and restart BICTool:

| BICTool runs as | Plug-in folder |
|---|---|
| portable program | `plugins` next to `BICTool.exe` |
| source code (`python -m bictool`, API) | `~/.bictool/plugins` |
| either, additionally | every folder in the environment variable `BICTOOL_PLUGIN_PATH` (separated like `PATH`) |

Names starting with `_` or `.` are ignored, so `_draft.py` is not loaded.

## As a Python package

For a plug-in that is installed with `pip` (e.g. inside an institute's Python environment), declare an entry
point in the group `bictool.plugins` in the package's `pyproject.toml`:

```toml
[project.entry-points."bictool.plugins"]
roi2-sd = "roi2_sd"          # name = the module that holds BICTOOL_PLUGIN and the hooks
```

Installed packages are found by `python -m bictool`, the API and the batch tool - not by the portable program,
which cannot see packages installed into a Python on the computer.

## Packages the plug-in needs

The portable program contains only Python's standard library, NumPy, SciPy and BICTool. A plug-in that needs
more (e.g. PyTorch, ONNX Runtime, scikit-image) works only with BICTool run from a Python installation that has
those packages. Import them inside the hook functions, so that the plug-in still loads everywhere and reports a
clear error where the package is missing.

## Versions

* Give each release a new `version` in `PluginInfo`; BICTool shows it and records it in every result
  (`StudyResult.plugins`).
* Set `requires_api` to the API versions you tested, e.g. `">=0.3,<1"`. A BICTool outside that range shows the
  plug-in as *incompatible* and does not load it.

## Switching plug-ins off

In the program, Settings > Plug-ins has a tick box per plug-in (takes effect after a restart). In the API, pass
`disabled=["name"]` to `api.load_plugins`, or `api.load_plugins(folders=[], installed=False)` for none at all.

## Trust

Plug-ins run inside BICTool with full access to the images and the computer. Install only plug-ins from people
you trust; BICTool never downloads plug-ins.

## See also

* [`PluginInfo`](../reference/plugin-types.md#plugininfo), [Changelog and versioning](../reference/changelog.md)
