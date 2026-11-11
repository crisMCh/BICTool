# The plug-in system: rules, decisions, verification

*Status: plug-in API 0.3.0 (2026-11-11), result schema 1.2. First version (API 0.2.0, pair and study metrics,
WM/GM methods) 2026-11-10. Provisional like the whole Python API (0.x). This document is for developers who
maintain BICTool; how to **write** a plug-in is in the [tutorial](../tutorials/first-plugin.md), the
[how-to guides](../README.md#how-to-guides) and the [hook reference](../reference/hooks.md).*

## Why

FR-08 asks that BICTool can be extended without changing BICTool itself. The expected extensions are
additional metrics and other WM/GM methods, but the system should stay open for other kinds (user, 2026-11-10).
A plug-in is ordinary Python code that BICTool calls at defined points ("hooks"). The same plug-ins run in the
program, the Python API and the batch tool.

## Rules the system keeps

These hold for every hook, present and future. A change that breaks one of them is a design change, not an
extension, and needs the user's decision.

1. **The verdict stays closed.** No plug-in output enters the study verdict, the metal rule or the verdict
   text. Plug-in values, markers and hints are descriptive only.
2. **BICTool's own chain is unchanged.** The falx/septum ROI placement, the segmentation, the QA recipe and the
   verdict model are not replaceable by plug-ins: the verdict model (head-ct-1) was trained on them. Without
   plug-ins, BICTool's results are byte-identical to a BICTool without the plug-in system (verified, see below).
3. **Read-only inputs.** Images and masks are handed out as read-only NumPy views, mappings as
   `MappingProxyType`, contexts as frozen dataclasses (`pipeline.read_only`, `read_only_masks`).
4. **One plug-in cannot stop a study.** Every hook is called for each plug-in separately
   (`PluginManager._call`), never through pluggy's joint call: an exception becomes a recorded error text,
   invalid return values are dropped with a note, and the study goes on.
5. **Plug-ins return data, BICTool renders it.** No plug-in gets a widget, the window or a painter. This keeps
   every hook usable in the program, the API and the batch tool alike.
6. **One pipeline.** Hooks are called only from `application/pipeline.py`, which the program's ViewModels, the
   API (`api.Study`) and the batch tool all use. Tests check that the program and the API give plug-ins the same
   inputs and get the same outputs (`tests/presentation/test_api_equivalence.py`).
7. **Nothing is computed for plug-ins that are not there.** Without plug-ins for a hook, the hook's inputs are
   not built (`has_*` checks; `PluginInputs` is created only when a slice, study or hint hook exists).
8. **Additive versioning.** A new hook or context field raises `API_VERSION` by a minor step and is listed in
   the [Changelog](../reference/changelog.md); existing hooks, fields and result keys are not renamed or removed before 1.0.
   Plug-ins declare the versions they accept (`PluginInfo.requires_api`, e.g. `">=0.3,<1"`).
9. **No patient identifiers** in anything BICTool writes about plug-ins (log lines, exports). The result handed
   to export plug-ins has no patient name, ID or birth date; DICOM UIDs only with `include_dicom_uids`.

## Where the code is

| Layer | Module | Role |
|---|---|---|
| Infrastructure | `infrastructure/plugin_loader.py` | Finds plug-ins: entry points (group `bictool.plugins`) and `.py` files / packages in the plug-in folders (`app_paths.plugin_folders()`: `plugins/` next to BICTool.exe, `~/.bictool/plugins`, `BICTOOL_PLUGIN_PATH`). |
| Application | `application/plugin_types.py` | Hook specifications and the types plug-ins see or return (internal home). |
| Application | `application/plugin_manager.py` | Checks and registers plug-ins (status active / disabled / incompatible / error), calls hooks per plug-in, validates values, markers, hints, WM/GM methods and export formats. `current()` / `use()` = the manager in use. |
| Application | `application/pipeline.py` | Builds the contexts (`pair_context`, `slice_context`, `study_context`, `PluginInputs`) and runs the hooks (`plugin_pair_metrics`, `plugin_study_phase`). |
| API | `api/plugins.py` | What plug-in authors import (`from bictool.api import plugins`). |
| API | `api/results.py`, `api/study.py` | Plug-in output in `StudyResult` (JSON schema 1.2); `load_plugins`, `export_formats`, `write_export`, `PluginFailed`. |
| Presentation | `viewmodels/qa_run_viewmodel.py` | Pair hook per QA job; one background job after the run for slice / study / hint hooks. |
| Presentation | `viewmodels/results_viewmodel.py`, `export_viewmodel.py`, `main_viewmodel.py` | Plug-in lines and notes, export rows and formats; `MainViewModel.study_result()` = the run as the API's `StudyResult`. |
| Presentation | `views/main_window.py`, `export_dialog.py`, `plugins_dialog.py`, `plugin_settings_dialog.py` | "Plug-in ROIs" switch, slow-method warning, plug-in formats in the Export dialog, Settings > Plug-ins, generated WM/GM settings dialog. |
| Batch | `application/batch_qa.py` | Runs `api.Study` (so all hooks), Plug-ins sheet/CSV, `--plugin-export`. |

## The hooks

| Hook | Called | Input | Returns | Shown / kept |
|---|---|---|---|---|
| `bictool_pair_metrics` | per measured ROI pair | `PairContext` | `MetricValue`s, `RoiMarker`s | slice note, export, `Measurement.plugin_values` |
| `bictool_slice_metrics` (0.3) | per slice with a measured pair, after its pairs | `SliceContext` | `MetricValue`s, `RoiMarker`s | slice note, export ("slice" rows), `StudyResult.slice_plugin_values` |
| `bictool_study_metrics` | once, after pairs and slices | `StudyContext` | `MetricValue`s, `RoiMarker`s (with slice) | Study summary Details, export, `StudyResult.plugin_values` |
| `bictool_artefact_hints` (0.3) | once, after the study metrics | `StudyContext` | `ArtefactHint`s | Study summary (with BICTool's artefacts), slice notes, export, `StudyResult.plugin_hints` |
| `bictool_tissue_methods` | once, at load | - | `TissueMethod`s | Brain Tissue > Method drop-down; `TissueSettings(method="<plug-in>.<id>")` |
| `bictool_export_formats` (0.3) | once, at load | - | `ExportFormat`s | Export dialog (Whole study), `api.write_export`, batch `--plugin-export` |

### Flow of one QA run

1. **Start of the program** (`main_window.main`): the plug-ins are loaded once, without those switched off in
   Settings > Plug-ins (QSettings `plugins/disabled`), and `PluginManager.log_status()` writes one log line each.
   WM/GM methods and export formats are collected at load.
2. **Each ROI pair** (QA job, background thread): BICTool measures the pair; if any metric or hint hook exists,
   the pair's `PairContext` is kept and the pair hook runs.
3. **After the last pair**: one more background job runs `pipeline.plugin_study_phase` - the slice hook for each
   measured slice, then the study hook, then the hint hook. `api.Study.measure` calls the same function.
4. **Export**: BICTool's own files first, then each chosen plug-in format with `MainViewModel.study_result()`.

### Inputs computed on demand (`PluginInputs`)

`SliceContext.brain_mask()` / `tissue()` and `StudyContext.brain_mask(i)` / `tissue(i)` share one
`PluginInputs` object per run. The brain mask is the one of the program's "Brain Mask" overlay (segmentation,
else the HU skull-strip). The WM/GM masks come from `pipeline.study_tissue_classifier` with the WM/GM method the
user chose (program: Brain Tissue > Method, snapshotted when the plug-in job starts; API: `Settings.tissue`).
Results are cached per slice and guarded by a lock. Because plug-in values then depend on the chosen method,
BICTool records the method when a plug-in asked for WM/GM masks (Study summary Details; export row "plug-in
input / WM/GM method").

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-10 | pluggy as hook mechanism (user's choice) | Established (pytest, tox), small, per-hook argument checking. |
| 2026-11-10 | `plugins/` folder next to BICTool.exe, plus entry points (user's choice) | Copy-a-file installation for users; packages for developers. |
| 2026-11-10 | Ensemble noise stays built in; plug-ins get a separate example (user's choice) | FR-06 noise is a core value, not an add-on. |
| 2026-11-10 | Plug-in security (approval list with SHA-256 fingerprints) postponed (user) | Decision on list location and default behaviour still open. Until then plug-ins are trusted code; docs say "only from people you trust". |
| 2026-11-11 | Contribution points, not access to internals | Plug-ins that reach into internals break on every refactor and make the equivalence checks impossible. Flexibility comes from rich read-only inputs and open result vocabularies (free names and units). |
| 2026-11-11 | Separate slice hook in addition to the study hook | Most per-slice metrics are simpler to write per slice; the study hook stays for looking across slices. |
| 2026-11-11 | ROI markers as bool masks of any shape, cropped to their box | New metrics may need ROIs of other shapes and places than BICTool's pairs, without touching those pairs. Cropping keeps memory small; JSON stores the mask as base64 of the packed bits. |
| 2026-11-11 | Artefact hints: one neutral line and slices, no severity or score | Same rule as BICTool's own artefact flags; BICTool is not diagnostic. |
| 2026-11-11 | Export formats get the public `api.StudyResult` | Stable, versioned, without patient identifiers; the same object in program, API and batch. The program builds it with `MainViewModel.study_result()`, tested equal to `api.Study.measure`. |
| 2026-11-11 | Unknown or unmarked `bictool_*` functions are named in the plug-in's status | pluggy ignores them silently; a typo or a plug-in made for a newer BICTool would otherwise just do nothing. |
| 2026-11-11 | Warning when a plug-in WM/GM method needs > 0.5 s per slice | WM/GM runs on the program's main thread (milliseconds for the built-in methods); a slow method freezes the window on every new slice. |
| 2026-11-11 | The portable build contains all of SciPy | Plug-ins can rely on NumPy and SciPy; before, only the SciPy parts BICTool imports were bundled. |

### Fixed along the way (2026-11-11)

* Plug-in WM/GM methods got no CNN labels and, with Adaptive smoothing off, no pixel spacing in the program,
  while the API gave both. The program now passes the same inputs.
* The program's ROI store did not record where an automatic ROI2 lies (ventricle / white matter / fallback) or
  that none was found; the API's store did. It now records both (the window does not use them, so nothing
  visible changed), which made the program's `StudyResult` equal to the API's.

## Not built (design only)

| Extension | Why not now | What it would need |
|---|---|---|
| Replacing BICTool's ROI placement | The MTF needs ROI1 on the falx/septum (seeds on the structure); ROI2 regions, square whole-pixel ROIs and manual editing depend on it; the verdict model was trained on this placement. A new metric that needs other ROIs uses its own ROIs (slice hook + markers) instead. | Clear rules for the placement, a method choice in GUI and API, the method recorded in results, verdict marked not validated. About 2-3 days. |
| Another segmentation model | Almost everything uses the 16 Mayo labels (ROIs, basal ganglia, brainstem band, segmentation check); the verdict is not validated with another model; the portable build cannot load packages it does not contain (e.g. PyTorch, ONNX Runtime). | Label mapping, cache keyed by model, verdict suppressed or marked, real-data check; a concrete model first. About 4-6 days plus checks. |
| Editing plug-in ROIs by hand | Not needed yet. | An editing tool for any ROI shape in the image view. About 2-3 days. |
| "Study opened / closed" hook | Stateless hooks are easier to isolate; no plug-in needs it yet. | Add when a concrete plug-in needs per-study state. |
| Security sandbox | Postponed by the user (see Decisions). Options: approval list (about half a day, recommended), helper process (3-5 days, robustness only), OS sandbox (weeks). | The user's decision on list location and default. |

## Verification (2026-11-11)

* **Tests:** 769 passed, 5 skipped (757 before the 0.3 work). New tests cover each hook, invalid markers and
  hints, the absence of extra work without plug-ins, the program's plug-in ROIs and slow-method warning, the
  export formats, and the equivalence of program and API (plug-in inputs and outputs; the whole `StudyResult`).
* **Real data, batch tool:** all CanonPrime and Siemens ER studies of the internal test data, run with the code
  before (commit `5f1b712`) and after the change, both without plug-ins: all outputs identical (CSV byte for
  byte, Excel cell by cell). With the three example plug-ins: BICTool's own outputs still identical; plug-in
  values only in the added Plug-ins sheet and CSV; no plug-in errors.
* **Real data, program window:** three real studies (CanonPrime; Siemens ER, one with implants), driven
  offscreen through the real window (cached segmentation, auto ROIs, confirm, Run QA, study export) with the old
  and the new code: Excel report, summary and per-slice CSV and the Study summary text identical; the same ROI
  pairs failed.
* **Portable build:** built and started with the three examples and a probe plug-in importing SciPy parts
  BICTool does not use (`cluster`, `fft`, `integrate`, `linalg`, `optimize`, `spatial`): all active. The built
  program's `--self-test` on a real study passed with the plug-ins active.
* All outputs derived from patient data were deleted after the comparison; the user's settings were not touched
  (the window test used its own settings file).

## Adding a hook later

Answer these before writing code, and follow the rules above:

1. **Input:** an existing context (preferred - add a field) or a new frozen dataclass?
2. **Output:** a plain frozen dataclass; how is it validated, and what note does an invalid entry leave?
3. **Where it appears:** program, API result (schema minor bump), exports, batch - all of them, never the GUI
   only.
4. **Failure:** what the user sees when the plug-in raises (a recorded problem; the study and the other plug-ins
   go on).
5. **Order:** the call order of plug-ins is not guaranteed (results are listed sorted by plug-in name); no
   plug-in may depend on another one having run.
6. **Threads:** hooks may run on a background thread and must not need a Qt event loop.
7. **Version and docs:** `API_VERSION` minor bump, entry in the [Changelog](../reference/changelog.md), an entry
   in [Hooks](../reference/hooks.md) and [Plug-in types](../reference/plugin-types.md), a how-to guide, an
   example in `examples/plugins/` (shipped in the ZIP), manual lines (EN and DE) where the program changes. The
   documentation tests (`tests/docs/`) fail until the reference names every new hook and field.

## Open

* Plug-in security decision (approval list: where the list lives, behaviour without a list).
* Screenshots of the plug-in elements (Plug-in ROIs switch, slow-method warning, plug-in formats in the Export
  dialog) are not in the manuals; they appear only when plug-ins are installed.
* `architecture/layers_description.md` does not describe the plug-in system yet (the user maintains it).
