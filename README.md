# BICTool

BICTool is an open-source, standalone desktop application for objective, automated, quantitative image-quality assessment of head/skull CT DICOM images, correlated with dose indicators (CTDIvol, DLP). It is developed for the Bundesamt für Strahlenschutz (BfS) at Otto-von-Guericke-Universität Magdeburg.

See `requirements/requirements.md` and `architecture/layers_description.md` for the current requirements and architecture.

> **Decision support only.** BICTool is not a medical device and not meant for diagnosis. Its study verdict
> comes from a provisional model and must always be checked by a qualified person.

## What BICTool does

BICTool measures image quality directly in the patient images of a non-contrast head CT - no phantom scan
needed. For a study it:

1. **segments the anatomy** with an AI network (falx, septum pellucidum, ventricles, lobes, basal ganglia, ...)
   and separates white matter, grey matter and CSF by their CT numbers;
2. **places two ROI pairs** on every suitable slice: one ROI on a thin structure (falx or septum) for
   **sharpness**, one in homogeneous tissue next to it (white matter or ventricle) for **noise**;
3. **measures** sharpness (MTF), noise and noise texture (noise power spectrum) and the structure-to-noise
   balance (PS/MTF ratio);
4. gives a **study verdict** - *Acceptable*, *Not acceptable*, *Uncertain* or *Not assessable* (metal
   streaks in the brain) - with a probability, its 95 % interval and the measurements behind it, plus hints
   where to look (possible artefacts, basal-ganglia visibility, a segmentation check);
5. **exports** the results as an Excel report or CSV tables, without patient names or IDs.

Input: an axial head-CT DICOM series with a soft-tissue (brain) kernel; any matrix size and pixel spacing.

## Getting the program (portable ZIP for Windows)

The ready-to-run program is a ZIP file, **`BICTool-<version>-win64.zip`** (about 0.8 GB). It is **not in this
repository and not on GitHub**: it contains the third-party CNN weights, which may only be used internally
for now (see below). There are two ways to get it:

- **Ask the BICTool project team** for the current ZIP (internal distribution).
- **Build it yourself** from this repository (Windows, Python environment and CNN weights as described under
  *Development setup*):

  ```bash
  python docs/manual/build_manual.py      # manuals and help pages (needs Microsoft Edge)
  python packaging/build_portable.py      # builds the program and the ZIP
  ```

  The ZIP is then at **`packaging/dist/BICTool-<version>-win64.zip`**; the unpacked program folder is next to
  it at `packaging/dist/BICTool/`. Both are build output and stay out of git.

**Installing and starting:** unpack the whole ZIP into a folder of your choice (no administrator rights
needed) and start `BICTool.exe`. Windows may warn at the first start because the program is not signed -
click *More info > Run anyway*. Requirements: Windows 10/11, 64-bit, 8 GB RAM recommended. The folders
`_internal` and `models` must stay next to `BICTool.exe`. To check an installation without the user
interface: `start /wait "" BICTool.exe --self-test "<DICOM series folder>"` (result in
`%LOCALAPPDATA%\BICTool\logs\bictool.log`).

## Using BICTool in short

1. *File > Open DICOM Folder (Series)…* - BICTool picks the axial ~4 mm head series and segments it
   automatically (about 1-2 s per slice).
2. **Step 1 - Segmentation:** check the overlays; choose the WM/GM method that fits the scan best.
3. **Step 2 - ROI selection:** *Start auto ROIs*, check them on a few slices (drag to move, right-click to
   delete or save), then tick *ROIs checked - lock them for QA*.
4. **Step 3 - Run QA:** the *Results* page shows the curves per slice and the *Study summary* with the verdict.
5. *Export > Export Results…* - Excel report and/or CSV tables.

**Help:** press **F1** over any control (or click **? Help**) for the help window with the user manual
(English/German) and the technical reference (algorithms, parameters, limits). The same documents are in the
program's `manual` folder as PDFs; their sources are `docs/manual/en.md`, `de.md` and
`technical_reference_en.md`.

## Development setup

```bash
pip install -e ".[dev]"
pytest
```

### CNN weights (not in git)

The anatomy segmentation needs the Mayo Clinic RIL weights `weights_mayoiNPH.hdf5` (Cai et al., Radiol Artif
Intell 2020). They are not in the repository (third-party, no license, see `packaging/THIRD_PARTY_NOTICES.md`).
Download them from the [Mayo repository](https://github.com/jasonccai/HeadCTSegmentation) - the
["recommended for routine use" weights](https://drive.google.com/file/d/1fSe7oDaE8NYh5GLUN6h-Yo-ZlrJHXeuZ/view?usp=sharing) -
and save them as `reference/segmentation/weights_mayoiNPH.hdf5` (gitignored; step-by-step instructions in
[reference/README.md](reference/README.md)). Then tell BICTool where the file is - either in
`~/.bictool/config.json`:

```json
{"segmentation": {"anatomy_weights_path": "D:/path/to/weights_mayoiNPH.hdf5"}}
```

or with the environment variable `BICTOOL_HEADCT_WEIGHTS` (wins over the file). Use `weights_mayoiNPH`, not
`weights_mayo40`. Without weights the app starts, but segmentation, auto ROIs and the QA run are unavailable;
the tests do not need them (they use a fake model).

## Building the portable package and the manuals

```bash
pip install -e ".[gui,dicom,ml,build]"
python docs/manual/build_manual.py      # manual PDFs (EN, DE, technical reference) + help pages -> docs/manual/build/
python packaging/build_portable.py      # -> packaging/dist/BICTool-<version>-win64.zip
python packaging/build_portable.py --skip-pyinstaller   # only refresh manuals, notices and the ZIP
```

- The manuals are written in `docs/manual/en.md`, `de.md` and `technical_reference_en.md`. Their screenshots
  come from a real, anonymised study and are not in git; `build_manual.py --images <folder>` points to them.
- `build_portable.py` takes the CNN weights from the configured path (or `--weights <file>`).
- The package contains the anatomy CNN weights (`models/weights_mayoiNPH.hdf5`). Their source (Mayo Clinic RIL) states no license; they are bundled for internal use
  while redistribution permission is requested. Remove them before an external release unless permission
  has been given (`packaging/THIRD_PARTY_NOTICES.md`).
- License: GNU GPL v3 or later (`LICENSE`).

## Project layout

- `source/bictool/` — application source, organized by architectural layer (presentation, application, domain, integration, infrastructure)
- `tests/` — automated tests, mirroring the source layout
- `architecture/` — architecture description and diagram
- `docs/` — method documentation (`docs/qa_method/`: physical QA recipe, head-CT quality classifier, model card) and the user manuals (`docs/manual/`)
- `packaging/` — portable Windows build (PyInstaller spec, launcher with self-test, third-party notices)
- `requirements/` — requirements specification
- `design/` — design artifacts (UI mockups, etc.)
