# BICTool

BICTool is an open-source, standalone desktop application for objective, automated, quantitative image-quality assessment of head/skull CT DICOM images, correlated with dose indicators (CTDIvol, DLP). It is developed for the Bundesamt für Strahlenschutz (BfS) at Otto-von-Guericke-Universität Magdeburg.

See `requirements/requirements.md` and `architecture/layers_description.md` for the current requirements and architecture.

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

## Portable Windows package and user manuals

```bash
pip install -e ".[gui,dicom,ml,build]"
python docs/manual/build_manual.py      # EN + DE manual PDFs -> docs/manual/build/ (needs Microsoft Edge)
python packaging/build_portable.py      # -> packaging/dist/BICTool-<version>-win64.zip
packaging/dist/BICTool/BICTool.exe --self-test "<DICOM series folder>"
```

- The manuals are written in `docs/manual/en.md` and `de.md`. Their screenshots come from a real, anonymised
  study and are not in git; `build_manual.py --images <folder>` points to them.
- The package contains the anatomy CNN weights (`models/weights_mayoiNPH.hdf5`, taken from the configured
  path or `--weights`). Their source (Mayo Clinic RIL) states no license; they are bundled for internal use
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
