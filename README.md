# BICTool

BICTool is an open-source, standalone desktop application for objective, automated, quantitative image-quality assessment of head/skull CT DICOM images, correlated with dose indicators (CTDIvol, DLP). It is developed for the Bundesamt für Strahlenschutz (BfS) at Otto-von-Guericke-Universität Magdeburg.

See `requirements/requirements.md` and `architecture/layers_description.md` for the current requirements and architecture.

## Development setup

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

- `source/bictool/` — application source, organized by architectural layer (presentation, application, domain, integration, infrastructure)
- `tests/` — automated tests, mirroring the source layout
- `architecture/` — architecture description and diagram
- `docs/` — method documentation (`docs/qa_method/`: physical QA recipe, head-CT quality classifier, model card)
- `requirements/` — requirements specification
- `design/` — design artifacts (UI mockups, etc.)
