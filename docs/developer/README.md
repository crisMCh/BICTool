# BICTool Developer Guide

BICTool assesses the image quality of head-CT series. Besides the program, it offers two ways in for your own
code:

* the **Python API** (`bictool.api`): run BICTool's assessment from a script, a research pipeline or an
  integration adapter, and get the results as Python objects or JSON;
* **plug-ins**: add your own metrics, ROIs, artefact hints, WM/GM methods or export formats to the program, the
  API and the batch tool, without changing BICTool.

> BICTool is decision support only; it is not a medical device. The study verdict comes from a provisional
> model and must be checked by a qualified person. Plug-in values never change the verdict.

**Status: provisional** - API version 0.3.0, result schema 1.2. Names may still change between minor versions;
every change is listed in the [Changelog](reference/changelog.md).

## Where to start

| I want to ... | Read |
|---|---|
| analyse a study from Python for the first time | Tutorial: [Your first analysis](tutorials/first-analysis.md) |
| write my first plug-in | Tutorial: [Your first plug-in](tutorials/first-plugin.md) |
| solve a specific task | the How-to guides below |
| look up a function, field, hook or error | the Reference below |
| understand how BICTool runs a study and why plug-ins work the way they do | the Explanations below |

The four kinds of pages have different jobs: **tutorials** teach by doing, **how-to guides** solve one task,
the **reference** describes every name precisely, **explanations** give the background. Each page links to the
others where you need them.

## Tutorials

* [Your first analysis](tutorials/first-analysis.md) - one study from folder to result, step by step.
* [Your first plug-in](tutorials/first-plugin.md) - a working metric plug-in, tried in the program and the API.

## How-to guides

Python API:

* [Check or change the ROIs before measuring](how-to/check-rois-before-measuring.md)
* [Choose the QA recipe and the WM/GM method](how-to/choose-settings.md)
* [Analyse many studies](how-to/run-many-studies.md)
* [Save results and read them back](how-to/save-and-read-results.md)

Plug-ins:

* [Add a metric with ROIs of its own](how-to/metric-with-own-roi.md)
* [Add possible-artefact hints](how-to/artefact-hints.md)
* [Add a WM/GM method](how-to/wmgm-method.md)
* [Add an export format](how-to/export-format.md)
* [Test a plug-in](how-to/test-a-plugin.md)
* [Install and share a plug-in](how-to/install-and-share-a-plugin.md)
* [Troubleshooting and FAQ](how-to/troubleshooting.md)

## Reference

Python API:

* [Functions and the Study class](reference/api-functions.md)
* [ROIs](reference/rois.md)
* [Settings](reference/settings.md)
* [Result types](reference/results.md)
* [Errors](reference/errors.md)
* [JSON format](reference/json-format.md)
* [Command line](reference/command-line.md)

Plug-ins:

* [Hooks](reference/hooks.md)
* [Plug-in types](reference/plugin-types.md)

Both:

* [Changelog and versioning](reference/changelog.md)

## Explanations

* [How BICTool runs a study](explanation/how-a-study-is-run.md)
* [The plug-in system: rules, decisions, verification](explanation/plugin-system.md)
* [Data protection](explanation/data-protection.md)

## Conventions used everywhere

| Topic | Convention |
|---|---|
| Slice numbers | 0-based indices in code and results; the program shows index + 1. |
| Units | In `api.UNITS` and in every JSON result; mm, HU, 1/mm. |
| Missing values | NaN in Python, `null` in JSON. |
| Ids | Short lower-case strings: `"falx"`, `"white_matter"`, `"not_acceptable"`, `"adaptive"`. |
| Immutability | Settings and results are frozen dataclasses; change a copy with `dataclasses.replace`. |
| Public names | Only the names in `bictool.api.__all__` and `bictool.api.plugins.__all__`. Everything else in `bictool` is internal and may change at any time. |
