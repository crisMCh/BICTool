# How to add possible-artefact hints

An artefact hint is one neutral line of text and the slices it concerns. The program lists it with BICTool's
own possible artefacts on the Study summary and in the notes of those slices; the export and the API keep it.
Hints are descriptive: no score, no verdict, no diagnosis.

## The plug-in

```python
import numpy as np

from bictool.api import plugins

BICTOOL_PLUGIN = plugins.PluginInfo(name="air-check", version="1.0", requires_api=">=0.3,<1",
                                    description="Slices with air inside the brain mask")

AIR_HU = -200.0
MIN_PIXELS = 20


@plugins.hookimpl
def bictool_artefact_hints(study):
    slices = tuple(
        index for index in study.measured_slices
        if np.count_nonzero(study.image(index)[study.brain_mask(index)] < AIR_HU) >= MIN_PIXELS
    )
    if not slices:
        return []
    return [plugins.ArtefactHint("Air-like values inside the brain mask", slices)]
```

The program then shows, for example:

```text
Possible artefact (plug-in air-check): Air-like values inside the brain mask (slice 12–14)
```

## Rules for the text

* One line, at most 200 characters; line breaks and repeated spaces are collapsed.
* Neutral and descriptive ("Air-like values inside the brain mask"), no judgement ("bad image", "unusable"),
  no diagnosis. BICTool is not diagnostic.
* Slices are 0-based indices of the series; the program shows them + 1.

Invalid hints are dropped with a note in `PluginHints.error`, shown on the Study summary.

## See also

* [`bictool_artefact_hints`](../reference/hooks.md#bictool_artefact_hints), [`ArtefactHint`](../reference/plugin-types.md#artefacthint)
* BICTool's own artefact flags: `docs/qa_method/artefact_flags.md`
