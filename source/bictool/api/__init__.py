"""BICTool's Python API (FR-08): the image-quality assessment of a head-CT
series from a script, a plug-in or an integration adapter - the same
pipeline the program runs.

    from bictool import api

    result = api.analyse_study("D:/studies/case1", label="case 1")
    print(result.verdict.verdict if result.verdict else "no verdict")
    open("case1.json", "w").write(result.to_json())

Or in steps, to check or change the ROIs before measuring:

    study = api.open_study("D:/studies/case1")
    study.segment()
    rois = study.place_rois()
    rois.delete(12, api.roi_key(api.FALX, "roi1"))
    result = study.measure(rois, reviewed=True)

Plug-ins (extra metrics, WM/GM methods, ...) are written against
`bictool.api.plugins`; `load_plugins` / `loaded_plugins` control which run.

Provisional (`API_VERSION` 0.x): the names in `__all__` are the API; they may
still change between minor versions, and every change is listed in
docs/developer. Everything outside `bictool.api` is internal.
"""

from bictool.api._version import API_VERSION
from bictool.api.errors import (
    BictoolError,
    Cancelled,
    InvalidRois,
    InvalidSegmentation,
    ModelNotAvailable,
    NotSegmented,
    PluginFailed,
    SeriesError,
)
from bictool.api.results import (
    ACCEPTABLE,
    FALLBACK,
    NOT_ACCEPTABLE,
    NOT_ASSESSABLE,
    SCHEMA_VERSION,
    UNCERTAIN,
    UNITS,
    VENTRICLE,
    WHITE_MATTER,
    ArtefactHints,
    BasalGangliaSlice,
    Curves,
    Failure,
    FeatureFinding,
    Measurement,
    NoiseScatter,
    SegmentationCheck,
    SeptumCheck,
    SeriesChoice,
    SeriesInfo,
    SliceRef,
    StudyResult,
    Verdict,
)
from bictool.api import plugins
from bictool.api.study import (
    Study,
    analyse_study,
    export_formats,
    list_series,
    load_anatomy_model,
    load_plugins,
    loaded_plugins,
    open_study,
    write_export,
)
from bictool.application.qa_settings import (
    METHOD_ADAPTIVE,
    METHOD_ANCHORED,
    METHOD_FCM,
    METHOD_LABELS,
    METHOD_THRESHOLD,
    AdaptiveSettings,
    BrainTissueSettings,
    FcmSettings,
    Settings,
    TissueSettings,
)
from bictool.application.roi_store import FALX, PAIRS, SEPTUM, SLOTS, RoiPairStore, RoiRect, roi_key
from bictool.domain.results.study_quality import RECIPE_OCTAVE, RECIPE_PHYSICAL

__all__ = [
    "API_VERSION",
    "SCHEMA_VERSION",
    # running
    "analyse_study",
    "list_series",
    "open_study",
    "load_anatomy_model",
    "Study",
    "SeriesChoice",
    # plug-ins (writing one: bictool.api.plugins)
    "plugins",
    "load_plugins",
    "loaded_plugins",
    "export_formats",
    "write_export",
    # settings
    "Settings",
    "TissueSettings",
    "AdaptiveSettings",
    "BrainTissueSettings",
    "FcmSettings",
    "METHOD_ADAPTIVE",
    "METHOD_ANCHORED",
    "METHOD_FCM",
    "METHOD_THRESHOLD",
    "METHOD_LABELS",
    "RECIPE_PHYSICAL",
    "RECIPE_OCTAVE",
    # ROIs
    "RoiPairStore",
    "RoiRect",
    "roi_key",
    "FALX",
    "SEPTUM",
    "PAIRS",
    "SLOTS",
    # results
    "StudyResult",
    "SeriesInfo",
    "SliceRef",
    "Measurement",
    "Curves",
    "Failure",
    "Verdict",
    "FeatureFinding",
    "NoiseScatter",
    "SeptumCheck",
    "ArtefactHints",
    "BasalGangliaSlice",
    "SegmentationCheck",
    "UNITS",
    "ACCEPTABLE",
    "NOT_ACCEPTABLE",
    "UNCERTAIN",
    "NOT_ASSESSABLE",
    "VENTRICLE",
    "WHITE_MATTER",
    "FALLBACK",
    # errors
    "BictoolError",
    "SeriesError",
    "ModelNotAvailable",
    "NotSegmented",
    "InvalidSegmentation",
    "InvalidRois",
    "PluginFailed",
    "Cancelled",
]
