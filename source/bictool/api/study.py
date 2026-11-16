"""One series of a study from folder to `StudyResult`, in one call
(`analyse_study`) or in steps with a chance to check or change the ROIs:

    study = open_study(folder)
    study.segment()
    rois = study.place_rois()          # RoiPairStore: set_manual / delete / restore_slice
    result = study.measure(rois, reviewed=True)

Synchronous - a caller that needs a responsive UI runs it in a thread (the
program itself runs the same pipeline steps in its own background jobs).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bictool.api import results as R
from bictool.api.errors import (
    Cancelled,
    InvalidRois,
    InvalidSegmentation,
    ModelNotAvailable,
    NotSegmented,
    PluginFailed,
    SeriesError,
)
from bictool.application import pipeline, plugin_manager
from bictool.application.qa_settings import Settings
from bictool.application.roi_store import FALX, PAIRS, SEPTUM, RoiPairStore, roi_key, roi_problem
from bictool.domain.results.study_quality import RECIPE_PHYSICAL
from bictool.integration import model_files
from bictool.integration.segmentation_cache import (
    CacheMismatch,
    load_confidence,
    load_segmentation,
    save_segmentation,
    series_key,
)

if TYPE_CHECKING:
    from bictool.integration.dicom_file_loader import DicomVolume

logger = logging.getLogger(__name__)

# progress(stage, done, total) - stage is "segmentation", "rois" or "measurement".
Progress = Callable[[str, int, int], None]
# The loaded anatomy U-Net (a Keras model; `load_anatomy_model`).
AnatomyModel = object
_SHIPPED_MODEL = object()
# A measurement that fails on one slice (e.g. no usable edge for the MTF) is
# recorded in `StudyResult.failed`; any other error is a bug and is raised.
_MEASUREMENT_FAILURES = (ValueError, ArithmeticError, IndexError)


def list_series(path: str | Path) -> list[R.SeriesChoice]:
    """The series in a study folder, best first (CT, axial, closest to 4 mm) -
    `open_study(path)` takes the first unless told otherwise."""
    if not Path(path).exists():
        raise SeriesError(f"{path} does not exist")
    try:
        candidates = pipeline.list_series(path)
    except ValueError as exc:
        raise _series_error(path, exc) from None
    return [R.SeriesChoice(i, c.label, c.num_slices, c.axial) for i, c in enumerate(candidates)]


def _series_error(path, exc: Exception) -> SeriesError:
    if "different studies" in str(exc):  # the loader's message names the patients - not repeated here
        return SeriesError(f"Several studies under {path} - point at one study's folder")
    return SeriesError(str(exc))


def load_anatomy_model(weights_path: str | Path | None = None) -> AnatomyModel:
    """The anatomy U-Net with its weights (`weights_path`, else the configured
    or bundled ones). Load it once to reuse it for several studies."""
    try:
        return pipeline.load_anatomy_model(weights_path)
    except ImportError as exc:
        raise ModelNotAvailable(f"The segmentation needs TensorFlow (pip install bictool[ml]): {exc}") from exc
    except FileNotFoundError as exc:
        raise ModelNotAvailable(str(exc)) from exc


class Study:
    """One series of a study and what has been computed on it so far."""

    def __init__(self, volume: DicomVolume, *, label: str | None = None, settings: Settings | None = None) -> None:
        self.volume = volume
        self.series = R.SeriesInfo.from_volume(volume)
        # Never the folder name - it often carries the scan date and time.
        self.label = label or f"study - {self.series.description or 'series'}"
        self.settings = settings or Settings()
        self.series_key = series_key(volume.slice_paths)
        self.masks_by_slice: dict[int, dict[str, np.ndarray]] = {}
        self.confidence_by_slice: dict[int, object] = {}
        self._classify = None   # WM/GM classifier of the current segmentation (cached)
        self._brains: dict[int, np.ndarray] = {}
        self._tissue: dict[int, dict] = {}

    # ---- segmentation ----

    @property
    def is_segmented(self) -> bool:
        return bool(self.masks_by_slice)

    def segment(
        self,
        model: AnatomyModel | None = None,
        *,
        cache_file: str | Path | None = None,
        progress: Progress | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> int:
        """CNN anatomy segmentation (about 1-2 s per slice). `model` = a loaded
        U-Net (`load_anatomy_model`), else the configured weights are loaded.
        With `cache_file`, a cache of this series is read instead, and a new one
        written when the segmentation is complete. Slices already segmented are
        skipped, so a cancelled run continues where it stopped. Returns the
        number of segmented slices."""
        cache = Path(cache_file) if cache_file is not None else None
        if cache is not None and cache.is_file():
            try:
                masks = load_segmentation(cache, key=self.series_key)
            except CacheMismatch as exc:
                raise InvalidSegmentation(str(exc)) from None
            self.set_segmentation(masks, load_confidence(cache))
            return len(self.masks_by_slice)
        if model is None:
            model = load_anatomy_model()
        nifti = pipeline.prepare_segmentation(self.volume)
        total = self.volume.num_slices
        for index in range(total):
            if index in self.masks_by_slice:
                continue
            if cancel is not None and cancel():
                self._segmentation_changed()
                raise Cancelled(f"segmentation stopped: {len(self.masks_by_slice)}/{total} slice(s) done")
            result = pipeline.segment_slice(model, nifti, index)
            if result is not None:
                self.masks_by_slice[index] = result.masks
                if result.confidence is not None:
                    self.confidence_by_slice[index] = result.confidence
            if progress is not None:
                progress("segmentation", index + 1, total)
        self._segmentation_changed()
        if cache is not None:
            save_segmentation(cache, self.masks_by_slice, self.confidence_by_slice, key=self.series_key)
        return len(self.masks_by_slice)

    def set_segmentation(self, masks_by_slice: dict[int, dict], confidence_by_slice: dict | None = None) -> None:
        """Use an existing segmentation: {slice index: {label name: mask}} in DICOM pixel space."""
        for index, masks in masks_by_slice.items():
            if not 0 <= index < self.volume.num_slices:
                raise InvalidSegmentation(f"Slice {index} does not exist - the series has {self.volume.num_slices}")
            shape = self.volume.get_slice_hu(index).shape
            if any(mask.shape != shape for mask in masks.values()):
                raise InvalidSegmentation(f"The masks of slice {index} do not match its {shape} image")
        self.masks_by_slice = dict(masks_by_slice)
        self.confidence_by_slice = dict(confidence_by_slice or {})
        self._segmentation_changed()

    def _segmentation_changed(self) -> None:
        self._classify, self._brains, self._tissue = None, {}, {}

    def _require_segmentation(self) -> None:
        if not self.masks_by_slice:
            raise NotSegmented("Segment the study first (Study.segment or Study.set_segmentation)")

    def _tissue_masks(self, index: int) -> dict:
        """WM/GM masks of a segmented slice (for a white-matter ROI2), computed once."""
        if index not in self._tissue:
            if self._classify is None:
                self._classify = pipeline.study_tissue_classifier(self.settings.tissue, self.masks_by_slice, self.volume)
                self._brains = pipeline.brain_masks(self.masks_by_slice, self.volume.num_slices)
            image = self.volume.get_slice_hu(index)
            brain = self._brains.get(index)
            brain = brain if brain is not None else np.zeros(image.shape, dtype=bool)
            self._tissue[index] = self._classify(image, brain, index).masks
        return self._tissue[index]

    # ---- ROIs ----

    def place_rois(self, progress: Progress | None = None) -> RoiPairStore:
        """The automatic falx and septum pairs on every segmented slice with enough
        brain. Change them with `set_manual` / `delete` / `restore_slice` before `measure`."""
        self._require_segmentation()
        volume, masks_by_slice = self.volume, self.masks_by_slice
        spacing = volume.pixel_spacing_mm
        eligible = pipeline.eligible_for_auto_rois(masks_by_slice)
        midline = pipeline.midline_direction(masks_by_slice)

        store = RoiPairStore(series_key=self.series_key)
        skipped: dict[str, list[int]] = {pair: [] for pair in PAIRS}
        ordered = sorted(masks_by_slice)
        for done, index in enumerate(ordered, start=1):
            masks, image = masks_by_slice[index], volume.get_slice_hu(index)
            for pair in PAIRS:
                roi1 = pipeline.auto_roi1(masks, pair, spacing, midline, image)
                if roi1 is None:
                    continue
                if index not in eligible:  # too little brain: no automatic ROIs
                    skipped[pair].append(index)
                    continue
                store.set_auto(index, roi_key(pair, "roi1"), roi1)
                self._place_roi2(store, index, pair, roi1)
            if progress is not None:
                progress("rois", done, len(ordered))
        store.skipped = {pair: tuple(slices) for pair, slices in skipped.items()}
        return store

    def _place_roi2(self, store: RoiPairStore, index: int, pair: str, roi1) -> None:
        found = pipeline.auto_roi2(
            roi1, pair, self.masks_by_slice[index], self.volume.pixel_spacing_mm, self._tissue_masks(index)
        )
        key2 = roi_key(pair, "roi2")
        if found is None:
            store.set_auto(index, key2, None)
            store.unplaced.add((index, pair))
        else:
            store.set_auto(index, key2, *found)
            store.unplaced.discard((index, pair))

    # ---- measurement ----

    def measure(
        self,
        rois: RoiPairStore | None = None,
        *,
        reviewed: bool = False,
        curves: bool = False,
        head_ct_model=_SHIPPED_MODEL,
        progress: Progress | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> R.StudyResult:
        """QA on every complete ROI pair of `rois` (default: `place_rois()`),
        then the study hints and verdict.

        As in the program, a manual ROI1 gets a new automatic ROI2 next to it
        (unless its ROI2 is manual too) - `rois` is updated. `reviewed` = a person
        checked the ROIs; `curves` = keep the MTF/NPS curves; `head_ct_model` =
        the verdict model (default: the shipped one). Raises `InvalidRois` before
        measuring anything if an ROI can't be measured."""
        self._require_segmentation()
        if rois is None:
            rois = self.place_rois()
        self._check_rois(rois)
        volume, recipe = self.volume, self.settings.recipe
        jobs = rois.ready_jobs()
        measurements, failed, contexts = [], [], []
        for done, (index, pair) in enumerate(jobs, start=1):
            if cancel is not None and cancel():
                raise Cancelled(f"measurement stopped after {done - 1}/{len(jobs)} ROI pair(s)")
            effective = rois.effective(index)
            key1, key2 = roi_key(pair, "roi1"), roi_key(pair, "roi2")
            roi1, roi2 = effective[key1], effective[key2]
            image, masks = volume.get_slice_hu(index), self.masks_by_slice.get(index)
            try:
                quality = pipeline.measure_pair(image, masks, roi1, roi2, volume.pixel_spacing_mm, recipe)
            except _MEASUREMENT_FAILURES as exc:  # one bad slice must not stop the study
                logger.info("QA failed on slice %d (%s): %s", index + 1, pair, exc)
                failed.append(R.Failure(self._slice_ref(index), pair, str(exc)))
            else:
                context = pipeline.pair_context(index, pair, image, masks, roi1, roi2, volume.pixel_spacing_mm, quality)
                contexts.append(context)
                measurements.append(R.make_measurement(
                    self._slice_ref(index), pair, roi1, roi2, rois.is_manual(index, key1), rois.is_manual(index, key2),
                    rois.auto_region(index, key2), quality, curves, pipeline.plugin_pair_metrics(context),
                ))
            if progress is not None:
                progress("measurement", done, len(jobs))
        for index, pair in sorted(rois.unplaced):
            if pair not in rois.complete_pairs(index) and roi_key(pair, "roi1") in rois.effective(index):
                failed.append(R.Failure(self._slice_ref(index), pair, "No ROI2 found"))

        hints = pipeline.image_hints(self.masks_by_slice, volume)
        verdict = None
        if recipe == RECIPE_PHYSICAL:
            model = model_files.default_head_ct_model() if head_ct_model is _SHIPPED_MODEL else head_ct_model
            by_pair = {pair: {m.slice.index: m._quality for m in measurements if m.pair == pair} for pair in PAIRS}
            verdict = pipeline.study_verdict(model, by_pair[FALX], by_pair[SEPTUM], hints.artefacts)
        check = pipeline.segmentation_check(self.masks_by_slice, volume.pixel_spacing_mm)
        plugins = pipeline.plugin_study_phase(volume, self.masks_by_slice, tuple(contexts), self.settings.tissue)
        return R.StudyResult(
            label=self.label,
            series=self.series,
            settings=self.settings,
            rois_reviewed=reviewed,
            rois_edited=rois.is_edited(),
            measurements=tuple(measurements),
            failed=tuple(sorted(failed, key=lambda f: (f.slice.index, PAIRS.index(f.pair)))),
            skipped=dict(rois.skipped),
            segmented_slices=len(self.masks_by_slice),
            verdict=R.make_verdict(verdict),
            artefacts=R.make_artefacts(hints.artefacts),
            basal_ganglia=R.make_basal_ganglia(hints.basal_ganglia),
            segmentation_check=R.make_segmentation_check(check),
            plugins=dict(plugin_manager.current().versions),
            plugin_values=plugins.study_values,
            slice_plugin_values=plugins.slice_values,
            plugin_hints=plugins.hints,
            _raw=R._Raw(verdict, hints.artefacts, hints.basal_ganglia, check, plugins.tissue_method),
        )

    def _check_rois(self, rois: RoiPairStore) -> None:
        """Refuse a store of another series and ROIs that can't be measured; re-place
        the automatic ROI2 next to a manual ROI1 (the program's rule)."""
        if rois.series_key not in (None, self.series_key):
            raise InvalidRois("These ROIs were placed on another series")
        for index in rois.slices():
            for pair in PAIRS:
                key1, key2 = roi_key(pair, "roi1"), roi_key(pair, "roi2")
                roi1 = rois.effective(index).get(key1)
                if roi1 is not None and rois.is_manual(index, key1) and not rois.is_manual(index, key2) \
                        and index in self.masks_by_slice and roi_problem(roi1, self.volume.get_slice_hu(index).shape) is None:
                    self._place_roi2(rois, index, pair, roi1)
        problems = []
        for index, pair in rois.ready_jobs():
            if not 0 <= index < self.volume.num_slices:
                problems.append(f"slice {index}: does not exist")
                continue
            shape = self.volume.get_slice_hu(index).shape
            for key, rect in rois.effective(index).items():
                if key.startswith(pair):
                    problem = roi_problem(rect, shape)
                    if problem:
                        problems.append(f"slice {index} {key}: {problem}")
        if problems:
            raise InvalidRois("; ".join(problems[:5]) + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""))

    def _slice_ref(self, index: int) -> R.SliceRef:
        return R.make_slice_ref(self.volume, index)


def load_plugins(folders=None, *, installed: bool = True, disabled=()) -> list:
    """(Re)loads the plug-ins every later run uses: `.py` files and package folders in
    `folders` (default: the plug-in folders - see `bictool.api.plugins`) and, with
    `installed`, Python packages registered for BICTool; names in `disabled` are
    skipped. `load_plugins(folders=[], installed=False)` runs without plug-ins.
    Returns what was found (`plugins.LoadedPlugin`, with status and message)."""
    manager = plugin_manager.load_plugins(None if folders is None else [Path(f) for f in folders], installed, disabled)
    plugin_manager.use(manager)
    return list(manager.plugins)


def loaded_plugins() -> list:
    """The plug-ins in use (loaded on first use if `load_plugins` was not called)."""
    return list(plugin_manager.current().plugins)


def export_formats() -> dict[str, str]:
    """The file formats plug-ins add: {"<plug-in>.<id>": label} (see `write_export`)."""
    return {format_id: fmt.label for format_id, fmt in sorted(plugin_manager.current().export_formats.items())}


def write_export(result: R.StudyResult, path: str | Path, format_id: str) -> Path:
    """Writes `result` in a plug-in's file format (`export_formats()`) to `path` -
    the format's extension is added if missing. Raises `PluginFailed` if the
    format is not available or the plug-in fails."""
    manager = plugin_manager.current()
    fmt = manager.export_formats.get(format_id)
    if fmt is None:
        raise PluginFailed(f"Export format {format_id!r} is not available - see api.export_formats()")
    path = Path(path)
    if path.suffix.lower() != fmt.extension.lower():
        path = path.with_name(path.name + fmt.extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    problem = manager.write_export(format_id, result, path)
    if problem:
        raise PluginFailed(f"Export format {format_id!r} failed: {problem}")
    return path


def open_study(
    path: str | Path, *, series: int = 0, label: str | None = None, settings: Settings | None = None
) -> Study:
    """Series number `series` of `list_series(path)` (default: the program's choice -
    CT, axial, closest to 4 mm) of the study in folder `path`."""
    if not Path(path).exists():
        raise SeriesError(f"{path} does not exist")
    try:
        volume = pipeline.open_series(path, series)
    except (ValueError, IndexError) as exc:
        raise _series_error(path, exc) from None
    return Study(volume, label=label, settings=settings)


def analyse_study(
    path: str | Path,
    *,
    series: int = 0,
    label: str | None = None,
    settings: Settings | None = None,
    model: AnatomyModel | None = None,
    cache_file: str | Path | None = None,
    curves: bool = False,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
) -> R.StudyResult:
    """The whole pipeline, fully automatic: series, segmentation, ROIs, QA, verdict."""
    study = open_study(path, series=series, label=label, settings=settings)
    study.segment(model, cache_file=cache_file, progress=progress, cancel=cancel)
    return study.measure(study.place_rois(progress), curves=curves, progress=progress, cancel=cancel)
