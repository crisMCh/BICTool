"""Batch QA over a whole dataset, without the GUI: for every study folder,
load the default series (CT, axial, closest to 4 mm - same choice the GUI
makes), segment it with the head-CT U-Net, place the automatic ROIs (septum
pellucidum and dural folds, separately) and run the per-slice QA, then write
one Excel workbook and two CSV files summarising everything.

Usage:
    python -m bictool.application.batch_qa DATASET_DIR [DATASET_DIR ...] --out OUT_DIR
        [--exclude TEXT ...] [--recipe physical|octave]

Each DATASET_DIR (e.g. one scanner's folder) contains one folder per study;
its name labels the dataset in the output. A wrapper folder holding nothing
but a single subfolder (as zip extraction tends to produce) is looked
through. `--exclude` skips every study whose folder name contains the given
text. Segmentation results are cached in OUT_DIR/cache (per study and
series), so a rerun skips the CNN. Output is derived from patient data - keep OUT_DIR out of version
control.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bictool.domain.artefacts.artefact_flags import evaluate_artefact_flags
from bictool.domain.metrics.basal_ganglia import evaluate_basal_ganglia
from bictool.domain.results.head_ct_verdict import evaluate_study_results
from bictool.domain.results.study_quality import DEFAULT_RECIPE, RECIPE_PHYSICAL, RECIPES
from bictool.domain.results.volume_qa import VolumeQa, evaluate_volume_qa
from bictool.domain.segmentation.brain_tissue_adaptive import (
    DEFAULT_PLAUSIBLE_RANGE,
    DEFAULT_SMOOTHING_MM,
    classify_brain_tissue_adaptive,
)
from bictool.domain.segmentation.brain_tissue_anchored import classify_brain_tissue_anchored, fit_anchored_model
from bictool.domain.segmentation.segmentation_brain_mask import brain_mask_from_segmentation
from bictool.domain.segmentation.tissue_references import estimate_tissue_references
from bictool.infrastructure.config import load_config
from bictool.integration.dicom_file_loader import load_dicom_volume
from bictool.integration.dicom_to_nifti import build_dicom_nifti_volume
from bictool.domain.roi_selection.structured_roi import DURAL_FOLDS, SEPTUM_PELLUCIDUM
from bictool.integration.model_files import default_chest_svm_model, default_head_ct_model, load_anatomy_weights
from bictool.integration.results_export import export_dataset_csv, export_dataset_excel
from bictool.integration.segmentation_cache import cache_path, load_segmentation, save_segmentation

logger = logging.getLogger(__name__)


def find_studies(dataset_dir: Path) -> list[Path]:
    """The study folders in `dataset_dir`, looking through single-subfolder wrappers."""
    folder = Path(dataset_dir)
    while True:
        entries = list(folder.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            folder = entries[0]
        else:
            break
    return sorted(entry for entry in entries if entry.is_dir())


def _segment_volume(volume, get_model: Callable[[], object]) -> dict[int, dict[str, np.ndarray]]:
    """The GUI's Segment Volume, synchronously: NIfTI detour, U-Net per slice,
    masks mapped back to DICOM pixel space."""
    from bictool.domain.segmentation.head_ct_unet import segment_slice

    nifti = build_dicom_nifti_volume(volume.slice_paths, volume.get_slice_hu)
    masks_by_slice = {}
    for index in range(volume.num_slices):
        if not nifti.is_segmentable(index):
            continue
        result = nifti.remap_result_to_dicom_space(segment_slice(get_model(), nifti.get_nifti_slice_hu(index)))
        masks_by_slice[index] = result.masks
    return masks_by_slice


TISSUE_ADAPTIVE, TISSUE_ANCHORED = "adaptive", "anchored"
TISSUE_METHODS = (TISSUE_ADAPTIVE, TISSUE_ANCHORED)


def _tissue_classifier(masks_by_slice, volume, method=TISSUE_ADAPTIVE):
    """WM/GM method for ROI2 placement: the app's Adaptive default (1 mm
    smoothing, the study's own tissue range - fixed range until the basal
    ganglia exist), or the anchored prototype (Adaptive where it has no model)."""
    references = estimate_tissue_references(masks_by_slice, volume.get_slice_hu)
    plausible = references.plausible_range() if references is not None else DEFAULT_PLAUSIBLE_RANGE
    sigma = DEFAULT_SMOOTHING_MM / volume.pixel_spacing_mm
    model = None
    if method == TISSUE_ANCHORED and references is not None:
        brains = brain_mask_from_segmentation(masks_by_slice, volume.num_slices)
        model = fit_anchored_model(masks_by_slice, volume.get_slice_hu, brains, volume.pixel_spacing_mm, references)

    def classify(slice_hu, brain_mask, slice_index=None):
        if model is not None and slice_index is not None:
            return classify_brain_tissue_anchored(slice_hu, brain_mask, model, slice_index)
        return classify_brain_tissue_adaptive(slice_hu, brain_mask, plausible_range=plausible, smoothing_sigma_px=sigma)

    return classify


def run_study(
    study_dir: Path,
    group: str,
    get_model: Callable[[], object],
    cache_dir: Path,
    recipe: str = DEFAULT_RECIPE,
    tissue_method: str = TISSUE_ADAPTIVE,
) -> VolumeQa:
    """One study end to end; any failure is recorded on the returned `VolumeQa`."""
    qa = VolumeQa(group=group, name=study_dir.name)
    try:
        volume = load_dicom_volume(study_dir)
        qa.slice_count = volume.num_slices
        qa.pixel_spacing_mm = volume.pixel_spacing_mm
        qa.acquisition_date = volume.acquisition_date
        qa.series_label = f"{volume.series_description or ''} ({volume.num_slices} slices, {volume.slice_thickness_mm} mm)".strip()

        cache = cache_path(cache_dir, group, study_dir.name, volume.slice_paths)
        if cache.is_file():
            masks_by_slice = load_segmentation(cache)
        else:
            masks_by_slice = _segment_volume(volume, get_model)
            save_segmentation(cache, masks_by_slice)
        qa.segmented_slices = len(masks_by_slice)
        qa.artefact_flags = evaluate_artefact_flags(
            masks_by_slice, volume.get_slice_hu, volume.num_slices, volume.pixel_spacing_mm
        )
        qa.basal_ganglia = evaluate_basal_ganglia(masks_by_slice, volume.get_slice_hu)

        qa.structures = evaluate_volume_qa(
            masks_by_slice,
            volume.get_slice_hu,
            volume.num_slices,
            volume.pixel_spacing_mm,
            _tissue_classifier(masks_by_slice, volume, tissue_method),
            recipe=recipe,
            svm_model=default_chest_svm_model(),
        )
        if recipe == RECIPE_PHYSICAL:
            # The study verdict comes from the falx; the septum only feeds the plausibility check.
            falx, septum = qa.structures.get(DURAL_FOLDS), qa.structures.get(SEPTUM_PELLUCIDUM)
            qa.head_ct_verdict = evaluate_study_results(
                default_head_ct_model(), falx.results if falx else {}, septum.results if septum else None
            )
    except Exception as exc:  # one broken study must not stop the dataset
        logger.exception("Study %s / %s failed", group, study_dir.name)
        qa.error = f"{type(exc).__name__}: {exc}"
    return qa


def run_batch(
    dataset_dirs: list[Path],
    out_dir: Path,
    weights_path: str | None = None,
    exclude: list[str] | None = None,
    recipe: str = DEFAULT_RECIPE,
    tissue_method: str = TISSUE_ADAPTIVE,
) -> list[VolumeQa]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_path or load_config().get("segmentation", {}).get("anatomy_weights_path")
    model = None

    def get_model():
        nonlocal model
        if model is None:
            if not weights_path:
                raise RuntimeError("No anatomy segmentation weights configured (BICTOOL_HEADCT_WEIGHTS)")
            from bictool.domain.segmentation import head_ct_unet

            model = head_ct_unet.build_model()
            load_anatomy_weights(model, weights_path)
        return model

    volumes = []
    for dataset_dir in dataset_dirs:
        group = Path(dataset_dir).name
        for study_dir in find_studies(Path(dataset_dir)):
            if any(text in study_dir.name for text in exclude or []):
                logger.info("=== %s / %s - excluded", group, study_dir.name)
                continue
            start = time.time()
            logger.info("=== %s / %s", group, study_dir.name)
            qa = run_study(study_dir, group, get_model, out_dir / "cache", recipe, tissue_method)
            volumes.append(qa)
            logger.info("    done in %.0f s%s", time.time() - start, f" - ERROR {qa.error}" if qa.error else "")

    export_dataset_excel(volumes, out_dir / "dataset_qa.xlsx")
    export_dataset_csv(volumes, out_dir / "dataset_qa_summary.csv", out_dir / "dataset_qa_slices.csv")
    return volumes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="+", type=Path, help="dataset folders, each holding one folder per study")
    parser.add_argument("--out", type=Path, required=True, help="output folder (Excel, CSVs, segmentation cache)")
    parser.add_argument("--weights", help="anatomy U-Net weights (default: BICTool config / BICTOOL_HEADCT_WEIGHTS)")
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="TEXT", help="skip studies whose folder name contains TEXT"
    )
    parser.add_argument(
        "--recipe",
        choices=RECIPES,
        default=DEFAULT_RECIPE,
        help="QA recipe: physical (default) or the legacy Octave one (docs/qa_method/physical_qa_recipe.md)",
    )
    parser.add_argument(
        "--tissue",
        choices=TISSUE_METHODS,
        default=TISSUE_ADAPTIVE,
        help="WM/GM method for the white-matter ROI2: adaptive (app default) or anchored (prototype)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_batch(args.datasets, args.out, args.weights, args.exclude, args.recipe, args.tissue)


if __name__ == "__main__":
    main()
