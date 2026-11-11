"""Batch QA over a whole dataset, without the GUI: for every study folder,
load the default series (CT, axial, closest to 4 mm - same choice the GUI
makes), segment it with the head-CT U-Net, place the automatic ROIs (septum
pellucidum and dural folds, separately) and run the per-slice QA, then write
one Excel workbook and two CSV files summarising everything.

Usage:
    python -m bictool.application.batch_qa DATASET_DIR [DATASET_DIR ...] --out OUT_DIR
        [--exclude TEXT ...] [--recipe physical|octave]

Each study runs through the public API (`bictool.api.Study` - the same steps
the GUI runs). Each DATASET_DIR (e.g. one scanner's folder) contains one folder per study;
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

from bictool.application import pipeline, plugin_manager
from bictool.application.qa_settings import METHOD_ADAPTIVE, METHOD_ANCHORED, Settings, TissueSettings
from bictool.api.study import Study
from bictool.domain.results.study_quality import DEFAULT_RECIPE, RECIPES
from bictool.domain.results.volume_qa import VolumeQa
from bictool.integration.results_export import export_dataset_csv, export_dataset_excel
from bictool.integration.segmentation_cache import cache_path

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


TISSUE_ADAPTIVE, TISSUE_ANCHORED = "adaptive", "anchored"
TISSUE_METHODS = (TISSUE_ADAPTIVE, TISSUE_ANCHORED)
# The WM/GM method for the white-matter ROI2: the app's Adaptive default, or the anchored prototype.
_TISSUE_SETTINGS = {TISSUE_ADAPTIVE: METHOD_ADAPTIVE, TISSUE_ANCHORED: METHOD_ANCHORED}


def run_study(
    study_dir: Path,
    group: str,
    get_model: Callable[[], object],
    cache_dir: Path,
    recipe: str = DEFAULT_RECIPE,
    tissue_method: str = TISSUE_ADAPTIVE,
    exports: tuple[str, ...] = (),
    export_dir: Path | None = None,
) -> VolumeQa:
    """One study end to end; any failure is recorded on the returned `VolumeQa`.
    `exports` = plug-in file formats written per study into `export_dir`."""
    qa = VolumeQa(group=group, name=study_dir.name)
    try:
        settings = Settings(recipe=recipe, tissue=TissueSettings(method=_TISSUE_SETTINGS[tissue_method]))
        # The batch outputs stay in gitignored folders, so the study keeps its folder name.
        study = Study(pipeline.open_series(study_dir), label=study_dir.name, settings=settings)
        qa.slice_count = study.series.slices
        qa.pixel_spacing_mm = study.series.pixel_spacing_mm
        qa.acquisition_date = study.series.acquisition_date
        qa.series_label = study.series.label

        cache = cache_path(cache_dir, group, study_dir.name, study.volume.slice_paths)
        study.segment(None if cache.is_file() else get_model(), cache_file=cache)
        result = study.measure()
        qa = result.to_volume_qa(group)
        qa.segmentation_confidence = study.confidence_by_slice
        _write_exports(result, exports, export_dir, f"{group} - {study_dir.name}")
    except Exception as exc:  # one broken study must not stop the dataset
        logger.exception("Study %s / %s failed", group, study_dir.name)
        qa.error = f"{type(exc).__name__}: {exc}"
    return qa


def _write_exports(result, exports: tuple[str, ...], export_dir: Path | None, name: str) -> None:
    """The plug-in file formats of one study (a failing plug-in is logged, the batch goes on)."""
    manager = plugin_manager.current()
    for format_id in exports:
        fmt = manager.export_formats.get(format_id)
        if fmt is None or export_dir is None:
            logger.warning("Export format %s is not available", format_id)
            continue
        export_dir.mkdir(parents=True, exist_ok=True)
        problem = manager.write_export(format_id, result, export_dir / f"{name}{fmt.extension}")
        if problem:
            logger.warning("Export format %s failed for %s: %s", format_id, name, problem)


def run_batch(
    dataset_dirs: list[Path],
    out_dir: Path,
    weights_path: str | None = None,
    exclude: list[str] | None = None,
    recipe: str = DEFAULT_RECIPE,
    tissue_method: str = TISSUE_ADAPTIVE,
    exports: tuple[str, ...] = (),
) -> list[VolumeQa]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None

    def get_model():
        nonlocal model
        if model is None:
            model = pipeline.load_anatomy_model(weights_path)
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
            qa = run_study(
                study_dir, group, get_model, out_dir / "cache", recipe, tissue_method, exports, out_dir / "plugin_exports"
            )
            volumes.append(qa)
            logger.info("    done in %.0f s%s", time.time() - start, f" - ERROR {qa.error}" if qa.error else "")

    export_dataset_excel(volumes, out_dir / "dataset_qa.xlsx")
    export_dataset_csv(
        volumes, out_dir / "dataset_qa_summary.csv", out_dir / "dataset_qa_slices.csv", out_dir / "dataset_qa_plugins.csv"
    )
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
    parser.add_argument(
        "--plugin-export", action="append", default=[], metavar="ID",
        help="also write each study in a plug-in file format (\"<plug-in>.<id>\") into OUT/plugin_exports",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_batch(args.datasets, args.out, args.weights, args.exclude, args.recipe, args.tissue, tuple(args.plugin_export))


if __name__ == "__main__":
    main()
