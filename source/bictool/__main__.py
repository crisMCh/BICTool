"""`python -m bictool`: without arguments the program starts; with a command,
the API runs from the command line:

    python -m bictool series STUDY_FOLDER
    python -m bictool analyse STUDY_FOLDER [--series N] [--label TEXT] [--out result.json]
        [--curves] [--uids] [--weights FILE] [--cache FILE]

`analyse` prints the verdict and writes the result as JSON (`--out`, else to
stdout). Exit code 0 = done, 2 = the study could not be analysed.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _series(args) -> int:
    from bictool import api

    for choice in api.list_series(args.folder):
        print(f"{choice.index}: {choice.label}")
    return 0


def _analyse(args) -> int:
    from bictool import api

    def progress(stage: str, done: int, total: int) -> None:
        print(f"\r{stage} {done}/{total}   ", end="", file=sys.stderr, flush=True)

    model = api.load_anatomy_model(args.weights) if args.weights else None
    result = api.analyse_study(
        args.folder, series=args.series, label=args.label, model=model, cache_file=args.cache,
        curves=args.curves, progress=progress,
    )
    print(file=sys.stderr)
    text = result.to_json(include_curves=args.curves, include_dicom_uids=args.uids)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    verdict = result.verdict.verdict if result.verdict else "no verdict"
    print(f"{result.label}: {verdict} ({len(result.measurements)} measurement(s), {len(result.failed)} failed)",
          file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from bictool.presentation.views.main_window import main as start_program

        start_program()
        return 0
    parser = argparse.ArgumentParser(prog="python -m bictool", description="BICTool - head-CT image quality")
    commands = parser.add_subparsers(dest="command", required=True)
    series = commands.add_parser("series", help="list the series of a study folder")
    series.add_argument("folder")
    series.set_defaults(run=_series)
    analyse = commands.add_parser("analyse", help="analyse one series of a study folder")
    analyse.add_argument("folder")
    analyse.add_argument("--series", type=int, default=0, help="series number from `series` (default: the program's choice)")
    analyse.add_argument("--label", help="name of the study in the result (default: 'study - <series>')")
    analyse.add_argument("--out", help="JSON file to write (default: standard output)")
    analyse.add_argument("--curves", action="store_true", help="include the MTF/NPS curves")
    analyse.add_argument("--uids", action="store_true", help="include the DICOM UIDs (they link to the patient)")
    analyse.add_argument("--weights", help="anatomy CNN weights (default: the configured ones)")
    analyse.add_argument("--cache", help="segmentation cache file (read if it exists, else written)")
    analyse.set_defaults(run=_analyse)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from bictool.api.errors import BictoolError

    try:
        return args.run(args)
    except BictoolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
