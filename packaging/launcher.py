"""Entry point of the packaged BICTool (portable ZIP, see build_portable.py).

A windowed app has no console: its log and any crash trace go to
%LOCALAPPDATA%\\BICTool\\logs\\bictool.log (the last session; the user manual's
troubleshooting section points there).

    BICTool.exe                          the program
    start /wait "" BICTool.exe --self-test <folder>
                                         checks the installation on one DICOM series
                                         (segmentation with the bundled weights, QA,
                                         study verdict); result in the log, exit code 0/1
"""

import faulthandler
import os
import sys
import tempfile
import time
from pathlib import Path

LOG = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "BICTool" / "logs" / "bictool.log"


def _log_to_file() -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stream = open(LOG, "w", encoding="utf-8", buffering=1)  # noqa: SIM115 - open for the whole session
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream
    faulthandler.enable(stream)


def _self_test(study: Path) -> int:
    from bictool import __version__
    from bictool.application.batch_qa import run_study
    from bictool.infrastructure.config import load_config
    from bictool.infrastructure.logging_config import configure_logging
    from bictool.integration.model_files import load_anatomy_weights

    configure_logging()
    weights = load_config().get("segmentation", {}).get("anatomy_weights_path")
    print(f"BICTool {__version__} self-test on {study}\nweights: {weights}")
    model = None

    def get_model():
        nonlocal model
        if model is None:
            from bictool.domain.segmentation import head_ct_unet

            model = head_ct_unet.build_model()
            load_anatomy_weights(model, weights)
        return model

    start = time.time()
    with tempfile.TemporaryDirectory() as cache:
        qa = run_study(study, study.parent.name, get_model, Path(cache))
    if qa.error:
        print(f"SELF-TEST FAILED: {qa.error}")
        return 1
    verdict = qa.head_ct_verdict.verdict if qa.head_ct_verdict else "not available"
    print(f"segmented slices: {qa.segmented_slices}; study verdict: {verdict}; {time.time() - start:.0f} s")
    print("SELF-TEST PASSED")
    return 0


_log_to_file()

if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
    sys.exit(_self_test(Path(sys.argv[2])))

from bictool.presentation.views.main_window import main  # noqa: E402 - after the log redirection

main()
