"""Where the files shipped with BICTool are: next to BICTool.exe in the
packaged app (portable ZIP, `packaging/`), in the repository when run from
source.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[3]
MANUAL_FILES = {"en": "BICTool_User_Manual_EN.pdf", "de": "BICTool_Benutzerhandbuch_DE.pdf"}
TECHNICAL_REFERENCE_FILE = "BICTool_Technical_Reference_EN.pdf"
HELP_INDEX_FILE = "help_index.json"


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_folder() -> Path:
    """The packaged app's folder, or the repository root when run from source."""
    return Path(sys.executable).parent if is_packaged() else _REPOSITORY


def _manual_folder() -> Path:
    return app_folder() / "manual" if is_packaged() else _REPOSITORY / "docs" / "manual" / "build"


def manual_path(language: str) -> Path:
    """The user manual PDF in `language` ("en"/"de") - it may not exist in a source checkout."""
    return _manual_folder() / MANUAL_FILES[language]


def technical_reference_path() -> Path:
    """The technical reference PDF (English) - it may not exist in a source checkout."""
    return _manual_folder() / TECHNICAL_REFERENCE_FILE


def help_folder() -> Path:
    """The help window's pages and `HELP_INDEX_FILE` (built by docs/manual/build_manual.py)."""
    return app_folder() / "help" if is_packaged() else _REPOSITORY / "docs" / "manual" / "build" / "help"


def notices_path() -> Path:
    """Third-party notices: the generated file in the package, the template in the repository."""
    return app_folder() / "THIRD_PARTY_NOTICES.txt" if is_packaged() else _REPOSITORY / "packaging" / "THIRD_PARTY_NOTICES.md"
