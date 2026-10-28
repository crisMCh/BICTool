"""Reads the help window's index (`help_index.json`, written by
docs/manual/build_manual.py next to the help pages): which documents exist,
their chapters and sections, and each section's plain text for the search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HelpSection:
    id: str        # the heading's anchor in the page
    title: str
    level: int     # 2 = chapter, 3 = section
    text: str


@dataclass(frozen=True)
class HelpDocument:
    key: str       # "user_guide" | "technical_reference"
    language: str  # "en" | "de"
    title: str
    path: Path     # the HTML page
    sections: tuple[HelpSection, ...]


def load_help_documents(folder: Path, index_file: str) -> list[HelpDocument]:
    """The documents listed in `folder / index_file`; empty when the help is
    not built or the index cannot be read (the window then says so)."""
    index = folder / index_file
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
        return [
            HelpDocument(
                key=entry["key"],
                language=entry["language"],
                title=entry["title"],
                path=folder / entry["file"],
                sections=tuple(HelpSection(s["id"], s["title"], int(s["level"]), s["text"]) for s in entry["sections"]),
            )
            for entry in data["documents"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Help index not readable (%s): %s", index, exc)
        return []
