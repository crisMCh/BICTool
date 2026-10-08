"""Small display/text helpers for the main window (slice ranges, DICOM
dates, file names). Qt-free, so they are testable without a GUI.
"""

from __future__ import annotations

import re
from datetime import datetime


def format_slice_ranges(slice_indices: list[int]) -> str:
    """Sorted 0-based indices -> 1-based display ranges, e.g. [19, 20, 21, 30] -> "20–22, 31"."""
    if not slice_indices:
        return ""
    parts = []
    start = previous = slice_indices[0]
    for index in slice_indices[1:] + [None]:
        if index is not None and index == previous + 1:
            previous = index
            continue
        parts.append(f"{start + 1}" if start == previous else f"{start + 1}–{previous + 1}")
        if index is not None:
            start = previous = index
    return ", ".join(parts)


def format_dicom_date(raw: str | None) -> str:
    """DICOM `DA` values are `YYYYMMDD` with no separators - reformatted
    to `DD-Month-YYYY` for display (matching MicroDicom's overlay, e.g.
    "11-September-2025"). Falls back to the raw string for anything that
    doesn't match the expected shape rather than raising on odd data.
    """
    if not raw or len(raw) != 8 or not raw.isdigit():
        return raw or "—"
    try:
        return datetime.strptime(raw, "%Y%m%d").strftime("%d-%B-%Y")
    except ValueError:
        return raw


def safe_file_stem(series_label: str) -> str:
    """Filesystem-safe stem from a series label. The label has "·"/"×" and
    spaces from its human-readable form (see `dicom_file_loader._build_candidate`);
    runs of "_" are collapsed so e.g. "CT · 1 slice(s) · 512×434" doesn't
    become "CT___1_slice_s_..._".
    """
    return re.sub(r"_+", "_", "".join(c if c.isalnum() else "_" for c in series_label)).strip("_")
