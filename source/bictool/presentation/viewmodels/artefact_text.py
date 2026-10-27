"""Text for the possible-artefact flags (`domain.artefacts.artefact_flags`) on
the Results page: plain lines for the study summary, numbers for the details.
They are hints where to look, not part of the verdict - except metal streaks
in the brain, which make the verdict Not assessable (metal rule). Qt-free.
"""

from __future__ import annotations

from bictool.domain.artefacts.artefact_flags import (
    BAND_DEPTH_HU,
    METAL_HU,
    METAL_STREAK_REACH_MM,
    METAL_STREAK_SHARE,
    STREAK_DEVIATION_HU,
    ArtefactFlags,
)
from bictool.presentation.viewmodels.display_text import format_slice_ranges

BAND_NOTE = (
    "Dark band across the brainstem: provisional measure (checked against non-expert ratings of 16 studies) - "
    "a hint where to look, not part of the verdict."
)


def artefact_basics(flags: ArtefactFlags | None) -> list[str]:
    """One plain line per flag that applies (none if nothing was found)."""
    if flags is None:
        return []
    lines = []
    if flags.metal_streak_slices:
        lines.append(
            f"Artefacts (metal): {_on_slices(flags.metal_slices)} - streaks reach the brain "
            f"{_on_slices(flags.metal_streak_slices)}."
        )
    elif flags.metal_slices:
        lines.append(f"Artefacts (metal): {_on_slices(flags.metal_slices)} - no streaks measured in the brain.")
    if flags.band_slices:
        lines.append(
            f"Artefacts (posterior fossa): possible dark band across the brainstem {_on_slices(flags.band_slices)}."
        )
    return lines


def slice_artefact_notes(flags: ArtefactFlags | None, index: int) -> list[str]:
    """Short notes for one slice (study table, Per-slice view)."""
    return list(flags.on_slice(index)) if flags is not None else []


def _on_slices(indices) -> str:
    indices = list(indices)
    return f"on slice{'s' if len(indices) > 1 else ''} {format_slice_ranges(indices)}"


def artefact_details(flags: ArtefactFlags | None) -> list[str]:
    """The numbers behind the basics."""
    if flags is None:
        return []
    details = []
    if flags.metal_slices:
        details.append(f"Metal: pixels above {METAL_HU:.0f} HU, highest {flags.metal_max_hu:.0f} HU.")
        if flags.metal_streak_shares:
            shares = ", ".join(f"{index + 1}: {100 * share:.0f} %" for index, share in sorted(flags.metal_streak_shares.items()))
            details.append(
                f"Metal streaks in the brain: share of brain tissue within {METAL_STREAK_REACH_MM:g} mm of the metal "
                f"that differs by more than {STREAK_DEVIATION_HU:g} HU from the brain median, per slice "
                f"(streaks from {100 * METAL_STREAK_SHARE:.0f} %, provisional): {shares}."
            )
    else:
        details.append(f"Metal: none (no pixels above {METAL_HU:.0f} HU).")
    if flags.band_depths:
        depths = ", ".join(f"{index + 1}: {depth:.1f}" for index, depth in sorted(flags.band_depths.items()))
        details.append(
            f"Brainstem dark band, depth in HU per posterior-fossa slice (flag from {BAND_DEPTH_HU:g} HU): {depths}."
        )
        details.append(BAND_NOTE)
    else:
        details.append("Brainstem dark band: no posterior-fossa slice with a segmented brainstem.")
    return details
