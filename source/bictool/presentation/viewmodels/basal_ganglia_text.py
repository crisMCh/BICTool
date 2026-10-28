"""Text for the basal-ganglia visibility (`domain.metrics.basal_ganglia`) on the
Results page and per slice. Provisional and descriptive - not part of the
verdict. Qt-free.
"""

from __future__ import annotations

from bictool.domain.metrics.basal_ganglia import (
    CLEARLY,
    HARDLY,
    NOT_PLAUSIBLE,
    PLAUSIBLE_DIFFERENCE_HU,
    BasalGangliaSlice,
)
from bictool.presentation.viewmodels.display_text import format_slice_ranges

BASAL_GANGLIA_NOTE = (
    "Basal ganglia visibility: provisional measure (border contrast of the CNN-labelled lentiform and caudate "
    "nuclei against the internal capsule), checked against a non-clinician's ratings - a hint, "
    "not part of the verdict. A CNN label whose contrast to the internal capsule is outside "
    f"{PLAUSIBLE_DIFFERENCE_HU[0]:g}-{PLAUSIBLE_DIFFERENCE_HU[1]:g} HU is not counted (provisional limit)."
)


def basal_ganglia_basics(slices: dict[int, BasalGangliaSlice] | None) -> list[str]:
    """One or two plain lines for the study summary (none without labelled nuclei)."""
    if not slices:
        return []
    visible = [index for index, s in slices.items() if s.visible]
    clearly = [index for index, s in slices.items() if s.level == CLEARLY]
    hardly = [index for index, s in slices.items() if s.level == HARDLY]
    implausible = [index for index, s in slices.items() if s.level == NOT_PLAUSIBLE]
    if visible:
        line = f"Basal ganglia (provisional): visible {_on_slices(visible)}"
        if clearly:
            line += f" (clearly {_on_slices(clearly)})"
        line += "."
    else:
        line = "Basal ganglia (provisional): not clearly visible on any slice"
        line += f" (hardly {_on_slices(hardly)})." if hardly else "."
    lines = [line]
    if implausible:
        lines.append(
            f"Basal ganglia: CNN label without matching image contrast {_on_slices(implausible)} - not counted."
        )
    return lines


def basal_ganglia_details(slices: dict[int, BasalGangliaSlice] | None) -> list[str]:
    if not slices:
        return []
    values = ", ".join(f"{index + 1}: {s.border:.2f} ({s.level})" for index, s in sorted(slices.items()))
    return [f"Basal ganglia border contrast per slice: {values}.", BASAL_GANGLIA_NOTE]


def basal_ganglia_note(slices: dict[int, BasalGangliaSlice] | None, index: int) -> str:
    """Short per-slice note ("" if the slice has no labelled nuclei)."""
    if not slices or index not in slices:
        return ""
    return f"basal ganglia {slices[index].level}"


def _on_slices(indices) -> str:
    indices = sorted(indices)
    return f"on slice{'s' if len(indices) > 1 else ''} {format_slice_ranges(indices)}"
