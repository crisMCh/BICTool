"""The ROI pairs and slots in words (menus, ROI table, Results).

The pairs, slots and the store of a study's ROIs (`RoiPairStore`: automatic,
manual and deleted ROIs per slice) are part of the public API
(`bictool.application.roi_store`) and re-exported here. Pure Python - no Qt.
"""

from __future__ import annotations

from bictool.application.roi_store import (  # noqa: F401 (re-exported)
    FALX,
    PAIRS,
    ROI_KEYS,
    SEPTUM,
    SLOTS,
    RoiPairStore,
    roi_key,
    split_key,
)

PAIR_LABELS = {FALX: "Falx", SEPTUM: "Septum"}


# What each slot is, in words (menus, ROI table): ROI1 sits on the structure,
# ROI2 in the homogeneous tissue next to it.
ROLE_LABELS = {
    "falx.roi1": "Falx",
    "falx.roi2": "White matter (next to falx)",
    "septum.roi1": "Septum",
    "septum.roi2": "Ventricle (next to septum)",
}
ROLE_SHORT = {"falx.roi1": "Falx", "falx.roi2": "White matter", "septum.roi1": "Septum", "septum.roi2": "Ventricle"}


def key_label(key: str) -> str:
    """'falx.roi1' -> 'Falx ROI1'."""
    pair, slot = split_key(key)
    return f"{PAIR_LABELS[pair]} {slot.upper()}"
