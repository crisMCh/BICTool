"""Named ROI pairs per slice - the ROI state behind the ROI Selection box.

Every slice can carry a **falx pair** (ROI1 on the falx / dural folds, ROI2 in
the white matter next to it) and a **septum pair** (ROI1 on the septum
pellucidum, ROI2 in the ventricle), measured side by side
(docs/qa_method/head_ct_classifier.md, decision D11). Each of the four slots
(`falx.roi1`, `falx.roi2`, `septum.roi1`, `septum.roi2`) holds:

* an automatic ROI (placed by `domain.roi_selection`),
* optionally a manual ROI (hand-drawn and saved into the slot, or an ROI the
  user moved) - it wins over the automatic one,
* optionally a "deleted" mark - the slot shows nothing on that slice until
  the user saves a new ROI into it or restores the slice.

`restore_slice` drops a slice's manual ROIs and deletions, so its automatic
ROIs show again. Pure Python - no Qt, testable on its own.
"""

from __future__ import annotations

from bictool.presentation.viewmodels.roi_selection import RoiRect

FALX = "falx"
SEPTUM = "septum"
PAIRS = (FALX, SEPTUM)
SLOTS = ("roi1", "roi2")
PAIR_LABELS = {FALX: "Falx", SEPTUM: "Septum"}


def roi_key(pair: str, slot: str) -> str:
    return f"{pair}.{slot}"


def split_key(key: str) -> tuple[str, str]:
    pair, slot = key.split(".")
    return pair, slot


ROI_KEYS = tuple(roi_key(pair, slot) for pair in PAIRS for slot in SLOTS)


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


class RoiPairStore:
    def __init__(self) -> None:
        self._auto: dict[int, dict[str, RoiRect]] = {}
        self._manual: dict[int, dict[str, RoiRect]] = {}
        self._deleted: dict[int, set[str]] = {}

    # --- automatic ROIs ---

    def set_auto(self, slice_index: int, key: str, rect: RoiRect | None) -> None:
        """The automatic ROI of a slot (None removes it)."""
        if rect is None:
            self._auto.get(slice_index, {}).pop(key, None)
        else:
            self._auto.setdefault(slice_index, {})[key] = rect

    def auto(self, slice_index: int, key: str) -> RoiRect | None:
        return self._auto.get(slice_index, {}).get(key)

    def clear_auto(self, pair: str | None = None) -> None:
        """Drops the automatic ROIs of `pair` (or all pairs) on every slice."""
        if pair is None:
            self._auto = {}
            return
        for rois in self._auto.values():
            for slot in SLOTS:
                rois.pop(roi_key(pair, slot), None)

    # --- the user's edits ---

    def set_manual(self, slice_index: int, key: str, rect: RoiRect) -> None:
        """A hand-drawn ROI saved into `key`, or an ROI the user moved - wins
        over the automatic one and undoes a deletion of that slot."""
        self._manual.setdefault(slice_index, {})[key] = rect
        self._deleted.get(slice_index, set()).discard(key)

    def is_manual(self, slice_index: int, key: str) -> bool:
        return key in self._manual.get(slice_index, {})

    def delete(self, slice_index: int, key: str) -> None:
        """Removes the slot's ROI from this slice entirely (manual and automatic)."""
        self._manual.get(slice_index, {}).pop(key, None)
        self._deleted.setdefault(slice_index, set()).add(key)

    def restore_slice(self, slice_index: int) -> None:
        """Back to the automatic ROIs on this slice: manual ROIs and deletions dropped."""
        self._manual.pop(slice_index, None)
        self._deleted.pop(slice_index, None)

    def has_edits(self, slice_index: int) -> bool:
        return bool(self._manual.get(slice_index)) or bool(self._deleted.get(slice_index))

    def clear(self) -> None:
        self._auto, self._manual, self._deleted = {}, {}, {}

    # --- what is shown / measured ---

    def effective(self, slice_index: int) -> dict[str, RoiRect]:
        """The slice's ROIs per slot: manual over automatic, deleted slots left out."""
        deleted = self._deleted.get(slice_index, set())
        rois = {}
        for key in ROI_KEYS:
            if key in deleted:
                continue
            rect = self._manual.get(slice_index, {}).get(key) or self._auto.get(slice_index, {}).get(key)
            if rect is not None:
                rois[key] = rect
        return rois

    def complete_pairs(self, slice_index: int) -> list[str]:
        """Pairs with both ROI1 and ROI2 on this slice - the ones QA can measure."""
        rois = self.effective(slice_index)
        return [pair for pair in PAIRS if all(roi_key(pair, slot) in rois for slot in SLOTS)]

    def slices(self) -> list[int]:
        return sorted(set(self._auto) | set(self._manual))

    def ready_jobs(self) -> list[tuple[int, str]]:
        """Every (slice, pair) with a complete pair, in slice order - one QA job each."""
        return [(index, pair) for index in self.slices() for pair in self.complete_pairs(index)]
