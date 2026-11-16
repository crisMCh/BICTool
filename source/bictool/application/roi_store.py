"""ROIs: the rectangle type, the two ROI pairs per slice and the store of a
study's ROIs (automatic ones, manual ones that win over them, deletions).

Every slice can carry a **falx pair** (ROI1 on the falx / dural folds, ROI2 in
the white matter next to it) and a **septum pair** (ROI1 on the septum
pellucidum, ROI2 in the ventricle), measured side by side
(docs/qa_method/head_ct_classifier.md, decision D11). Each of the four slots
(`falx.roi1`, `falx.roi2`, `septum.roi1`, `septum.roi2`) holds an automatic
ROI, optionally a manual ROI (it wins over the automatic one) and optionally a
"deleted" mark. Public through `bictool.api`. Pure Python/NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bictool.domain.roi_selection.structured_roi import DURAL_FOLDS, SEPTUM_PELLUCIDUM


@dataclass(frozen=True)
class RoiRect:
    """An ROI rectangle in image-pixel coordinates (x0 <= x1, y0 <= y1)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def roi_problem(rect: RoiRect, image_shape: tuple[int, int]) -> str | None:
    """Why `rect` can't be measured on an image of `image_shape` (rows, columns), or None.
    The NPS needs a square ROI; the automatic ROIs and the program's drawn ones always are."""
    values = (rect.x0, rect.y0, rect.x1, rect.y1)
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
        return f"ROI coordinates must be whole pixels: {rect}"
    if rect.width <= 0 or rect.height <= 0:
        return f"ROI must have x0 < x1 and y0 < y1: {rect}"
    if rect.x0 < 0 or rect.y0 < 0 or rect.x1 > image_shape[1] or rect.y1 > image_shape[0]:
        return f"ROI lies outside the {image_shape[1]}x{image_shape[0]} image: {rect}"
    if rect.width != rect.height:
        return f"ROI must be square (the noise power spectrum needs it): {rect}"
    return None


def candidate_to_rect(candidate) -> RoiRect:
    """A domain `roi_selection.structured_roi.RoiCandidate` (row/col corners)
    as the View's `RoiRect` (x/y corners)."""
    return RoiRect(candidate.col0, candidate.row0, candidate.col1, candidate.row1)


def crop_roi(pixels: np.ndarray, rect: RoiRect) -> np.ndarray:
    """Crops `pixels` (2D array, row-major: [y, x]) to `rect`."""
    return pixels[rect.y0 : rect.y1, rect.x0 : rect.x1]


FALX = "falx"
SEPTUM = "septum"
PAIRS = (FALX, SEPTUM)
SLOTS = ("roi1", "roi2")


def roi_key(pair: str, slot: str) -> str:
    return f"{pair}.{slot}"


def split_key(key: str) -> tuple[str, str]:
    pair, slot = key.split(".")
    return pair, slot


ROI_KEYS = tuple(roi_key(pair, slot) for pair in PAIRS for slot in SLOTS)


def _check_key(key: str) -> None:
    if key not in ROI_KEYS:
        raise ValueError(f"Unknown ROI slot {key!r} - one of {ROI_KEYS}")


# ROI pair -> the segmentation structure its ROI1 sits on.
PAIR_STRUCTURES = {FALX: DURAL_FOLDS, SEPTUM: SEPTUM_PELLUCIDUM}


class RoiPairStore:
    """A series' ROIs per slice and slot (`roi_key`), and what the automatic
    placement found: the region of each automatic ROI2, the pairs whose ROI2
    search found nothing (`unplaced`) and the slices left out for too little
    brain (`skipped`). `series_key` names the series the automatic ROIs were
    placed on (None for a store built by hand)."""

    def __init__(self, series_key: str | None = None) -> None:
        self.series_key = series_key
        self._auto: dict[int, dict[str, RoiRect]] = {}
        self._manual: dict[int, dict[str, RoiRect]] = {}
        self._deleted: dict[int, set[str]] = {}
        self._regions: dict[tuple[int, str], str] = {}
        self.unplaced: set[tuple[int, str]] = set()          # (slice, pair): no automatic ROI2 found
        self.skipped: dict[str, tuple[int, ...]] = {}       # pair -> structure found, too little brain

    # --- automatic ROIs ---

    def set_auto(self, slice_index: int, key: str, rect: RoiRect | None, region: str | None = None) -> None:
        """The automatic ROI of a slot (None removes it); `region` = where an
        automatic ROI2 lies (`homogeneous_roi.VENTRICLE` / `WHITE_MATTER` / `FALLBACK`)."""
        _check_key(key)
        if rect is None:
            self._auto.get(slice_index, {}).pop(key, None)
            self._regions.pop((slice_index, key), None)
        else:
            self._auto.setdefault(slice_index, {})[key] = rect
            if region is None:
                self._regions.pop((slice_index, key), None)
            else:
                self._regions[(slice_index, key)] = region

    def auto_region(self, slice_index: int, key: str) -> str | None:
        """Where the automatic ROI of a slot lies (None if unknown)."""
        return self._regions.get((slice_index, key))

    def auto(self, slice_index: int, key: str) -> RoiRect | None:
        return self._auto.get(slice_index, {}).get(key)

    def clear_auto(self, pair: str | None = None) -> None:
        """Drops the automatic ROIs of `pair` (or all pairs) on every slice."""
        if pair is None:
            self._auto, self._regions = {}, {}
            return
        for index, rois in self._auto.items():
            for slot in SLOTS:
                rois.pop(roi_key(pair, slot), None)
                self._regions.pop((index, roi_key(pair, slot)), None)

    # --- the user's edits ---

    def set_manual(self, slice_index: int, key: str, rect: RoiRect) -> None:
        """A hand-drawn ROI saved into `key`, or an ROI the user moved - wins
        over the automatic one and undoes a deletion of that slot."""
        _check_key(key)
        self._manual.setdefault(slice_index, {})[key] = rect
        self._deleted.get(slice_index, set()).discard(key)

    def is_manual(self, slice_index: int, key: str) -> bool:
        return key in self._manual.get(slice_index, {})

    def delete(self, slice_index: int, key: str) -> None:
        """Removes the slot's ROI from this slice entirely (manual and automatic)."""
        _check_key(key)
        self._manual.get(slice_index, {}).pop(key, None)
        self._deleted.setdefault(slice_index, set()).add(key)

    def restore_slice(self, slice_index: int) -> None:
        """Back to the automatic ROIs on this slice: manual ROIs and deletions dropped."""
        self._manual.pop(slice_index, None)
        self._deleted.pop(slice_index, None)

    def has_edits(self, slice_index: int) -> bool:
        return bool(self._manual.get(slice_index)) or bool(self._deleted.get(slice_index))

    def is_edited(self) -> bool:
        """Some slice has a manual ROI or a deletion."""
        return any(self.has_edits(index) for index in set(self._manual) | set(self._deleted))

    def clear(self) -> None:
        self._auto, self._manual, self._deleted, self._regions = {}, {}, {}, {}
        self.unplaced, self.skipped = set(), {}

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
