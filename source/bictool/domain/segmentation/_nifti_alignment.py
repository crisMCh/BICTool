"""Pure image-analysis logic for aligning a DICOM series' slices with a
NIfTI volume derived from the same scan: discovering the in-plane
orientation transform between the two representations, matching each
DICOM slice to its NIfTI counterpart, and mapping segmentation results
computed in NIfTI space back onto each DICOM slice's own pixel grid.

Deliberately free of file I/O, pydicom, and `dicom2nifti` - this module
only ever sees already-loaded NumPy arrays, matching the Domain Layer's
"free of persistence and network concerns... receives inputs and returns
outputs" rule (`architecture/layers_description.md`). Reading DICOM files,
deduplicating slices by `ImagePositionPatient`, and running the actual
`dicom2nifti` conversion are Integration-Layer concerns - see
`integration.dicom_to_nifti.build_dicom_nifti_volume`, which reads the
files/runs the conversion and then calls `align_dicom_to_nifti` here to do
the actual reasoning over the resulting arrays.

Why an orientation transform needs discovering at all: `head_ct_unet`'s
model was trained on properly orientation-canonicalized data (its own
training pipeline consumes DICOM-derived NIfTI volumes, not raw DICOM
pixel arrays), but `DicomVolume.get_slice_hu()` returns each slice's raw,
as-acquired pixel array with no such canonicalization. Confirmed
empirically (2026-09-17, on a real Siemens NAEOTOM Alpha scan - the DICOM
series backing this project's `reference/segmentation/*_2SPC_bad_3_*`
reference pair): the in-plane difference between a DICOM slice and its
corresponding NIfTI slice is an *exact* transpose plus a single-axis flip -
a lossless array operation, not an interpolated resample (`dicom2nifti`
embeds any real gantry-tilt/obliquity into the NIfTI affine's metadata
rather than resampling pixel data). Rather than re-derive that transform
from the DICOM/NIfTI affines by hand (a notoriously easy-to-get-backwards
LPS/RAS, row/column-cosine calculation - an earlier attempt at exactly that
during this investigation produced a wrong result on the first try), this
module *empirically* discovers the transform once per volume: it tries all
8 transpose/flip combinations against one real reference slice pair and
keeps the one with (near) zero difference, raising `OrientationError`
loudly if none match well - a wrong or undetected orientation would
silently mirror/rotate the segmentation overlay on real patient anatomy,
so this must fail loudly rather than guess.

Not every DICOM slice necessarily has a NIfTI counterpart: confirmed on the
same real scan, one slice ("I0000000") sits at a wildly different patient
position from every other slice's otherwise-consistent, evenly spaced
stack (an ~84mm jump, with different in-plane x/y too) - almost certainly a
scout/localizer image sharing this series' `SeriesInstanceUID` and
`Rows`/`Columns` (a separate, pre-existing gap in `discover_series`'s own
geometry-based splitting, out of scope to fix here). So this module matches
each given DICOM slice to its NIfTI slice individually, by content, rather
than assuming a 1:1-by-order correspondence - a slice with no good match is
simply excluded (`is_segmentable` returns False), not a fatal error.

The reference slice used to discover the orientation transform must itself
have real content: a uniform/blank slice (e.g. pure air) can't discriminate
between candidate transpose/flip combinations at all, since flipping or
transposing a constant image returns that same constant image - every
candidate ties. Confirmed as a real bug on a real scan (2026-09-25, a
gantry-tilted stroke-protocol study whose first slice happened to be an
entirely blank -1024 HU margin slice): always using a fixed index (e.g.
"slice 0") as the reference silently picked an untested, wrong transform
purely because of tie-breaking order, correct-looking only because it was
never actually validated against real anatomy - every slice with real
content then failed to match anything, which looked like (but had nothing
to do with) the gantry tilt. `align_dicom_to_nifti` now auto-selects the
slice with the highest pixel variance as its reference by default, which is
guaranteed to have real content to discriminate against.
"""

from __future__ import annotations

import dataclasses

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# A real match measured 0.0 exactly (see module docstring); a residual
# above this means the geometry assumptions this module relies on (no true
# resampling, a single global in-plane transform) don't hold for this
# volume, and guessing would risk a silently wrong overlay.
_MAX_MATCH_MAD = 1.0


class OrientationError(RuntimeError):
    """Raised when the DICOM<->NIfTI in-plane orientation transform can't
    be determined reliably for a volume - see module docstring.
    """


class SliceNotSegmentableError(KeyError):
    """Raised by `DicomNiftiVolume.get_nifti_slice_hu`/`map_mask_to_dicom_space`
    for a DICOM slice index with no matching NIfTI slice (see module
    docstring - e.g. a scout/localizer sharing this series' geometry).
    """


@dataclass(frozen=True)
class AxisTransform:
    """A lossless in-plane transform (transpose + per-axis flip) between a
    DICOM slice's own pixel grid and its corresponding NIfTI slice."""

    transpose: bool
    flip0: bool
    flip1: bool

    def apply(self, arr: np.ndarray) -> np.ndarray:
        out = arr.T if self.transpose else arr
        if self.flip0:
            out = out[::-1, :]
        if self.flip1:
            out = out[:, ::-1]
        return out

    def invert(self, arr: np.ndarray) -> np.ndarray:
        # Flips are self-inverse; undo them (in either order, they commute)
        # before undoing the transpose, exactly reversing `apply`.
        out = arr
        if self.flip1:
            out = out[:, ::-1]
        if self.flip0:
            out = out[::-1, :]
        return out.T if self.transpose else out


@dataclass
class DicomNiftiVolume:
    """A NIfTI volume plus everything needed to run segmentation on it and
    map results back onto the DICOM slices it was aligned against.

    Built by `align_dicom_to_nifti` below (pure), or by `integration.
    dicom_to_nifti.build_dicom_nifti_volume` (which handles the real file
    I/O/`dicom2nifti` conversion first, then calls into this module).
    """

    nifti_hu: np.ndarray  # (rows, cols, n_slices), float32 HU
    _dicom_index_to_nifti_index: dict[int, int]
    _transform: AxisTransform

    def is_segmentable(self, dicom_index: int) -> bool:
        """False for a DICOM slice with no matching NIfTI slice (see module
        docstring) - callers must check this before segmenting an index."""
        return dicom_index in self._dicom_index_to_nifti_index

    def get_nifti_slice_hu(self, dicom_index: int) -> np.ndarray:
        """The NIfTI-space HU slice corresponding to DICOM slice `dicom_index`
        (duplicate DICOM slices map to the same NIfTI slice).

        Raises `SliceNotSegmentableError` if `is_segmentable(dicom_index)`
        is False.
        """
        if dicom_index not in self._dicom_index_to_nifti_index:
            raise SliceNotSegmentableError(
                f"DICOM slice {dicom_index} has no matching NIfTI slice - see module docstring."
            )
        nifti_index = self._dicom_index_to_nifti_index[dicom_index]
        return self.nifti_hu[:, :, nifti_index]

    def map_mask_to_dicom_space(self, nifti_mask: np.ndarray) -> np.ndarray:
        """Maps a boolean mask computed on a `get_nifti_slice_hu` slice back
        onto that DICOM slice's own pixel grid/orientation."""
        return self._transform.invert(nifti_mask)

    def remap_result_to_dicom_space(self, result):
        """Maps every mask in a segmentation result computed on a NIfTI-
        space slice (`get_nifti_slice_hu`) back onto that DICOM slice's own
        pixel grid, so every consumer downstream (the cache, overlays, the
        selection dialog, QA/metrics) keeps working in DICOM pixel space -
        segmentation is the only subsystem that ever detours through NIfTI
        space.

        Rebuilds via `type(result)(masks=...)` rather than depending on a
        specific `SegmentationResult` class - `result` need only be "some
        object with a `.masks` dict", matching `viewmodels.segmentation_
        overlay.decode_masks_for_checkboxes`'s own type-agnostic contract.
        """
        masks = {name: self.map_mask_to_dicom_space(mask) for name, mask in result.masks.items()}
        if dataclasses.is_dataclass(result):  # keeps the other fields (e.g. the CNN confidence)
            return dataclasses.replace(result, masks=masks)
        return type(result)(masks=masks)

    def rekeyed(self, dicom_index_to_alignment_key: dict[int, int]) -> "DicomNiftiVolume":
        """Returns a new `DicomNiftiVolume` with every entry re-keyed
        through `dicom_index_to_alignment_key` - e.g. mapping duplicate
        DICOM slice indices back onto the representative index this
        alignment was actually computed against - without a caller needing
        to reach into this object's own internals to do that expansion.
        """
        return DicomNiftiVolume(
            nifti_hu=self.nifti_hu,
            _dicom_index_to_nifti_index={
                dicom_index: self._dicom_index_to_nifti_index[key]
                for dicom_index, key in dicom_index_to_alignment_key.items()
                if key in self._dicom_index_to_nifti_index
            },
            _transform=self._transform,
        )


def _match_slice_to_nifti(
    transformed_slice: np.ndarray, nifti_hu: np.ndarray, claimed: set[int]
) -> tuple[int, float] | None:
    """Finds the not-yet-claimed NIfTI slice with the smallest mean-abs-HU
    difference to `transformed_slice`, or None if `nifti_hu` has no
    unclaimed slice of a matching shape at all.
    """
    best: tuple[int, float] | None = None
    for nifti_index in range(nifti_hu.shape[2]):
        if nifti_index in claimed:
            continue
        nifti_slice = nifti_hu[:, :, nifti_index]
        if nifti_slice.shape != transformed_slice.shape:
            continue
        mad = float(np.mean(np.abs(transformed_slice - nifti_slice)))
        if best is None or mad < best[1]:
            best = (nifti_index, mad)
    return best


def _select_reference_key(dicom_slices: dict[int, np.ndarray]) -> int:
    """Picks the slice with the most information content (highest pixel
    variance) to discover the orientation transform from - see the module
    docstring for why a fixed/arbitrary choice (e.g. "always slice 0") is
    unsafe: a uniform slice can't discriminate between transpose/flip
    candidates at all, so it risks silently locking in an untested, wrong
    transform.

    This is the single best guess, kept as its own function since
    `align_dicom_to_nifti`'s default (`reference_key=None`) tries it first -
    but a variance-only guess isn't infallible either (see `_reference_key_
    candidates`'s own docstring), so callers needing robustness should use
    that instead of stopping here.
    """
    return _reference_key_candidates(dicom_slices)[0]


def _reference_key_candidates(dicom_slices: dict[int, np.ndarray]) -> list[int]:
    """Every slice key, ordered by decreasing pixel variance - `align_dicom_
    to_nifti` tries each in turn until one actually produces a confident
    match, rather than trusting the single highest-variance slice
    unconditionally.

    Why more than one candidate is needed: high variance means "has enough
    content to discriminate a transpose/flip candidate," not "definitely has
    a real NIfTI counterpart." Confirmed as a real failure on real data
    (2026-09-25, a Siemens photon-counting 'good postop' scan): `dicom2nifti`
    silently dropped one DICOM slice that had no real NIfTI counterpart
    (same class of gap as the scout/localizer case the module docstring
    already documents) - and that exact orphaned slice happened to have the
    *highest* variance of the whole series (an artifact/outlier, not real
    anatomical detail), so picking only the single highest-variance slice
    reproduced the same class of bug the variance heuristic was originally
    added to fix, just from the opposite direction.
    """
    return sorted(dicom_slices, key=lambda key: float(np.var(dicom_slices[key])), reverse=True)


def align_dicom_to_nifti(
    dicom_slices: dict[int, np.ndarray], nifti_hu: np.ndarray, reference_key: int | None = None
) -> DicomNiftiVolume:
    """Aligns each of `dicom_slices` (arbitrary integer keys -> 2D HU
    arrays, e.g. one per deduplicated DICOM slice) against `nifti_hu` (an
    already-converted NIfTI volume), discovering the shared orientation
    transform from a reference slice and matching every other slice
    individually by content.

    Pure - no file I/O, no knowledge of DICOM concepts like duplicate
    slices or `ImagePositionPatient`. `reference_key` must be a real key of
    `dicom_slices` if given, and is tried exactly once (no fallback) - for
    when a caller has a specific reason to force a particular slice (e.g. a
    test). Left as `None` (the default), every slice is tried in turn as a
    candidate reference (highest pixel variance first, via `_reference_key_
    candidates`) until one produces a confident match - not just the single
    best-looking one - since the highest-variance slice can itself turn out
    to be an outlier with no real NIfTI counterpart (see `_reference_key_
    candidates`'s own docstring for the real case that proved this
    necessary).
    """
    candidate_keys = [reference_key] if reference_key is not None else _reference_key_candidates(dicom_slices)

    best_overall: tuple[int, int, float, AxisTransform] | None = None  # (reference_key, nifti_index, mad, transform)
    for candidate_reference_key in candidate_keys:
        dicom_reference_slice = dicom_slices[candidate_reference_key]
        best_for_reference: tuple[int, float, AxisTransform] | None = None
        for candidate in (
            AxisTransform(transpose=t, flip0=f0, flip1=f1)
            for t in (False, True)
            for f0 in (False, True)
            for f1 in (False, True)
        ):
            transformed = candidate.apply(dicom_reference_slice)
            match = _match_slice_to_nifti(transformed, nifti_hu, claimed=set())
            if match is not None and (best_for_reference is None or match[1] < best_for_reference[1]):
                best_for_reference = (match[0], match[1], candidate)

        if best_for_reference is not None and (best_overall is None or best_for_reference[1] < best_overall[2]):
            best_overall = (candidate_reference_key, *best_for_reference)
        if best_for_reference is not None and best_for_reference[1] <= _MAX_MATCH_MAD:
            break  # confident match found - no need to try further candidates

    if best_overall is None:
        raise OrientationError(
            "No transpose/flip combination matched any NIfTI slice's shape - "
            "cannot determine the DICOM<->NIfTI orientation mapping for this scan."
        )
    reference_key, reference_nifti_index, reference_mad, transform = best_overall
    if reference_mad > _MAX_MATCH_MAD:
        tried = f"{len(candidate_keys)} candidate reference slice(s)" if len(candidate_keys) > 1 else "the given reference slice"
        raise OrientationError(
            f"Best DICOM<->NIfTI orientation match still differs by {reference_mad:.2f} HU on "
            f"average, after trying {tried} - this scan's geometry doesn't match the lossless "
            "transpose/flip assumption this module relies on (see module docstring). Refusing to "
            "guess an overlay mapping."
        )
    logger.info(
        "nifti_alignment: orientation transform=%s (reference key=%s, match residual=%.4f HU)",
        transform,
        reference_key,
        reference_mad,
    )

    key_to_nifti_index: dict[int, int] = {reference_key: reference_nifti_index}
    claimed = {reference_nifti_index}
    skipped = 0
    for key, slice_hu in dicom_slices.items():
        if key == reference_key:
            continue
        transformed = transform.apply(slice_hu)
        match = _match_slice_to_nifti(transformed, nifti_hu, claimed)
        if match is None or match[1] > _MAX_MATCH_MAD:
            skipped += 1
            logger.warning(
                "nifti_alignment: slice key %s has no matching NIfTI slice "
                "(best residual %s) - excluding it from segmentation.",
                key,
                f"{match[1]:.2f} HU" if match else "n/a",
            )
            continue
        key_to_nifti_index[key] = match[0]
        claimed.add(match[0])

    if skipped:
        logger.info(
            "nifti_alignment: %d/%d slice(s) excluded from segmentation (no matching NIfTI slice)",
            skipped,
            len(dicom_slices),
        )

    return DicomNiftiVolume(nifti_hu=nifti_hu, _dicom_index_to_nifti_index=key_to_nifti_index, _transform=transform)
