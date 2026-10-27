"""`segmentation_plausibility` on synthetic label maps - no patient data."""

import numpy as np

from bictool.domain.segmentation.segmentation_plausibility import (
    FRAGMENTATION_LIMIT,
    assess_plausibility,
    slice_fragments,
)

PX = 0.5  # mm; a 20 mm² piece = 80 px


def _clean(n=120):
    """Left half frontal, right half parietal - two compact labels."""
    frontal = np.zeros((n, n), bool)
    frontal[10:110, 10:60] = True
    parietal = np.zeros((n, n), bool)
    parietal[10:110, 60:110] = True
    return {"Frontal Lobe": frontal, "Parietal Lobe": parietal}


def test_a_compact_segmentation_has_no_extra_pieces():
    assert slice_fragments(_clean(), PX) == (0, 0)


def test_scattered_pieces_count_and_tiny_specks_do_not():
    masks = _clean()
    for k in range(5):  # five 12x12 px pieces (36 mm²) of frontal inside the parietal half
        masks["Frontal Lobe"][20 + 15 * k:32 + 15 * k, 80:92] = True
        masks["Parietal Lobe"][20 + 15 * k:32 + 15 * k, 80:92] = False
    masks["Frontal Lobe"][100:103, 100:103] = True  # 9 px = 2 mm² speck: ignored
    masks["Parietal Lobe"][100:103, 100:103] = False

    extra, islands = slice_fragments(masks, PX)

    assert extra == 5          # five frontal pieces beyond its main piece (the parietal stays one piece around them)
    assert islands == 5        # each surrounded by the parietal lobe


def _with_pieces(count, size_px):
    masks = _clean()
    for k in range(count):
        r, c = 12 + (size_px + 2) * (k // 4), 62 + (size_px + 2) * (k % 4)
        masks["Frontal Lobe"][r:r + size_px, c:c + size_px] = True
        masks["Parietal Lobe"][r:r + size_px, c:c + size_px] = False
    return masks


def test_the_study_value_is_the_median_over_slices_with_enough_brain():
    clean = assess_plausibility({i: _clean() for i in range(5)}, PX)
    bad = assess_plausibility({i: _with_pieces(32, 10) for i in range(5)}, PX)      # 32 pieces of 25 mm²
    specks = assess_plausibility({i: _with_pieces(32, 5) for i in range(5)}, PX)    # 32 specks of 6 mm²

    assert clean.fragmentation == 0 and not clean.unreliable and clean.slices == 5
    assert bad.fragmentation > FRAGMENTATION_LIMIT and bad.unreliable and len(bad.worst_slices) == 3
    assert not specks.unreliable
    assert assess_plausibility({}, PX).slices == 0 and not assess_plausibility({}, PX).unreliable
