"""Tests `domain.artefacts.artefact_flags` with synthetic slices and label maps."""

import numpy as np
import pytest

from bictool.domain.artefacts.artefact_flags import (
    BAND_DEPTH_HU,
    METAL_HU,
    ArtefactFlags,
    band_depth,
    evaluate_artefact_flags,
    find_brainstem,
    metal_pixels,
    posterior_fossa_slices,
)

SPACING = 0.5  # mm


def _disc(shape, centre, radius):
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2


def _fossa_masks(shape=(200, 200)):
    """Brainstem disc in front of a cerebellum band, as on a posterior-fossa slice."""
    stem = _disc(shape, (70, 100), 30)
    cerebellum = np.zeros(shape, bool)
    cerebellum[102:170, 30:170] = True
    return {"Brain Parenchyma": stem, "Cerebellum": cerebellum}


def _slice(masks, noise=0.0, seed=0):
    img = np.full(masks["Cerebellum"].shape, -1000.0)
    img[masks["Cerebellum"] | masks["Brain Parenchyma"]] = 35.0
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return img


def test_metal_needs_a_few_connected_pixels_above_the_threshold():
    img = np.zeros((50, 50))
    img[10, 10] = METAL_HU + 500  # a single hot pixel: noise, not metal
    assert not metal_pixels(img).any()
    img[20:22, 20:22] = METAL_HU + 300
    assert metal_pixels(img).sum() == 4
    img[30:40, 30:40] = METAL_HU - 600  # dense bone stays below
    assert metal_pixels(img).sum() == 4


def test_the_brainstem_is_the_parenchyma_piece_touching_the_cerebellum():
    masks = _fossa_masks()
    masks["Brain Parenchyma"] = masks["Brain Parenchyma"] | _disc((200, 200), (15, 20), 8)  # far away, unrelated
    stem = find_brainstem(masks, SPACING)
    assert stem is not None and stem[70, 100] and not stem[15, 20]


def test_a_thin_strip_is_not_a_brainstem():
    masks = _fossa_masks()
    strip = np.zeros((200, 200), bool)
    strip[60:101, 99:101] = True  # 1 mm wide, touches the cerebellum
    masks["Brain Parenchyma"] = strip
    assert find_brainstem(masks, SPACING) is None


def test_a_dark_left_right_band_across_the_brainstem_is_measured():
    masks = _fossa_masks()
    stem = find_brainstem(masks, SPACING)
    clean = _slice(masks, noise=2.0)
    banded = clean.copy()
    banded[66:74, :] -= 8.0  # 4 mm dark band, left to right
    assert band_depth(clean, stem, SPACING) < BAND_DEPTH_HU
    assert band_depth(banded, stem, SPACING) > BAND_DEPTH_HU


def test_a_front_to_back_band_is_not_counted():
    masks = _fossa_masks()
    stem = find_brainstem(masks, SPACING)
    img = _slice(masks, noise=2.0)
    img[:, 96:104] -= 8.0  # runs anterior-posterior
    assert band_depth(img, stem, SPACING) < BAND_DEPTH_HU


def test_posterior_fossa_slices_are_in_the_skull_base_half_with_enough_cerebellum():
    fossa = _fossa_masks()
    high = {"Brain Parenchyma": _disc((200, 200), (100, 100), 90), "Cerebellum": np.zeros((200, 200), bool)}
    stray = {"Brain Parenchyma": _disc((200, 200), (100, 100), 10), "Cerebellum": _disc((200, 200), (100, 100), 6)}
    masks_by_slice = {0: fossa, 1: fossa, 2: high, 3: high, 4: high, 5: stray}  # 5: vertex, a few cerebellum pixels
    assert posterior_fossa_slices(masks_by_slice) == [0, 1]


def test_volume_flags_combine_metal_and_bands():
    masks = _fossa_masks()
    slices = {0: _slice(masks, 2.0, 0), 1: _slice(masks, 2.0, 1), 2: _slice(masks, 2.0, 2)}
    slices[1][66:74, :] -= 8.0
    slices[2][5:8, 5:8] = 3071.0
    high = {"Brain Parenchyma": _disc((200, 200), (100, 100), 90), "Cerebellum": np.zeros((200, 200), bool)}
    masks_by_slice = {0: masks, 1: masks, 2: high, 3: high}
    flags = evaluate_artefact_flags(masks_by_slice, lambda i: slices.get(i, np.zeros((200, 200))), 4, SPACING)
    assert flags.metal_slices == (2,) and flags.metal_max_hu == pytest.approx(3071.0)
    assert set(flags.band_depths) == {0, 1} and flags.band_slices == (1,)


def test_no_segmentation_still_finds_metal():
    img = np.zeros((40, 40))
    img[1:4, 1:4] = 4000.0
    flags = evaluate_artefact_flags({}, lambda i: img, 2, SPACING)
    assert flags.metal_slices == (0, 1) and flags.band_depths == {} and flags.band_slices == ()
    assert ArtefactFlags().band_slices == ()
