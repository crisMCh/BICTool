from types import SimpleNamespace

import numpy as np

from bictool.presentation.viewmodels.segmentation_overlay import decode_masks_for_checkboxes


def _mask(value: bool, shape=(4, 4)) -> np.ndarray:
    return np.full(shape, value, dtype=bool)


def test_direct_one_to_one_mapping():
    result = SimpleNamespace(masks={"Septum Pellucidum": _mask(True), "Cerebellum": _mask(False)})
    mapping = {"Septum pellucidum": ["Septum Pellucidum"]}

    decoded = decode_masks_for_checkboxes(result, mapping)

    assert decoded["Septum pellucidum"].all()


def test_combines_multiple_raw_classes_with_or():
    # WP2's "Basal ganglia" = Caudate Nucleus OR Lentiform Nucleus.
    caudate = np.zeros((4, 4), dtype=bool)
    caudate[0, 0] = True
    lentiform = np.zeros((4, 4), dtype=bool)
    lentiform[3, 3] = True
    result = SimpleNamespace(masks={"Caudate Nucleus": caudate, "Lentiform Nucleus": lentiform})
    mapping = {"Basal ganglia": ["Caudate Nucleus", "Lentiform Nucleus"]}

    decoded = decode_masks_for_checkboxes(result, mapping)

    assert decoded["Basal ganglia"][0, 0]
    assert decoded["Basal ganglia"][3, 3]
    assert not decoded["Basal ganglia"][1, 1]


def test_works_with_a_single_key_stroke_style_result():
    result = SimpleNamespace(masks={"Lesion": _mask(True)})
    mapping = {"Suspected lesion": ["Lesion"]}

    decoded = decode_masks_for_checkboxes(result, mapping)

    assert decoded["Suspected lesion"].all()


def test_empty_mapping_returns_empty_dict():
    result = SimpleNamespace(masks={"Cerebellum": _mask(True)})

    decoded = decode_masks_for_checkboxes(result, {})

    assert decoded == {}
