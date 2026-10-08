"""Tests `segment_slice_for_stroke`'s pre/post-processing logic against a
fake model - see `test_head_ct_unet.py`'s module docstring for why a real
model is never used in this test suite (a real `predict()` call takes
minutes on CPU).
"""

import numpy as np

from bictool.domain.segmentation import aisd_stroke_unet


class _FakeModel:
    def __init__(self, labels_512: np.ndarray, nb_classes: int) -> None:
        self._labels_512 = labels_512
        self._nb_classes = nb_classes

    def predict(self, batch, verbose=0):
        flat_labels = self._labels_512.reshape(-1)
        probabilities = np.zeros((flat_labels.size, self._nb_classes), dtype=np.float32)
        probabilities[np.arange(flat_labels.size), flat_labels] = 1.0
        return probabilities[np.newaxis, ...]


def test_segment_slice_for_stroke_returns_a_single_lesion_mask():
    labels_512 = np.zeros((512, 512), dtype=np.int64)
    labels_512[50:80, 50:80] = 1  # lesion class
    model = _FakeModel(labels_512, nb_classes=aisd_stroke_unet.NB_CLASSES)

    result = aisd_stroke_unet.segment_slice_for_stroke(model, np.zeros((512, 512), dtype=np.float64))

    assert set(result.masks) == {"Lesion"}
    assert result.masks["Lesion"][50:80, 50:80].all()
    assert not result.masks["Lesion"][0:10, 0:10].any()


def test_no_lesion_predicted_gives_an_all_false_mask():
    labels_512 = np.zeros((512, 512), dtype=np.int64)  # everything background
    model = _FakeModel(labels_512, nb_classes=aisd_stroke_unet.NB_CLASSES)

    result = aisd_stroke_unet.segment_slice_for_stroke(model, np.zeros((512, 512), dtype=np.float64))

    assert not result.masks["Lesion"].any()


def test_segment_slice_for_stroke_crops_back_to_the_original_shape():
    labels_512 = np.zeros((512, 512), dtype=np.int64)
    model = _FakeModel(labels_512, nb_classes=aisd_stroke_unet.NB_CLASSES)

    result = aisd_stroke_unet.segment_slice_for_stroke(model, np.zeros((512, 434), dtype=np.float64))

    assert result.masks["Lesion"].shape == (512, 434)

