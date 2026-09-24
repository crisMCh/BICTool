"""Tests `segment_slice`'s pre/post-processing logic against a fake model,
not a real Keras U-Net - a real `model.predict()` call on this architecture
takes ~3 minutes on CPU (measured directly), which would make the test
suite unusable. `test_unet_architecture.py` covers the real architecture's
shape correctness; this file covers `head_ct_unet.py`'s own logic (pad,
argmax, per-class mask decode, crop-back) in isolation.
"""

import numpy as np
import pytest

from bictool.domain.segmentation import head_ct_unet


class _FakeModel:
    """Stands in for a loaded Keras model: `predict(batch, verbose=0)`
    returns a fixed, hand-crafted probability array of the right shape,
    with `argmax` landing on a chosen class per pixel-region.
    """

    def __init__(self, labels_512: np.ndarray, nb_classes: int) -> None:
        self._labels_512 = labels_512
        self._nb_classes = nb_classes

    def predict(self, batch, verbose=0):
        flat_labels = self._labels_512.reshape(-1)
        probabilities = np.zeros((flat_labels.size, self._nb_classes), dtype=np.float32)
        probabilities[np.arange(flat_labels.size), flat_labels] = 1.0
        return probabilities[np.newaxis, ...]


def test_segment_slice_decodes_one_mask_per_class_name():
    labels_512 = np.zeros((512, 512), dtype=np.int64)
    labels_512[100:200, 100:200] = 4  # Septum Pellucidum
    model = _FakeModel(labels_512, nb_classes=head_ct_unet.NB_CLASSES)

    result = head_ct_unet.segment_slice(model, np.zeros((512, 512), dtype=np.float64))

    assert set(result.masks) == set(head_ct_unet._CLASS_NAMES.values())
    assert result.masks["Septum Pellucidum"][100:200, 100:200].all()
    assert not result.masks["Septum Pellucidum"][0:50, 0:50].any()


def test_segment_slice_crops_masks_back_to_the_original_non_square_shape():
    labels_512 = np.zeros((512, 512), dtype=np.int64)
    model = _FakeModel(labels_512, nb_classes=head_ct_unet.NB_CLASSES)

    result = head_ct_unet.segment_slice(model, np.zeros((512, 434), dtype=np.float64))

    for mask in result.masks.values():
        assert mask.shape == (512, 434)


def test_background_class_zero_is_not_a_named_structure():
    # Class index 0 is background per the confirmed label map - it should
    # never appear as a mask key.
    assert 0 not in head_ct_unet._CLASS_NAMES
    assert len(head_ct_unet._CLASS_NAMES) == head_ct_unet.NB_CLASSES - 1


def test_load_weights_raises_a_clear_error_for_a_missing_file(tmp_path):
    model = _FakeModel(np.zeros((512, 512), dtype=np.int64), nb_classes=head_ct_unet.NB_CLASSES)
    missing_path = tmp_path / "does_not_exist.hdf5"

    with pytest.raises(FileNotFoundError, match="not found"):
        head_ct_unet.load_weights(model, missing_path)
