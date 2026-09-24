from bictool.domain.segmentation._unet_architecture import build_unet


def test_build_unet_input_shape():
    model = build_unet(nb_classes=17)
    assert model.input_shape == (None, 512, 512, 1)


def test_build_unet_output_shape_matches_nb_classes():
    model = build_unet(nb_classes=17)
    assert model.output_shape == (None, 512 * 512, 17)


def test_build_unet_supports_a_different_class_count():
    # aisd_stroke_unet.py shares this same builder with nb_classes=2.
    model = build_unet(nb_classes=2)
    assert model.output_shape == (None, 512 * 512, 2)


def test_build_unet_returns_a_fresh_model_each_call():
    # Two independent models (anatomy + stroke) must not share weights/state.
    model_a = build_unet(nb_classes=17)
    model_b = build_unet(nb_classes=17)
    assert model_a is not model_b
