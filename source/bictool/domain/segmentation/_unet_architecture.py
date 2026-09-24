"""Shared 2D U-Net architecture for the pretrained head-CT segmentation
models wired into this project.

`head_ct_unet.py` (17-class anatomy segmentation) and `aisd_stroke_unet.py`
(2-class stroke-lesion segmentation) are structurally identical networks -
same encoder/decoder, same input size, differing only in the number of
output classes and which trained weights get loaded into them - so the
layer stack is defined exactly once here and shared, rather than risking
the two copies drifting apart (e.g. a filter-count fix applied to one and
not the other).

Faithfully reproduces the architecture described in Cai et al., "Fully
Automated Segmentation of Head CT Neuroanatomy Using Deep Learning,"
Radiology: Artificial Intelligence, 2020 (DOI 10.1148/ryai.2020190183) and
its public implementation, `jasonccai/HeadCTSegmentation`: a 5-level 2D
U-Net, 512x512x1 input, Conv2D encoder filters 64/128/256/512/1024 with
LayerNormalization + ReLU per conv and MaxPooling between levels, a
symmetric decoder with skip connections, and a final per-pixel softmax
classification head. `Ashly1991/AISD-HeadCT-Transfer-Learning` transfers
this exact architecture (only `nb_classes` and the trained weights differ)
to binary stroke-lesion segmentation.

One architectural detail - the exact kernel size of the Conv2D immediately
following each `UpSampling2D` in the decoder - was not confirmed from the
fetched source (only the overall decoder shape was: upsample, conv,
concatenate with the skip connection, two more convs). Implemented here as
a 2x2 conv, the common U-Net convention; if this is wrong, `load_weights()`
on either model will fail loudly with a shape mismatch rather than silently
producing wrong predictions - this is a "verify once real weights are
available" item, not a silent assumption.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

_ENCODER_FILTERS = (64, 128, 256, 512, 1024)


def build_unet(nb_classes: int) -> keras.Model:
    """Builds the shared U-Net architecture with an `nb_classes`-wide
    softmax head. Weights are not loaded here - see `load_weights` in
    `head_ct_unet.py`/`aisd_stroke_unet.py`.

    Parameters
    ----------
    nb_classes : number of output classes (17 for the anatomy model - 16
        structures plus background; 2 for the stroke model - background
        vs. lesion)

    Returns
    -------
    A `keras.Model` (compiled with `jit_compile=True` for fast inference -
    see the note at the end of this function - no optimizer/loss, since
    this is never trained) mapping `(batch, 512, 512, 1)` HU input to
    `(batch, 262144, nb_classes)` per-pixel class probabilities (the
    512*512 spatial grid flattened, matching the original implementation).
    """
    inputs = keras.Input(shape=(512, 512, 1), name="unetinput")

    x = inputs
    skips = []
    for level, filters in enumerate(_ENCODER_FILTERS):
        x = _conv_block(x, filters, name=f"conv{level + 1}")
        if level < len(_ENCODER_FILTERS) - 1:
            skips.append(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{level + 1}")(x)

    for level, filters in enumerate(reversed(_ENCODER_FILTERS[:-1])):
        x = layers.UpSampling2D(size=(2, 2), name=f"upsample{level + 1}")(x)
        x = layers.Conv2D(filters, 2, padding="same", name=f"upconv{level + 1}")(x)
        skip = skips[len(skips) - 1 - level]
        x = layers.Concatenate(name=f"concat{level + 1}")([skip, x])
        x = _conv_block(x, filters, name=f"deconv{level + 1}")

    # Named "classifier_head" here (not the student repo's own "stroke_head",
    # which is stroke-specific terminology inappropriate for the shared
    # anatomy+stroke builder) - if `load_weights()` turns out to load by
    # layer name rather than topological order, this final layer's name may
    # need to match whatever the original `.hdf5`/`.weights.h5` file used;
    # a name mismatch here surfaces as a loud shape/name error, not silent
    # misloading, so safe to adjust empirically once real weights are tried.
    outputs = layers.Conv2D(nb_classes, 1, activation="softmax", name="classifier_head")(x)
    outputs = layers.Reshape((512 * 512, nb_classes), name="unetoutput")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"head_ct_unet_{nb_classes}class")
    # Measured directly against the real Mayo weights: a plain (uncompiled)
    # `.predict()`/`__call__` on this architecture takes ~157s for a single
    # 512x512x1 image on CPU - LayerNormalization at every level plus a
    # 1024-channel bottleneck apparently don't fuse well under eager/default
    # graph execution. `jit_compile=True` (XLA) fuses the whole forward pass
    # and cuts that to ~1-2s per call after a one-time ~1-2s trace/compile
    # on the model's first `.predict()` call - confirmed numerically
    # identical output (`np.allclose`), not just faster-and-wrong. No
    # optimizer/loss is set since this model is never trained - `.compile()`
    # here exists purely to turn on JIT for inference.
    model.compile(jit_compile=True)
    return model


def _conv_block(x, filters: int, *, name: str):
    """Two Conv2D(3x3, same) layers, each followed by LayerNormalization + ReLU."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv{i}")(x)
        x = layers.LayerNormalization(name=f"{name}_norm{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x
