"""Real Grad-CAM for the trained MobileNetV2 classifier.

No placeholder heatmaps: this module only runs when a real model artifact exists,
and it computes gradients of the predicted class score with respect to the final
convolutional feature maps.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


def _resolve_layer(model: Any, dotted_name: str):
    """Resolves a possibly nested layer path like 'mobilenetv2_1.00_224/out_relu'."""
    parts = dotted_name.split("/")
    current = model
    for part in parts[:-1]:
        current = current.get_layer(part)
    return current, current.get_layer(parts[-1])


def compute_heatmap(bundle: Any, batch, class_index: int):
    """Returns a 2-D numpy heatmap normalised to [0, 1]."""
    import numpy as np
    import tensorflow as tf

    owner, conv_layer = _resolve_layer(bundle.model, bundle.last_conv_layer)

    if owner is bundle.model:
        grad_model = tf.keras.models.Model(bundle.model.inputs, [conv_layer.output, bundle.model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(batch, training=False)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
    else:
        # Backbone is a nested model: run the backbone first, then the head.
        backbone_model = tf.keras.models.Model(owner.inputs, conv_layer.output)
        with tf.GradientTape() as tape:
            conv_outputs = backbone_model(batch, training=False)
            tape.watch(conv_outputs)
            features = conv_outputs
            started = False
            for layer in owner.layers:
                if layer.name == conv_layer.name:
                    started = True
                    continue
                if started:
                    features = layer(features, training=False)
            started = False
            for layer in bundle.model.layers:
                if layer is owner:
                    started = True
                    continue
                if started:
                    features = layer(features, training=False)
            loss = features[:, class_index]
        grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradients could not be computed for the selected convolutional layer.")

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if float(maximum) == 0.0:
        raise RuntimeError("Grad-CAM produced an empty activation map.")
    heatmap = heatmap / maximum
    return np.asarray(heatmap)


def gradcam_overlay_base64(bundle: Any, batch, image, class_index: int, alpha: float = 0.4) -> str:
    """Overlays the heatmap on the preprocessed input and returns a PNG data URL."""
    import numpy as np
    from matplotlib import cm
    from PIL import Image

    heatmap = compute_heatmap(bundle, batch, class_index)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")(np.arange(256))[:, :3]
    coloured = colormap[heatmap_uint8]
    coloured_image = Image.fromarray(np.uint8(coloured * 255)).resize(image.size, Image.BILINEAR)

    blended = Image.blend(image.convert("RGB"), coloured_image.convert("RGB"), alpha=alpha)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
