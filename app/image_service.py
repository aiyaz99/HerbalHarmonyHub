"""Image validation, MobileNetV2 preprocessing and real inference."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from .config import settings
from .data_service import get_plant_record
from .errors import InvalidImageError
from .logging_config import get_logger
from .model_loader import get_model_bundle

logger = get_logger(__name__)

PREPROCESSING_SPEC = {
    "targetSize": "224 x 224 px (MobileNetV2 input)",
    "steps": [
        "Decode JPG/PNG and convert to RGB",
        "Resize to 224 x 224",
        "Apply tf.keras.applications.mobilenet_v2.preprocess_input (scales to [-1, 1])",
        "Batch as a single tensor of shape (1, 224, 224, 3)",
    ],
}


def validate_upload(filename: str | None, content_type: str | None, payload: bytes) -> None:
    if not payload:
        raise InvalidImageError("The uploaded file is empty.")
    if len(payload) > settings.max_upload_bytes:
        raise InvalidImageError(
            f"Image is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.",
            {"bytes": len(payload), "limit": settings.max_upload_bytes},
        )
    if content_type not in settings.allowed_mime_types:
        raise InvalidImageError(
            "Unsupported file type. Upload a JPG, JPEG or PNG image.",
            {"received": content_type, "allowed": list(settings.allowed_mime_types)},
        )
    # Verify the bytes really are a decodable image of the declared kind.
    try:
        from PIL import Image

        probe = Image.open(io.BytesIO(payload))
        probe.verify()
        if probe.format not in {"JPEG", "PNG"}:
            raise InvalidImageError(f"Decoded image format {probe.format} is not supported.")
    except InvalidImageError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad upload
        raise InvalidImageError("The uploaded file could not be decoded as an image.", {"reason": str(exc)}) from exc


def decode_and_preprocess(payload: bytes):
    """Returns (batch_tensor, pil_rgb_resized_image)."""
    import numpy as np
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    image = Image.open(io.BytesIO(payload)).convert("RGB")
    image = image.resize((settings.image_size, settings.image_size))
    array = np.asarray(image, dtype="float32")
    batch = preprocess_input(np.expand_dims(array, axis=0))
    return batch, image


def predict(payload: bytes, filename: str, top_k: int | None = None, with_gradcam: bool = True) -> dict[str, Any]:
    """Runs the real trained classifier. Raises ModelNotReadyError when untrained."""
    import numpy as np

    bundle = get_model_bundle()
    batch, image = decode_and_preprocess(payload)
    probabilities = np.asarray(bundle.model.predict(batch, verbose=0))[0]

    if probabilities.shape[0] != len(bundle.class_names):
        from .errors import ModelNotReadyError

        raise ModelNotReadyError(
            "Model output size does not match the class mapping. Re-export both artifacts from training.",
            {"outputs": int(probabilities.shape[0]), "classes": len(bundle.class_names)},
        )

    k = min(top_k or settings.default_top_k, len(bundle.class_names))
    order = np.argsort(probabilities)[::-1][:k]

    candidates: list[dict[str, Any]] = []
    for index in order:
        label = bundle.class_names[int(index)]
        record = get_plant_record(label)
        candidates.append(
            {
                "label": label,
                "confidence": float(probabilities[int(index)]),
                "plant": record,
                "knowledgeRecordAvailable": record is not None,
            }
        )

    explainability: dict[str, Any] = {"gradCamAvailable": False, "message": "", "heatmap": None}
    if with_gradcam:
        from .gradcam import gradcam_overlay_base64

        try:
            heatmap = gradcam_overlay_base64(bundle, batch, image, int(order[0]))
            explainability = {
                "gradCamAvailable": True,
                "message": f"Grad-CAM computed on layer '{bundle.last_conv_layer}' for the predicted class.",
                "heatmap": heatmap,
            }
        except Exception as exc:  # noqa: BLE001 - never fail the prediction on XAI
            logger.warning("Grad-CAM failed: %s", exc)
            explainability = {
                "gradCamAvailable": False,
                "message": f"Grad-CAM could not be computed for this request: {exc}",
                "heatmap": None,
            }

    top = candidates[0]
    warnings: list[str] = []
    if not top["knowledgeRecordAvailable"]:
        warnings.append(
            f"The classifier predicted '{top['label']}' but the supplied knowledge CSV has no matching row, "
            "so no botanical or remedy information can be shown for it."
        )

    return {
        "source": "model",
        "fileName": filename,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "top": top,
        "candidates": candidates,
        "preprocessing": PREPROCESSING_SPEC,
        "explainability": explainability,
        "model": {
            "classCount": len(bundle.class_names),
            "trainedAt": bundle.metadata.get("trained_at"),
            "architecture": bundle.metadata.get("architecture", "MobileNetV2 (transfer learning)"),
        },
        "warnings": warnings,
    }
