"""Lazy loader for the trained MobileNetV2 artifact.

There is deliberately NO fallback model. Until `training/train_mobilenetv2.py` has
been run on the real dataset and produced the artifacts, every inference call
raises ``ModelNotReadyError`` and the API returns HTTP 503.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from .config import settings
from .errors import ModelNotReadyError
from .logging_config import get_logger

logger = get_logger(__name__)

_LOCK = threading.Lock()
_BUNDLE: "ModelBundle | None" = None


@dataclass
class ModelBundle:
    model: Any
    class_names: list[str]
    metadata: dict[str, Any]
    last_conv_layer: str


def artifacts_present() -> bool:
    return settings.model_path.exists() and settings.class_map_path.exists()


def model_status() -> dict[str, Any]:
    """Non-throwing status used by /api/health."""
    metadata: dict[str, Any] = {}
    if settings.model_metadata_path.exists():
        try:
            metadata = json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"error": "model_metadata.json is not valid JSON"}
    return {
        "ready": artifacts_present(),
        "modelPath": str(settings.model_path),
        "classMapPath": str(settings.class_map_path),
        "classCount": len(_read_class_names()) if settings.class_map_path.exists() else None,
        # Metrics are ONLY whatever real training wrote out. Never hardcoded here.
        "metadata": metadata or None,
        "loaded": _BUNDLE is not None,
    }


def _read_class_names() -> list[str]:
    """Reads class_indices.json written by training (mapping name -> index)."""
    try:
        raw = json.loads(settings.class_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelNotReadyError(
            "Class mapping file is missing or unreadable.",
            {"classMapPath": str(settings.class_map_path), "reason": str(exc)},
        ) from exc
    mapping = raw.get("class_indices", raw)
    if not isinstance(mapping, dict) or not mapping:
        raise ModelNotReadyError("Class mapping file does not contain a class_indices object.")
    ordered = sorted(mapping.items(), key=lambda item: int(item[1]))
    return [str(name) for name, _ in ordered]


def find_last_conv_layer(model: Any) -> str:
    """Find the final convolutional layer, including inside nested models."""
    import tensorflow as tf

    def walk(layers: list[Any], prefix: str = "") -> str | None:
        # Search from the end because Grad-CAM should use the final
        # convolutional feature map.
        for layer in reversed(layers):
            layer_name = f"{prefix}{layer.name}"

            # Prefer the final MobileNetV2 activation layer.
            if layer.name == "out_relu":
                try:
                    shape = tuple(layer.output.shape)
                    if len(shape) == 4:
                        return layer_name
                except Exception:
                    pass

            # Search inside nested Functional/Model layers.
            if isinstance(layer, tf.keras.Model):
                found = walk(
                    list(layer.layers),
                    prefix=f"{prefix}{layer.name}/",
                )
                if found:
                    return found

            # Check convolutional layers directly.
            if isinstance(
                layer,
                (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D),
            ):
                try:
                    shape = tuple(layer.output.shape)
                    if len(shape) == 4:
                        return layer_name
                except Exception:
                    pass

        return None

    found = walk(list(model.layers))

    if not found:
        raise ModelNotReadyError(
            "Could not locate a convolutional layer for Grad-CAM in the loaded model."
        )

    return found

def get_model_bundle() -> ModelBundle:
    """Loads (once) and returns the real trained model. Raises if not trained yet."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        if not artifacts_present():
            raise ModelNotReadyError(
                "No trained MobileNetV2 model artifact is available yet. "
                "Run training/train_mobilenetv2.py on the real dataset first.",
                {
                    "expectedModel": str(settings.model_path),
                    "expectedClassMap": str(settings.class_map_path),
                },
            )
        import tensorflow as tf

        class_names = _read_class_names()
        logger.info("Loading model from %s (%d classes)", settings.model_path, len(class_names))
        model = tf.keras.models.load_model(settings.model_path)
        metadata: dict[str, Any] = {}
        if settings.model_metadata_path.exists():
            try:
                metadata = json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        _BUNDLE = ModelBundle(
            model=model,
            class_names=class_names,
            metadata=metadata,
            last_conv_layer=find_last_conv_layer(model),
        )
        return _BUNDLE


def reset_model_cache() -> None:
    """Used by tests and after re-deploying a new artifact."""
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None
