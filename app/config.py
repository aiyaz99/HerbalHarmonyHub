"""Environment-driven configuration for the HerbaScan ML backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _path(name: str, default: str) -> Path:
    raw = Path(_env(name, default))
    return raw if raw.is_absolute() else (BASE_DIR / raw)


def _origins() -> list[str]:
    raw = _env("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:5173")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """All runtime paths and limits. Nothing here fabricates data."""

    app_name: str = "HerbaScan ML API"
    version: str = "1.0.0"
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    # Directories the user places real data / artifacts into.
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data"))
    artifacts_dir: Path = field(default_factory=lambda: _path("ARTIFACTS_DIR", "artifacts"))

    # Image classifier artifacts (produced by training/train_mobilenetv2.py).
    model_path: Path = field(
        default_factory=lambda: _path("MODEL_PATH", "artifacts/plant_classifier.keras")
    )
    class_map_path: Path = field(
        default_factory=lambda: _path("CLASS_MAP_PATH", "artifacts/class_indices.json")
    )
    model_metadata_path: Path = field(
        default_factory=lambda: _path("MODEL_METADATA_PATH", "artifacts/model_metadata.json")
    )

    # Recommender artifacts (produced by training/build_recommender.py).
    recommender_dir: Path = field(
        default_factory=lambda: _path("RECOMMENDER_DIR", "artifacts/recommender")
    )

    # Upload validation.
    max_upload_bytes: int = field(
        default_factory=lambda: int(_env("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
    )
    allowed_mime_types: tuple[str, ...] = ("image/jpeg", "image/jpg", "image/png")

    allowed_origins: list[str] = field(default_factory=_origins)

    image_size: int = 224
    default_top_k: int = 4
    default_top_n: int = 3


settings = Settings()
