"""Structured, honest error types. The API never substitutes fake results."""

from __future__ import annotations


class ApiError(Exception):
    code = "api_error"
    status_code = 400

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class InvalidImageError(ApiError):
    code = "invalid_image"
    status_code = 400


class InvalidInputError(ApiError):
    code = "invalid_input"
    status_code = 422


class ModelNotReadyError(ApiError):
    """Raised when no trained MobileNetV2 artifact exists yet."""

    code = "model_not_ready"
    status_code = 503


class RecommenderNotReadyError(ApiError):
    """Raised when the TF-IDF artifacts have not been built yet."""

    code = "recommender_not_ready"
    status_code = 503


class DatasetNotReadyError(ApiError):
    code = "dataset_not_ready"
    status_code = 503
