"""HerbaScan FastAPI ML service.

Endpoints:
    GET  /api/health
    POST /api/predict-image      (multipart image)
    POST /api/recommend          (JSON symptoms)
    POST /api/combined-analysis  (multipart image + symptoms)

Honesty guarantees:
    * No endpoint ever returns a fabricated prediction, score or heatmap.
    * If the trained model or the TF-IDF artifacts are missing, the affected
      endpoint returns HTTP 503 with code `model_not_ready` / `recommender_not_ready`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import data_service, image_service, model_loader, recommendation_service
from app.config import settings
from app.errors import ApiError, InvalidInputError
from app.logging_config import get_logger

logger = get_logger("herbascan.api")

app = FastAPI(title=settings.app_name, version=settings.version, docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    logger.warning("%s: %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "An unexpected server error occurred.", "detail": {}}},
    )


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
class RecommendRequest(BaseModel):
    symptoms: str = Field(..., min_length=1, description="Free-text symptom description")
    top_n: int = Field(default=settings.default_top_n, ge=1, le=10)


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "time": datetime.now(timezone.utc).isoformat(),
        "imageModel": model_loader.model_status(),
        "recommender": recommendation_service.recommender_status(),
        "datasets": data_service.dataset_status(),
    }


async def _read_upload(image: UploadFile) -> bytes:
    payload = await image.read()
    image_service.validate_upload(image.filename, image.content_type, payload)
    return payload


@app.post("/api/predict-image")
async def predict_image(
    image: UploadFile = File(...),
    top_k: int = Form(default=settings.default_top_k),
    gradcam: bool = Form(default=True),
) -> dict[str, Any]:
    payload = await _read_upload(image)
    return image_service.predict(
        payload,
        filename=image.filename or "upload",
        top_k=top_k,
        with_gradcam=gradcam,
    )


@app.get("/api/plants")
async def plants() -> dict[str, Any]:
    """Knowledge base records, built only from the supplied plant CSV.

    Raises ``DatasetNotReadyError`` (HTTP 503, code ``dataset_not_ready``) when the
    real CSV has not been placed in ``data/`` yet — no substitute records exist.
    """
    records = data_service.load_plant_records()
    values = sorted(records.values(), key=lambda item: str(item.get("name", "")).lower())
    families = sorted({r["family"] for r in values if r.get("family")})
    symptoms = sorted({s for r in values for s in r.get("symptoms", []) if s})
    return {
        "source": "dataset",
        "sourceFile": data_service.MEDICAL_PLANTS_FILE,
        "count": len(values),
        "families": families,
        "symptoms": symptoms,
        "records": values,
    }


@app.get("/api/plants/{plant_id}")
async def plant_detail(plant_id: str) -> dict[str, Any]:
    record = data_service.get_plant_record(plant_id)
    if record is None:
        raise InvalidInputError(
            f"No knowledge record exists for '{plant_id}' in the supplied dataset.",
            {"plantId": plant_id},
        )
    return record


@app.post("/api/recommend")
async def recommend(body: RecommendRequest) -> dict[str, Any]:
    return recommendation_service.recommend(body.symptoms, top_n=body.top_n)




@app.post("/api/combined-analysis")
async def combined_analysis(
    image: UploadFile = File(...),
    symptoms: str = Form(...),
    top_n: int = Form(default=settings.default_top_n),
    top_k: int = Form(default=settings.default_top_k),
) -> dict[str, Any]:
    if len((symptoms or "").strip()) < 3:
        raise InvalidInputError("Please describe the symptoms in a little more detail.")
    payload = await _read_upload(image)

    # Both stages must be genuinely available; a partial result is reported as such.
    prediction = image_service.predict(payload, filename=image.filename or "upload", top_k=top_k)
    recommendation = recommendation_service.recommend(symptoms, top_n=top_n)
    relation = recommendation_service.compatibility(prediction["top"]["plant"], recommendation["tokens"])

    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "recommendation": recommendation,
        "compatibility": relation,
        "disclaimer": (
            "Informational and traditional herbal knowledge only. This combined analysis is not a "
            "medical diagnosis, prescription or dosage instruction."
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
