"""Real TF-IDF + cosine-similarity recommender over the supplied CSV records.

The vectorizer and document matrix are fitted OFFLINE by
``training/build_recommender.py`` and persisted with joblib. At request time this
module only transforms the query and ranks existing records — it never refits and
never fabricates a match.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import settings
from .data_service import get_plant_record
from .errors import InvalidInputError, RecommenderNotReadyError
from .logging_config import get_logger

logger = get_logger(__name__)

VECTORIZER_FILE = "tfidf_vectorizer.joblib"
MATRIX_FILE = "tfidf_matrix.joblib"
RECORDS_FILE = "records.json"
METADATA_FILE = "recommender_metadata.json"

MIN_SIMILARITY = 0.05

_LOCK = threading.Lock()
_BUNDLE: "RecommenderBundle | None" = None


@dataclass
class RecommenderBundle:
    vectorizer: Any
    matrix: Any
    records: list[dict[str, Any]]
    metadata: dict[str, Any]


def artifacts_present() -> bool:
    directory = settings.recommender_dir
    return all((directory / name).exists() for name in (VECTORIZER_FILE, MATRIX_FILE, RECORDS_FILE))


def recommender_status() -> dict[str, Any]:
    metadata = None
    metadata_path = settings.recommender_dir / METADATA_FILE
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"error": "recommender_metadata.json is not valid JSON"}
    return {
        "ready": artifacts_present(),
        "artifactsDir": str(settings.recommender_dir),
        "metadata": metadata,
        "loaded": _BUNDLE is not None,
    }


def get_bundle() -> RecommenderBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        if not artifacts_present():
            raise RecommenderNotReadyError(
                "TF-IDF recommender artifacts are not built yet. "
                "Run training/build_recommender.py after placing the real CSV files in data/.",
                {"expectedDir": str(settings.recommender_dir)},
            )
        import joblib

        directory = settings.recommender_dir
        vectorizer = joblib.load(directory / VECTORIZER_FILE)
        matrix = joblib.load(directory / MATRIX_FILE)
        records = json.loads((directory / RECORDS_FILE).read_text(encoding="utf-8"))
        metadata: dict[str, Any] = {}
        metadata_path = directory / METADATA_FILE
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        logger.info("Loaded recommender with %d records", len(records))
        _BUNDLE = RecommenderBundle(vectorizer=vectorizer, matrix=matrix, records=records, metadata=metadata)
        return _BUNDLE


def reset_recommender_cache() -> None:
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None


def _query_terms(vectorizer: Any, query: str) -> list[str]:
    """Terms from the query that actually exist in the fitted vocabulary."""
    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    seen: list[str] = []
    for term in analyzer(query):
        if term in vocabulary and term not in seen:
            seen.append(term)
    return seen


def recommend(query: str, top_n: int | None = None) -> dict[str, Any]:
    text = (query or "").strip()
    if len(text) < 3:
        raise InvalidInputError("Please describe the symptoms in a little more detail (at least 3 characters).")

    from sklearn.metrics.pairwise import cosine_similarity

    bundle = get_bundle()
    n = top_n or settings.default_top_n
    vector = bundle.vectorizer.transform([text])
    similarities = cosine_similarity(vector, bundle.matrix)[0]

    tokens = _query_terms(bundle.vectorizer, text)

    ranked = sorted(range(len(similarities)), key=lambda i: float(similarities[i]), reverse=True)
    matches: list[dict[str, Any]] = []
    for index in ranked:
        score = float(similarities[index])
        if score < MIN_SIMILARITY:
            break
        record = bundle.records[index]
        document_terms = set(record.get("terms", []))
        matched_keywords = [token for token in tokens if token in document_terms]
        matched_symptoms = [
            symptom
            for symptom in record.get("symptoms", [])
            if any(token in symptom for token in tokens)
        ]
        plant_record = get_plant_record(record.get("plantKey", record.get("plantName", "")))
        matches.append(
            {
                "plantKey": record.get("plantKey"),
                "plantName": record.get("plantName"),
                "plant": plant_record,
                "score": score,
                "matchedKeywords": matched_keywords,
                "matchedSymptoms": matched_symptoms,
                "traditionalUses": record.get("traditionalUses", []),
                "remedies": record.get("remedies", []),
                "precautions": record.get("precautions", []),
                "sourceFiles": record.get("sourceFiles", []),
                "explanation": (
                    f"Cosine similarity {score:.2f} between your description and the recorded "
                    f"symptom/remedy text for {record.get('plantName')}"
                    + (f"; shared terms: {', '.join(matched_keywords)}." if matched_keywords else ".")
                ),
            }
        )
        if len(matches) >= n:
            break

    return {
        "source": "model",
        "query": text,
        "tokens": tokens,
        "matches": matches,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "method": "sklearn TfidfVectorizer + cosine_similarity (fitted offline)",
        "recordCount": len(bundle.records),
        "warnings": (
            []
            if matches
            else [
                "No record in the supplied datasets reached the minimum similarity threshold for this description."
            ]
        ),
        "disclaimer": (
            "Educational and traditional herbal information only. This is not a diagnosis, "
            "prescription or dosage instruction."
        ),
    }


def compatibility(plant_record: dict[str, Any] | None, tokens: list[str]) -> dict[str, Any]:
    """Overlap between an identified plant's recorded information and the query terms."""
    if not plant_record:
        return {
            "matchedSymptoms": [],
            "relevance": 0.0,
            "message": "No knowledge record exists for the identified class, so relevance cannot be computed.",
        }

    # Check all relevant knowledge fields, not only the symptoms field.
    fields = [
        "symptoms",
        "traditionalUses",
        "properties",
        "remedies",
        "precautions",
    ]

    recorded = []

    for field in fields:
        for value in plant_record.get(field, []):
            recorded.append(str(value).lower())

    # Also include the summary if available.
    summary = str(plant_record.get("summary", "")).lower()
    if summary:
        recorded.append(summary)

    matched = [
        item for item in recorded
        if any(token.lower() in item for token in tokens)
    ]

    # Remove duplicates while preserving order.
    matched = list(dict.fromkeys(matched))

    relevance = (
        0.0
        if not tokens or not recorded
        else min(1.0, len(matched) / max(1, min(len(tokens), 4)))
    )

    message = (
        f"Recorded information for {plant_record.get('name')} mentions "
        f"{', '.join(matched)}, which overlaps with the described symptoms."
        if matched
        else f"Recorded information for {plant_record.get('name')} does not mention the described symptoms."
    )

    return {
        "matchedSymptoms": matched,
        "relevance": relevance,
        "message": message,
    }

