"""TF-IDF + cosine-similarity recommender over Excel-derived records."""
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
    d = settings.recommender_dir
    return all((d / n).exists() for n in (VECTORIZER_FILE, MATRIX_FILE, RECORDS_FILE))

def recommender_status() -> dict[str, Any]:
    metadata = None
    p = settings.recommender_dir / METADATA_FILE
    if p.exists():
        try:
            metadata = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"error": "recommender_metadata.json is not valid JSON"}
    return {"ready": artifacts_present(), "artifactsDir": str(settings.recommender_dir),
            "metadata": metadata, "loaded": _BUNDLE is not None}

def get_bundle() -> RecommenderBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        if not artifacts_present():
            raise RecommenderNotReadyError(
                "TF-IDF artifacts are not built. Run training/build_recommender.py after placing "
                "WebScrap_Dataset_final.xlsx in data/.", {"expectedDir": str(settings.recommender_dir)})
        import joblib
        d = settings.recommender_dir
        vectorizer = joblib.load(d / VECTORIZER_FILE)
        matrix = joblib.load(d / MATRIX_FILE)
        records = json.loads((d / RECORDS_FILE).read_text(encoding="utf-8"))
        metadata = {}
        p = d / METADATA_FILE
        if p.exists():
            try:
                metadata = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        logger.info("Loaded Excel recommender with %d records", len(records))
        _BUNDLE = RecommenderBundle(vectorizer, matrix, records, metadata)
        return _BUNDLE

def reset_recommender_cache() -> None:
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None

def _query_terms(vectorizer: Any, query: str) -> list[str]:
    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    return list(dict.fromkeys(t for t in analyzer(query) if t in vocabulary))

def recommend(query: str, top_n: int | None = None) -> dict[str, Any]:
    text = (query or "").strip()
    if len(text) < 3:
        raise InvalidInputError("Please describe the symptoms in a little more detail (at least 3 characters).")
    from sklearn.metrics.pairwise import cosine_similarity
    bundle = get_bundle()
    n = top_n or settings.default_top_n
    similarities = cosine_similarity(bundle.vectorizer.transform([text]), bundle.matrix)[0]
    tokens = _query_terms(bundle.vectorizer, text)
    ranked = sorted(range(len(similarities)), key=lambda i: float(similarities[i]), reverse=True)
    matches = []
    for index in ranked:
        score = float(similarities[index])
        if score < MIN_SIMILARITY:
            break
        r = bundle.records[index]
        plant_record = get_plant_record(r.get("plantKey", r.get("plantName", "")))
        terms = set(r.get("terms", []))
        keywords = [t for t in tokens if t in terms]
        matches.append({
            "plantKey": r.get("plantKey"), "plantName": r.get("plantName"), "plant": plant_record,
            "score": score, "matchedKeywords": keywords,
            "matchedSymptoms": r.get("symptoms", []),
            "traditionalUses": r.get("traditionalUses", []), "remedies": r.get("remedies", []),
            "precautions": r.get("precautions", []), "sourceFiles": r.get("sourceFiles", []),
            "recordId": r.get("recordId"), "commonName": r.get("commonName", ""),
            "scientificName": r.get("botanicalName", ""), "partOfPlant": r.get("partOfPlant", ""),
            "preparation": r.get("preparation", ""), "doctorSource": r.get("doctorSource", ""),
            "url": r.get("url", ""), "scientificEvidence": r.get("scientificEvidence", ""),
            "safetyEvidence": r.get("safetyEvidence", ""),
            "explanation": f"Cosine similarity {score:.2f} against a record from WebScrap_Dataset_final.xlsx"
                + (f"; shared terms: {', '.join(keywords)}." if keywords else "."),
        })
        if len(matches) >= n:
            break
    return {
        "source": "WebScrap_Dataset_final.xlsx", "query": text, "tokens": tokens,
        "matches": matches, "createdAt": datetime.now(timezone.utc).isoformat(),
        "method": "sklearn TF-IDF + cosine similarity (fitted offline on Excel records)",
        "recordCount": len(bundle.records),
        "warnings": [] if matches else [
            "No record in WebScrap_Dataset_final.xlsx reached the minimum similarity threshold for this description."
        ],
        "disclaimer": "Educational and traditional herbal information only. A similarity match does not prove effectiveness or safety. This is not a diagnosis, prescription, or dosage instruction.",
    }

def compatibility(plant_record: dict[str, Any] | None, tokens: list[str]) -> dict[str, Any]:
    if not plant_record:
        return {"matchedSymptoms": [], "relevance": 0.0,
                "message": "No knowledge record exists for the identified class, so relevance cannot be computed."}
    fields = ["symptoms", "traditionalUses", "properties", "remedies", "precautions"]
    recorded = [str(v).lower() for f in fields for v in plant_record.get(f, [])]
    summary = str(plant_record.get("summary", "")).lower()
    if summary:
        recorded.append(summary)
    matched = list(dict.fromkeys(item for item in recorded if any(t.lower() in item for t in tokens)))
    relevance = 0.0 if not tokens or not recorded else min(1.0, len(matched) / max(1, min(len(tokens), 4)))
    message = (f"Recorded information for {plant_record.get('name')} mentions {', '.join(matched)}, which overlaps with the described symptoms."
               if matched else f"Recorded information for {plant_record.get('name')} does not mention the described symptoms.")
    return {"matchedSymptoms": matched, "relevance": relevance, "message": message}
