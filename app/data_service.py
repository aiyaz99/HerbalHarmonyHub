"""Loads the user-supplied real CSV datasets and exposes plant knowledge records.

NOTHING in this module invents data. Every field returned comes from a CSV the
user placed under ``data/``. If a file is missing, the caller receives an explicit
``DatasetNotReadyError`` or ``None`` — never a placeholder record.

Expected files (either the plain name or the "(1)"-suffixed copy is accepted):
    data/medical plants.csv                     or data/medical plants(1).csv
    data/disease_symptom_dataset1_norm.csv      or ...(1).csv
    data/AyurGenixAI_Dataset2.csv               or ...(1).csv
    data/images/    extracted "Segmented Medicinal Leaf Images_Dataset 3.zip"
                    (medical_plant_images.zip is deliberately NOT used)

"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import settings
from .errors import DatasetNotReadyError
from .logging_config import get_logger

logger = get_logger(__name__)

MEDICAL_PLANTS_FILE = "medical plants.csv"
DISEASE_SYMPTOM_FILE = "disease_symptom_dataset1_norm.csv"
AYURGENIX_FILE = "AyurGenixAI_Dataset2.csv"
IMAGE_ZIP_NAME = "Segmented Medicinal Leaf Images_Dataset 3.zip"
IMAGE_DIR_NAME = "images"
MAPPING_FILE = Path(__file__).resolve().parent / "plant_name_mapping.json"


def _variants(name: str) -> list[str]:
    """Accept both the plain filename and the '(1)'-suffixed download copy."""
    stem, _, ext = name.rpartition(".")
    return [name, f"{stem}(1).{ext}"]


def resolve_data_file(name: str, data_dir: Path | None = None) -> Path | None:
    """Returns the first existing variant of a dataset filename, else None."""
    base = data_dir or settings.data_dir
    for candidate in _variants(name):
        path = base / candidate
        if path.exists():
            return path
    return None


def expected_data_paths(name: str, data_dir: Path | None = None) -> list[str]:
    base = data_dir or settings.data_dir
    return [str(base / candidate) for candidate in _variants(name)]



# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def normalise_plant_name(value: str) -> str:
    """Normalise a class-folder / CSV plant name through the explicit mapping only.

    The mapping file is documented and editable; unmapped names are returned
    slug-normalised, never guessed into a different species.
    """
    key = _slug(value)
    mapping = load_name_mapping()
    return mapping.get(key, key)


@lru_cache(maxsize=1)
def load_name_mapping() -> dict[str, str]:
    if not MAPPING_FILE.exists():
        return {}
    try:
        raw = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - config error
        logger.error("plant_name_mapping.json is not valid JSON: %s", exc)
        return {}
    return {_slug(k): _slug(v) for k, v in raw.get("aliases", {}).items()}


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    parts = re.split(r"[;|\n]+|(?<=[a-z\)])\s*,\s*", text)
    return [p.strip(" .;-") for p in parts if p and p.strip(" .;-")]


def _first_column(columns: list[str], *candidates: str) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:
        for low, original in lowered.items():
            if candidate in low:
                return original
    return None


# --------------------------------------------------------------------------- #
# dataset access
# --------------------------------------------------------------------------- #
def dataset_status() -> dict[str, Any]:
    """Reports which real files are actually present. No assumptions."""
    data_dir = settings.data_dir
    image_dir = data_dir / IMAGE_DIR_NAME
    classes: list[str] = []
    if image_dir.is_dir():
        classes = sorted(p.name for p in image_dir.iterdir() if p.is_dir())
    return {
        "dataDir": str(data_dir),
        "medicalPlantsCsv": resolve_data_file(MEDICAL_PLANTS_FILE) is not None,
        "diseaseSymptomCsv": resolve_data_file(DISEASE_SYMPTOM_FILE) is not None,
        "ayurGenixCsv": resolve_data_file(AYURGENIX_FILE) is not None,

        "imageDatasetExtracted": image_dir.is_dir(),
        "imageClassFolderCount": len(classes),
        "imageClassFolders": classes,
    }


@lru_cache(maxsize=1)
def load_plant_records() -> dict[str, dict[str, Any]]:
    """Builds plant knowledge records from the supplied medicinal plant CSV.

    Returns a mapping of normalised plant key -> record. Raises when the file is
    absent so the API can answer with an explicit dataset error.
    """
    import pandas as pd  # local import keeps app import cheap

    csv_path = resolve_data_file(MEDICAL_PLANTS_FILE)
    if csv_path is None:
        raise DatasetNotReadyError(
            f"Plant knowledge CSV not found. Place '{MEDICAL_PLANTS_FILE}' in {settings.data_dir}.",
            {"expectedPaths": expected_data_paths(MEDICAL_PLANTS_FILE)},
        )


    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).strip() for c in frame.columns]
    columns = list(frame.columns)

    col_name = _first_column(columns, "plant name", "common name", "common name (in tamil)", "name", "plant")
    if col_name is None:
        raise DatasetNotReadyError(
            "Could not find a plant-name column in the plant knowledge CSV.",
            {"columns": columns},
        )
    col_botanical = _first_column(columns, "botanical name", "scientific name", "species")
    col_family = _first_column(columns, "family")
    col_properties = _first_column(columns, "properties", "medicinal properties", "chemical")
    col_uses = _first_column(columns, "traditional uses", "uses", "medicinal uses", "benefits", "application")
    col_symptoms = _first_column(columns, "symptoms", "diseases", "conditions", "indications")
    col_remedies = _first_column(columns, "remedies", "remedy", "preparation", "formulation")
    col_precautions = _first_column(columns, "precautions", "side effects", "contraindications", "caution")
    col_summary = _first_column(columns, "description", "summary", "about")
    col_local = _first_column(columns, "local names", "regional names", "other names", "hindi")

    records: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        raw_name = str(row[col_name]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        key = normalise_plant_name(raw_name)
        uses = _split_list(row[col_uses]) if col_uses else []
        remedies = _split_list(row[col_remedies]) if col_remedies else []
        record = {
            "id": key,
            "classLabel": key,
            "name": raw_name,
            "localNames": _split_list(row[col_local]) if col_local else [],
            "botanicalName": str(row[col_botanical]).strip() if col_botanical else "",
            "family": str(row[col_family]).strip() if col_family else "",
            "summary": str(row[col_summary]).strip() if col_summary else "",
            "properties": _split_list(row[col_properties]) if col_properties else [],
            "traditionalUses": uses,
            "symptoms": [s.lower() for s in (_split_list(row[col_symptoms]) if col_symptoms else [])],
            "remedies": [{"title": "Traditional preparation", "description": item} for item in remedies],
            "precautions": _split_list(row[col_precautions]) if col_precautions else [],
            "source": MEDICAL_PLANTS_FILE,
        }
        # Merge duplicate rows for the same plant instead of dropping information.
        if key in records:
            existing = records[key]
            for field in ("localNames", "properties", "traditionalUses", "symptoms", "precautions"):
                merged = existing[field] + [v for v in record[field] if v not in existing[field]]
                existing[field] = merged
            existing["remedies"].extend(record["remedies"])
        else:
            records[key] = record
    logger.info("Loaded %d plant knowledge records from %s", len(records), csv_path.name)
    return records


def get_plant_record(class_label: str) -> dict[str, Any] | None:
    """Returns the knowledge record for a predicted class label, or None.

    ``None`` means "the supplied CSV has no row for this class" — the API says so
    explicitly instead of filling in invented botanical details.
    """
    try:
        records = load_plant_records()
    except DatasetNotReadyError:
        return None
    key = normalise_plant_name(class_label)
    if key in records:
        return records[key]
    # Allow a documented one-directional containment match (e.g. "neem-leaf" -> "neem").
    for candidate_key, record in records.items():
        if key.startswith(f"{candidate_key}-") or candidate_key.startswith(f"{key}-"):
            return record
    return None
