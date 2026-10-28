"""Build an Excel-only TF-IDF recommender. Run from repo root: python training/build_recommender.py"""
from pathlib import Path
from datetime import datetime, timezone
import json, re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "WebScrap_Dataset_final.xlsx"
OUT = ROOT / "artifacts" / "recommender"
COLS = {
    "scientificName": "Scientific_name", "commonName": "Common_name",
    "symptom": "Disease/Symptom", "partOfPlant": "Part of Plant",
    "remedy": "Remedy", "preparation": "Preparation",
    "doctorSource": "Doctor/Source", "url": "URL",
    "scientificEvidence": "Scientific Evidence", "safetyEvidence": "Safety Evidence",
}

def clean(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\\s+", " ", str(value)).strip()

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Workbook not found at {SOURCE}. Put it in data/ first.")
    df = pd.read_excel(SOURCE, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in COLS.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing workbook columns: {missing}. Found: {list(df.columns)}")

    records, documents = [], []
    for idx, row in df.iterrows():
        v = {key: clean(row[col]) for key, col in COLS.items()}
        if not any(v.values()):
            continue
        plant_name = v["commonName"] or v["scientificName"] or "Plant not specified"
        document = " ".join(x for x in [
            v["scientificName"], v["commonName"], v["symptom"], v["partOfPlant"],
            v["remedy"], v["preparation"], v["doctorSource"],
            v["scientificEvidence"], v["safetyEvidence"]
        ] if x)
        if not document:
            continue
        documents.append(document)
        records.append({
            "recordId": f"excel-row-{idx + 2}",
            "plantKey": v["scientificName"] or plant_name,
            "plantName": plant_name,
            "botanicalName": v["scientificName"],
            "commonName": v["commonName"],
            "symptoms": [v["symptom"]] if v["symptom"] else [],
            "diseases": [v["symptom"]] if v["symptom"] else [],
            "traditionalUses": [],
            "remedies": [v["remedy"]] if v["remedy"] else [],
            "precautions": [v["safetyEvidence"]] if v["safetyEvidence"] else [],
            "partOfPlant": v["partOfPlant"],
            "preparation": v["preparation"],
            "doctorSource": v["doctorSource"],
            "url": v["url"],
            "scientificEvidence": v["scientificEvidence"],
            "safetyEvidence": v["safetyEvidence"],
            "sourceFiles": ["WebScrap_Dataset_final.xlsx"],
        })

    if not records:
        raise ValueError("No usable records found in workbook.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode")
    matrix = vectorizer.fit_transform(documents)
    analyzer = vectorizer.build_analyzer()
    for rec, doc in zip(records, documents):
        rec["terms"] = list(dict.fromkeys(analyzer(doc)))

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, OUT / "tfidf_matrix.joblib")
    (OUT / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "sourceFiles": ["data/WebScrap_Dataset_final.xlsx"],
        "recordCount": len(records),
        "vocabularySize": len(vectorizer.vocabulary_),
        "matrixShape": list(matrix.shape),
        "method": "sklearn TfidfVectorizer + cosine similarity",
        "ngramRange": [1, 2],
        "datasetOnly": True,
    }
    (OUT / "recommender_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Built Excel-only recommender: {len(records)} records, {len(vectorizer.vocabulary_)} vocabulary terms.")
    print(f"Artifacts saved to {OUT}")

if __name__ == "__main__":
    main()
