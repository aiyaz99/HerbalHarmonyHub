
"""Build an Excel-only TF-IDF recommender.

Run from repository root:
    python training/build_recommender.py
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "WebScrap_Dataset_final.xlsx"
OUT = ROOT / "artifacts" / "recommender"

SOURCE_FILENAME = "WebScrap_Dataset_final.xlsx"
SOURCE_SHEET = "WebScrap_Dataset"

COLS = {
    "scientificName": "Scientific_name",
    "commonName": "Common_name",
    "symptom": "Disease/Symptom",
    "partOfPlant": "Part of Plant",
    "remedy": "Remedy",
    "preparation": "Preparation",
    "doctorSource": "Doctor/Source",
    "url": "URL",
    "scientificEvidence": "Scientific Evidence",
    "safetyEvidence": "Safety Evidence",
}


def clean(value):
    """Convert Excel values to clean strings."""
    if value is None or pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def main():

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Workbook not found at {SOURCE}. Put it in data/ first."
        )

    # Read the specified Excel worksheet.
    df = pd.read_excel(
        SOURCE,
        sheet_name=SOURCE_SHEET,
        engine="openpyxl",
    )

    df.columns = [str(c).strip() for c in df.columns]

    # Validate all ten required columns.
    missing = [
        column
        for column in COLS.values()
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing workbook columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    records = []
    documents = []

    for idx, row in df.iterrows():

        # Extract all ten original Excel values.
        v = {
            key: clean(row[column])
            for key, column in COLS.items()
        }

        # Ignore completely empty rows.
        if not any(v.values()):
            continue

        plant_name = (
            v["commonName"]
            or v["scientificName"]
            or "Plant not specified"
        )

        # Text used for TF-IDF matching.
        document = " ".join(
            value
            for value in [
                v["scientificName"],
                v["commonName"],
                v["symptom"],
                v["partOfPlant"],
                v["remedy"],
                v["preparation"],
                v["doctorSource"],
                v["scientificEvidence"],
                v["safetyEvidence"],
            ]
            if value
        )

        if not document:
            continue

        # Preserve the original Excel column names and values.
        source_record = {
            "Scientific_name": v["scientificName"],
            "Common_name": v["commonName"],
            "Disease/Symptom": v["symptom"],
            "Part of Plant": v["partOfPlant"],
            "Remedy": v["remedy"],
            "Preparation": v["preparation"],
            "Doctor/Source": v["doctorSource"],
            "URL": v["url"],
            "Scientific Evidence": v["scientificEvidence"],
            "Safety Evidence": v["safetyEvidence"],
        }

        # Excel row number (header is row 1).
        excel_row = idx + 2

        documents.append(document)

        records.append({

            # Record identification.
            "recordId": f"excel-row-{excel_row}",
            "sourceRow": excel_row,
            "sourceDataset": SOURCE_FILENAME,
            "sourceFiles": [SOURCE_FILENAME],

            # Plant information.
            "plantKey": v["scientificName"] or plant_name,
            "plantName": plant_name,
            "botanicalName": v["scientificName"],
            "commonName": v["commonName"],

            # Recommendation information.
            "symptoms": [v["symptom"]] if v["symptom"] else [],
            "diseases": [v["symptom"]] if v["symptom"] else [],
            "traditionalUses": [],
            "remedies": [v["remedy"]] if v["remedy"] else [],
            "precautions": (
                [v["safetyEvidence"]]
                if v["safetyEvidence"]
                else []
            ),

            # Individual source fields (backward compatibility).
            "partOfPlant": v["partOfPlant"],
            "preparation": v["preparation"],
            "doctorSource": v["doctorSource"],
            "url": v["url"],
            "scientificEvidence": v["scientificEvidence"],
            "safetyEvidence": v["safetyEvidence"],

            # IMPORTANT:
            # Preserve the complete original Excel row.
            "sourceRecord": source_record,
        })

    if not records:
        raise ValueError(
            "No usable records found in workbook."
        )

    # Build TF-IDF vectorizer.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        strip_accents="unicode",
    )

    matrix = vectorizer.fit_transform(documents)

    # Store searchable terms for every record.
    analyzer = vectorizer.build_analyzer()

    for record, document in zip(records, documents):
        record["terms"] = list(
            dict.fromkeys(analyzer(document))
        )

    # Save all recommender artifacts.
    OUT.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        vectorizer,
        OUT / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        matrix,
        OUT / "tfidf_matrix.joblib",
    )

    (OUT / "records.json").write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save metadata.
    metadata = {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "sourceFiles": [
            f"data/{SOURCE_FILENAME}"
        ],
        "sourceDataset": SOURCE_FILENAME,
        "worksheet": SOURCE_SHEET,
        "recordCount": len(records),
        "vocabularySize": len(vectorizer.vocabulary_),
        "matrixShape": list(matrix.shape),
        "method": (
            "sklearn TfidfVectorizer + "
            "cosine_similarity at query time"
        ),
        "ngramRange": [1, 2],
        "datasetOnly": True,
    }

    (OUT / "recommender_metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Final validation.
    assert len(records) == matrix.shape[0]

    assert all(
        len(record["sourceRecord"]) == 10
        for record in records
    )

    print("\nExcel-only recommender built successfully!")
    print("-------------------------------------------")
    print(f"Source workbook : {SOURCE_FILENAME}")
    print(f"Worksheet      : {SOURCE_SHEET}")
    print(f"Total records  : {len(records)}")
    print(f"Vocabulary     : {len(vectorizer.vocabulary_)}")
    print(f"Matrix shape   : {matrix.shape}")
    print(f"Artifacts      : {OUT}")

    # Verify the particular Excel row 302.
    row_302 = next(
        (
            record
            for record in records
            if record["sourceRow"] == 302
        ),
        None,
    )

    if row_302:
        print("\nExcel row 302 verification:")
        print(
            json.dumps(
                row_302["sourceRecord"],
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
