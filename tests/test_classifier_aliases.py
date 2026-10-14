"""Checks the six approved classifier-label aliases resolve to existing CSV records."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import data_service

# Exact classifier label -> (expected existing record id, expected botanical name, display name)
APPROVED = {
    "Azadirachta Indica (Neem)": ("vembu", "Azadirachta indica", "Neem"),
    "Citrus Limon (Lemon)": ("elumitchai", "Citrus limon", "Nimbu"),
    "Mangifera Indica (Mango)": ("ma", "Mangifera indica", "Aam"),
    "Punica Granatum (Pomegranate)": ("mathulai", "Punica granatum", "Anar"),
    "Tabernaemontana Divaricata (Crape Jasmine)": ("nanthiyavattam", "Tabernaemontana divaricata", "Chandni"),
    "Trigonella Foenum-graecum (Fenugreek)": ("vendayam", "Trigonella feenugracum", "Methi"),
}


class ClassifierAliasTests(unittest.TestCase):
    def tearDown(self):
        data_service.load_plant_records.cache_clear()
        data_service.load_name_mapping.cache_clear()
        data_service.load_display_names.cache_clear()

    def test_labels_normalise_to_existing_ids(self):
        for label, (record_id, _, _) in APPROVED.items():
            self.assertEqual(data_service.normalise_plant_name(label), record_id, label)

    def test_lookup_returns_correct_record(self):
        # Original supplied names/botanical names only, used to exercise the real loader.
        with patch.object(data_service, "resolve_data_file", return_value=data_service.DISPLAY_MAPPING_FILE):
            data_service.load_plant_records.cache_clear()
            for label, (record_id, botanical, display) in APPROVED.items():
                record = data_service.get_plant_record(label)
                self.assertIsNotNone(record, label)
                self.assertEqual(record["id"], record_id)
                self.assertEqual(record["classLabel"], record_id)
                self.assertEqual(record["botanicalName"], botanical)
                self.assertEqual(record["name"], display)


if __name__ == "__main__":
    unittest.main()
