from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrosat.data import load_input_records, load_training_records, records_to_dataframe


class DataRecordLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = REPO_ROOT / "data" / "raw"
        if not cls.data_root.exists():
            raise unittest.SkipTest("Normalized raw data is not available in this checkout.")

    def test_training_records_are_loaded_with_image_paths(self) -> None:
        records = load_training_records(data_root=self.data_root)
        self.assertEqual(len(records), 1296)
        self.assertTrue(records[0].image_path.exists())

        frame = records_to_dataframe(records)
        self.assertEqual(len(frame), 1296)
        self.assertEqual(sorted(frame["target"].value_counts().to_dict().items()), [("chla", 305), ("turbidity", 991)])

    def test_input_records_are_loaded_per_target(self) -> None:
        records = load_input_records(data_root=self.data_root)
        self.assertEqual(len(records), 5)
        self.assertEqual(sum(1 for record in records if record.target == "turbidity"), 3)
        self.assertEqual(sum(1 for record in records if record.target == "chla"), 2)
