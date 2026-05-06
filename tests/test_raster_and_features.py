from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrosat.data import load_training_records
from hydrosat.features import build_feature_table, feature_row
from hydrosat.raster import read_patch


class RasterAndFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = REPO_ROOT / "data" / "raw"
        if not cls.data_root.exists():
            raise unittest.SkipTest("Normalized raw data is not available in this checkout.")
        cls.record = load_training_records(data_root=cls.data_root, targets=("turbidity",))[0]

    def test_read_patch_returns_expected_shape(self) -> None:
        patch = read_patch(self.record.image_path, self.record.lon, self.record.lat, size=32)
        self.assertEqual(patch.data.shape[1:], (32, 32))
        self.assertGreaterEqual(patch.data.shape[0], 12)
        self.assertGreater(patch.profile.width, 0)
        self.assertGreater(patch.profile.height, 0)

    def test_feature_row_is_deterministic_and_dense(self) -> None:
        first = feature_row(self.record, patch_size=32)
        second = feature_row(self.record, patch_size=32)
        self.assertEqual(first["filename"], self.record.filename)
        self.assertEqual(first["area"], self.record.area)
        self.assertAlmostEqual(float(first["b01_mean"]), float(second["b01_mean"]), places=6)
        self.assertAlmostEqual(float(first["water_fraction"]), float(second["water_fraction"]), places=6)
        numeric_feature_count = sum(
            1
            for key, value in first.items()
            if key not in {"target", "filename", "image_path", "area", "source_csv"} and isinstance(value, (int, float))
        )
        self.assertGreater(numeric_feature_count, 1000)
        self.assertTrue(math.isfinite(float(first["season_sin"])))

    def test_build_feature_table_for_small_batch(self) -> None:
        records = load_training_records(data_root=self.data_root, targets=("turbidity",))[:3]
        frame = build_feature_table(records, patch_size=32, progress_every=0)
        self.assertEqual(len(frame), 3)
        self.assertIn("ndwi_green_nir_mean", frame.columns)
        self.assertIn("center3_b01_p50", frame.columns)
