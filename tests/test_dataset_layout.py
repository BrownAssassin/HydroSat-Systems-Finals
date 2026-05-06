from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrosat.data import discover_input_layout, discover_training_layout


class DatasetLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = REPO_ROOT / "data" / "raw"
        if not cls.data_root.exists():
            raise unittest.SkipTest("Normalized raw data is not available in this checkout.")

    def test_training_layout_counts(self) -> None:
        layout = discover_training_layout(self.data_root)
        self.assertEqual([area.name for area in layout.areas], ["area1", "area2", "area3", "area5", "area6", "area7"])
        self.assertEqual(layout.total_turbidity_rows, 991)
        self.assertEqual(layout.total_chla_rows, 305)

    def test_sample_input_layout(self) -> None:
        layout = discover_input_layout(data_root=self.data_root)
        self.assertEqual(layout.image_dir.name, "area8_images")
        self.assertEqual(layout.turbidity_points.row_count, 3)
        self.assertEqual(layout.chla_points.row_count, 2)
