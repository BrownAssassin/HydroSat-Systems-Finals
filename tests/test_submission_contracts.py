from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrosat.contracts import (
    validate_prediction_key,
    validate_submission_outputs,
    write_prediction_file,
)


class SubmissionContractTest(unittest.TestCase):
    def test_write_and_validate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_prediction_file(
                output_dir,
                "turbidity",
                {"area8_2024-01-15.tif_-122.823396_44.498493": [1.25]},
            )
            write_prediction_file(
                output_dir,
                "chla",
                {"area8_2024-01-15.tif_-122.669408_45.517321": [0.75]},
            )

            result_paths = validate_submission_outputs(output_dir)
            self.assertEqual(set(result_paths), {"result_turbidity.json", "result_chla.json"})

    def test_invalid_prediction_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction_key("bad_key_without_coordinates")

    def test_invalid_prediction_value_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_prediction_file(
                    Path(tmpdir),
                    "turbidity",
                    {"area8_2024-01-15.tif_-122.823396_44.498493": [1.0, 2.0]},
                )
