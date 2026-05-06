from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrosat.contracts import validate_submission_outputs
from hydrosat.data import load_training_records
from hydrosat.features import build_feature_table
from hydrosat.infer import main as infer_main
from hydrosat.train import candidate_models, evaluate_candidate, feature_columns


class TrainingAndInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = REPO_ROOT / "data" / "raw"
        if not cls.data_root.exists():
            raise unittest.SkipTest("Normalized raw data is not available in this checkout.")

    def test_grouped_cv_smoke_on_small_feature_batch(self) -> None:
        all_records = load_training_records(data_root=self.data_root, targets=("turbidity",))
        records = []
        for area in ("area1", "area2", "area3", "area5"):
            area_records = [
                record
                for record in all_records
                if record.area == area
            ][:3]
            records.extend(area_records)

        frame = build_feature_table(records, patch_size=32, progress_every=0)
        cols = feature_columns(frame)
        models = candidate_models(random_state=42)
        result = evaluate_candidate(
            frame=frame,
            target="turbidity",
            model_name="hgb",
            model=models["hgb"],
            variant="full",
            columns=cols,
            group_by="area",
            max_features=40,
        )
        self.assertGreaterEqual(float(result["rmse"]), 0.0)
        self.assertGreaterEqual(len(result["folds"]), 2)

    def test_sample_inference_outputs_validate_against_contract(self) -> None:
        model_paths = [REPO_ROOT / "models" / "turbidity.joblib", REPO_ROOT / "models" / "chla.joblib"]
        if not all(path.exists() for path in model_paths):
            raise unittest.SkipTest("Trained baseline model bundles are not available in this checkout.")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            exit_code = infer_main(
                [
                    "--input-root",
                    str(REPO_ROOT / "data" / "raw" / "sample_input"),
                    "--output-dir",
                    str(output_dir),
                    "--models-dir",
                    str(REPO_ROOT / "models"),
                ]
            )
            self.assertEqual(exit_code, 0)

            result_paths = validate_submission_outputs(output_dir)
            with result_paths["result_turbidity.json"].open("r", encoding="utf-8") as handle:
                turbidity = json.load(handle)
            with result_paths["result_chla.json"].open("r", encoding="utf-8") as handle:
                chla = json.load(handle)

            self.assertEqual(len(turbidity), 3)
            self.assertEqual(len(chla), 2)
