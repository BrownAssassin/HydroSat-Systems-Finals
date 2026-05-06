"""Inference entrypoint for the final-round tabular baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*LGBMRegressor.*",
    category=UserWarning,
)

from .config import (
    ALL_TARGETS,
    DEFAULT_DATA_ROOT,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUT_DIR,
)
from .contracts import make_prediction_key, write_prediction_file
from .data import discover_input_layout, load_input_records
from .features import build_feature_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HydroSat Track 2 baseline inference.")
    parser.add_argument("--input-root", type=Path, default=None, help="Competition input root or local sample input root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where submission JSON files will be written.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Normalized local data root.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing deployable model artifacts.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N rows during feature extraction.")
    parser.add_argument("--check-layout", action="store_true", help="Validate the input layout and exit without running inference.")
    return parser


def load_model_bundle(models_dir: Path, target: str) -> dict[str, object]:
    """Load one saved model bundle for inference."""

    model_path = models_dir / f"{target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model bundle for target '{target}': {model_path}")
    return joblib.load(model_path)


def predict_target(
    input_root: Path,
    target: str,
    models_dir: Path,
    progress_every: int,
    data_root: Path,
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """Generate contract-ready predictions for one target."""

    bundle = load_model_bundle(models_dir=models_dir, target=target)
    records = load_input_records(input_root=input_root, data_root=data_root, targets=(target,))
    feature_frame = build_feature_table(
        records,
        patch_size=int(bundle["patch_size"]),
        progress_every=progress_every,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*valid feature names.*LGBMRegressor.*",
            category=UserWarning,
        )
        predictions = np.clip(bundle["pipeline"].predict(feature_frame), 0, None)

    payload: dict[str, list[float]] = {}
    for record, prediction in zip(records, predictions):
        payload[make_prediction_key(record.filename, record.lon, record.lat)] = [float(prediction)]
    return payload, feature_frame


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_root = args.input_root
    if input_root is None:
        env_input = Path(_getenv("INPUT_DIR", "/input"))
        input_root = env_input if env_input.exists() else args.data_root / "sample_input"

    output_dir = args.output_dir or Path(_getenv("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    models_dir = Path(_getenv("HYDROSAT_MODELS_DIR", str(args.models_dir)))

    try:
        layout = discover_input_layout(input_root=input_root, data_root=args.data_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        _print_layout_summary(layout, output_dir, models_dir)

        if args.check_layout:
            return 0

        for target in ALL_TARGETS:
            payload, feature_frame = predict_target(
                input_root=layout.root,
                target=target,
                models_dir=models_dir,
                progress_every=args.progress_every,
                data_root=args.data_root,
            )
            output_path = write_prediction_file(output_dir, target, payload)
            print(
                f"{target}: wrote {len(payload)} predictions to {output_path} "
                f"using {len(feature_frame)} feature rows",
                flush=True,
            )
        return 0
    except Exception as exc:  # pragma: no cover - exercised through CLI behavior
        print(f"[hydrosat.infer] {exc}", file=sys.stderr)
        return 2


def _getenv(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def _print_layout_summary(layout, output_dir: Path, models_dir: Path) -> None:
    print("HydroSat baseline inference")
    print(f"Input root: {layout.root}")
    print(f"Image directory: {layout.image_dir}")
    print(
        "Turbidity rows: "
        f"{layout.turbidity_points.row_count} "
        f"across {layout.turbidity_points.unique_filenames} file(s)"
    )
    print(
        "Chl-a rows: "
        f"{layout.chla_points.row_count} "
        f"across {layout.chla_points.unique_filenames} file(s)"
    )
    print(f"Output directory: {output_dir}")
    print(f"Models directory: {models_dir}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
