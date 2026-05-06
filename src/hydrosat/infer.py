"""Inference entrypoint scaffold for the final round."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import DEFAULT_DATA_ROOT, DEFAULT_MODELS_DIR
from .contracts import CHLA_RESULT_FILENAME, TURBIDITY_RESULT_FILENAME
from .data import discover_input_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HydroSat Track 2 inference scaffold.")
    parser.add_argument("--input-root", type=Path, default=None, help="Competition input root or local sample input root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where submission JSON files will be written.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Normalized local data root.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing deployable model artifacts.")
    parser.add_argument("--check-layout", action="store_true", help="Validate the input layout and exit without running inference.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_root = args.input_root
    if input_root is None:
        env_input = Path(_getenv("INPUT_DIR", "/input"))
        input_root = env_input if env_input.exists() else args.data_root / "sample_input"

    output_dir = args.output_dir or Path(_getenv("OUTPUT_DIR", "artifacts/output"))
    models_dir = args.models_dir if args.models_dir is not None else Path(_getenv("HYDROSAT_MODELS_DIR", "models"))
    models_dir = Path(_getenv("HYDROSAT_MODELS_DIR", str(models_dir)))

    try:
        layout = discover_input_layout(input_root=input_root, data_root=args.data_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        _print_layout_summary(layout, output_dir, models_dir)

        if args.check_layout:
            return 0

        model_files = [path for path in models_dir.glob("*") if path.is_file() and not path.name.startswith(".")]
        if not model_files:
            raise RuntimeError(
                "No deployable model artifacts were found in "
                f"{models_dir}. Add weights under that directory before running inference."
            )

        raise RuntimeError(
            "Inference is scaffolded but not implemented yet. Replace hydrosat.infer.main() "
            "with real model loading and write the final outputs via "
            f"'{TURBIDITY_RESULT_FILENAME}' and '{CHLA_RESULT_FILENAME}'."
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI behavior
        print(f"[hydrosat.infer] {exc}", file=sys.stderr)
        return 2


def _getenv(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def _print_layout_summary(layout, output_dir: Path, models_dir: Path) -> None:
    print("HydroSat inference scaffold")
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
