"""Track 2 output-contract helpers."""

from __future__ import annotations

from pathlib import Path
import json
import re
from numbers import Real

from .config import CHLA_RESULT_FILENAME, TURBIDITY_RESULT_FILENAME

_PREDICTION_KEY_PATTERN = re.compile(
    r"^(?P<filename>.+\.tif)_(?P<lon>-?\d+(?:\.\d+)?)_(?P<lat>-?\d+(?:\.\d+)?)$"
)

_TARGET_TO_FILENAME = {
    "turbidity": TURBIDITY_RESULT_FILENAME,
    "chla": CHLA_RESULT_FILENAME,
}


def make_prediction_key(filename: str, lon: float | str, lat: float | str) -> str:
    """Build the canonical Track 2 prediction key."""

    return f"{filename}_{lon}_{lat}"


def validate_prediction_key(key: str) -> None:
    """Validate one Track 2 prediction key."""

    if not _PREDICTION_KEY_PATTERN.match(key):
        raise ValueError(
            "Prediction keys must follow the form "
            "'{filename}_{Lon}_{Lat}' with a .tif filename."
        )


def validate_prediction_mapping(predictions: dict[str, list[float]]) -> dict[str, list[float]]:
    """Validate and normalize one submission mapping."""

    normalized: dict[str, list[float]] = {}
    for key, value in predictions.items():
        validate_prediction_key(key)
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError(f"Prediction value for '{key}' must be a single-item list.")
        scalar = value[0]
        if isinstance(scalar, bool) or not isinstance(scalar, Real):
            raise ValueError(f"Prediction value for '{key}' must be numeric.")
        normalized[key] = [float(scalar)]
    return normalized


def result_filename_for_target(target: str) -> str:
    """Return the required result filename for one target."""

    try:
        return _TARGET_TO_FILENAME[target]
    except KeyError as exc:
        raise ValueError(f"Unsupported target '{target}'. Expected one of {sorted(_TARGET_TO_FILENAME)}.") from exc


def write_prediction_file(output_dir: Path | str, target: str, predictions: dict[str, list[float]]) -> Path:
    """Write one validated prediction JSON to the correct Track 2 filename."""

    normalized = validate_prediction_mapping(predictions)
    output_path = Path(output_dir) / result_filename_for_target(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
    return output_path


def validate_submission_outputs(output_dir: Path | str) -> dict[str, Path]:
    """Validate that both required Track 2 output files exist and match the contract."""

    output_root = Path(output_dir)
    result_paths = {
        TURBIDITY_RESULT_FILENAME: output_root / TURBIDITY_RESULT_FILENAME,
        CHLA_RESULT_FILENAME: output_root / CHLA_RESULT_FILENAME,
    }
    for path in result_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing required submission output: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Submission file must contain a JSON object: {path}")
        validate_prediction_mapping(payload)
    return result_paths
