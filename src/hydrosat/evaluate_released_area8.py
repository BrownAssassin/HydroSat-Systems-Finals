from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .infer import infer_csv
from .scoring import parameter_metrics, pair_summary


def parse_prediction_key(key: str) -> tuple[str, str, str]:
    try:
        filename, lon, lat = key.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"Invalid prediction key: {key}") from exc
    return filename, lon, lat


def build_eval_csv(truth: dict[str, float], target: str, dest: Path) -> Path:
    value_column = "turb_value" if target == "turbidity" else "cha_value"
    rows = []
    for key in truth:
        filename, lon, lat = parse_prediction_key(key)
        rows.append(
            {
                "filename": filename,
                "Lon": lon,
                "Lat": lat,
                value_column: "",
            }
        )
    df = pd.DataFrame(rows, columns=["filename", "Lon", "Lat", value_column])
    csv_name = "track2_turb_test_point.csv" if target == "turbidity" else "track2_cha_test_point.csv"
    csv_path = dest / csv_name
    df.to_csv(csv_path, index=False)
    return csv_path


def canonical_key(key: str) -> tuple[str, float, float]:
    filename, lon, lat = parse_prediction_key(key)
    return filename, round(float(lon), 6), round(float(lat), 6)


def score_predictions(truth: dict[str, float], predictions: dict[str, list[float]]) -> dict[str, float]:
    truth_map = {canonical_key(key): float(value) for key, value in truth.items()}
    prediction_map = {canonical_key(key): float(value[0]) for key, value in predictions.items()}
    if set(truth_map) != set(prediction_map):
        missing = sorted(set(truth_map) - set(prediction_map))[:5]
        extra = sorted(set(prediction_map) - set(truth_map))[:5]
        raise ValueError(
            "Prediction keys do not match truth keys. "
            f"missing={missing} extra={extra}"
        )

    ordered_keys = list(truth_map.keys())
    y_true = [truth_map[key] for key in ordered_keys]
    y_pred = [prediction_map[key] for key in ordered_keys]
    return parameter_metrics(y_true, y_pred)


def write_prediction_outputs(output_dir: Path, turbidity: dict[str, list[float]], chla: dict[str, list[float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    turb_text = json.dumps(turbidity, indent=2)
    chla_text = json.dumps(chla, indent=2)
    (output_dir / "turbidity_result.json").write_text(turb_text, encoding="utf-8")
    (output_dir / "chla_result.json").write_text(chla_text, encoding="utf-8")
    (output_dir / "result_turbidity.json").write_text(turb_text, encoding="utf-8")
    (output_dir / "result_chla.json").write_text(chla_text, encoding="utf-8")


def evaluate_released_area8(
    released_root: Path,
    model_dir: Path,
    work_dir: Path,
    output_dir: Path,
    report_dir: Path,
    patch_size: int,
    progress_every: int,
) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    turb_truth_path = released_root / "track2_turb_test_true.json"
    chla_truth_path = released_root / "track2_cha_test_true.json"
    image_root = released_root / "area8_images"

    turb_truth = json.loads(turb_truth_path.read_text(encoding="utf-8"))
    chla_truth = json.loads(chla_truth_path.read_text(encoding="utf-8"))

    build_eval_csv(turb_truth, "turbidity", work_dir)
    build_eval_csv(chla_truth, "chla", work_dir)

    old_image_root = os.environ.get("HYDROSAT_IMAGE_ROOT")
    os.environ["HYDROSAT_IMAGE_ROOT"] = str(image_root)
    try:
        turb_predictions = infer_csv(
            work_dir,
            "track2_turb_test_point.csv",
            "turbidity",
            model_dir,
            patch_size,
            progress_every,
        )
        chla_predictions = infer_csv(
            work_dir,
            "track2_cha_test_point.csv",
            "chla",
            model_dir,
            patch_size,
            progress_every,
        )
    finally:
        if old_image_root is None:
            os.environ.pop("HYDROSAT_IMAGE_ROOT", None)
        else:
            os.environ["HYDROSAT_IMAGE_ROOT"] = old_image_root

    write_prediction_outputs(output_dir, turb_predictions, chla_predictions)

    turbidity_metrics = score_predictions(turb_truth, turb_predictions)
    chla_metrics = score_predictions(chla_truth, chla_predictions)
    summary = {
        "released_root": str(released_root.resolve()),
        "image_root": str(image_root.resolve()),
        "model_dir": str(model_dir.resolve()),
        "work_dir": str(work_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        **pair_summary(turbidity_metrics, chla_metrics),
        "scoring_formula": {
            "nrmse": "rmse / mean_truth",
            "parameter_score": "(0.5 * max(0, r2) + 0.5 * max(0, 1 - nrmse)) * 100",
            "algorithm_score": "0.5 * turbidity_score + 0.5 * chla_score",
        },
    }

    (report_dir / "released_area8_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md = "\n".join(
        [
            "# Released Area8 Local Evaluation",
            "",
            f"- Released root: `{summary['released_root']}`",
            f"- Image root: `{summary['image_root']}`",
            f"- Model dir: `{summary['model_dir']}`",
            f"- Output dir: `{summary['output_dir']}`",
            "",
            "## Turbidity",
            f"- Count: {turbidity_metrics['count']}",
            f"- RMSE: {turbidity_metrics['rmse']:.4f}",
            f"- R2: {turbidity_metrics['r2']:.4f}",
            f"- NRMSE: {turbidity_metrics['nrmse']:.4f}",
            f"- Score: {turbidity_metrics['score']:.4f}",
            "",
            "## Chl-a",
            f"- Count: {chla_metrics['count']}",
            f"- RMSE: {chla_metrics['rmse']:.4f}",
            f"- R2: {chla_metrics['r2']:.4f}",
            f"- NRMSE: {chla_metrics['nrmse']:.4f}",
            f"- Score: {chla_metrics['score']:.4f}",
            "",
            "## Final",
            f"- Algorithm score: {summary['algorithm_score']:.4f}",
        ]
    )
    (report_dir / "released_area8_scores.md").write_text(report_md, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--released-root", type=Path, default=Path("track2_download_link_1"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/eval_input/released_area8"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/output/released_area8"))
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/reports/released_area8"))
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    summary = evaluate_released_area8(
        released_root=args.released_root,
        model_dir=args.model_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        patch_size=args.patch_size,
        progress_every=args.progress_every,
    )
    print((args.report_dir / "released_area8_scores.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
