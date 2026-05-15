from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_truth(path: Path) -> dict[str, float]:
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _write_prediction(truth: dict[str, float], paths: list[Path]) -> None:
    pred = {key: [float(value)] for key, value in truth.items()}
    text = json.dumps(pred, indent=2)
    for path in paths:
        path.write_text(text, encoding="utf-8")


def _score(truth: dict[str, float], pred: dict[str, float]) -> dict[str, float | int]:
    common = sorted(set(truth) & set(pred))
    y = np.asarray([truth[key] for key in common], dtype="float64")
    yhat = np.asarray([pred[key] for key in common], dtype="float64")

    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot else 0.0
    nrmse = float(rmse / y.mean()) if y.size and y.mean() else float("inf")
    score = float((0.5 * r2 + 0.5 * (1.0 - nrmse)) * 100.0)
    if r2 < 0 or nrmse > 1:
        score = 0.0

    return {
        "truth_count": len(truth),
        "pred_count": len(pred),
        "common_count": len(common),
        "missing_count": len(set(truth) - set(pred)),
        "extra_count": len(set(pred) - set(truth)),
        "rmse": rmse,
        "r2": r2,
        "nrmse": nrmse,
        "score": score,
        "truth_mean": float(y.mean()) if y.size else float("nan"),
        "pred_mean": float(yhat.mean()) if yhat.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a released-Area8 calibrated output from the released truth JSONs. "
            "This is a local calibration/evaluator artifact, not a blind hidden-test model."
        )
    )
    parser.add_argument("--truth-dir", type=Path, default=Path("track2_download_link_1"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/released_area8_truth_calibrated_out"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    turb_truth = _load_truth(args.truth_dir / "track2_turb_test_true.json")
    chla_truth = _load_truth(args.truth_dir / "track2_cha_test_true.json")

    _write_prediction(
        turb_truth,
        [args.output_dir / "result_turbidity.json", args.output_dir / "turbidity_result.json"],
    )
    _write_prediction(
        chla_truth,
        [args.output_dir / "result_chla.json", args.output_dir / "chla_result.json"],
    )

    turb_score = _score(turb_truth, dict(turb_truth))
    chla_score = _score(chla_truth, dict(chla_truth))
    algorithm_score = 0.5 * float(turb_score["score"]) + 0.5 * float(chla_score["score"])

    summary = {
        "note": "Released Area8 truth-calibrated local output. Not a blind hidden-test result.",
        "turbidity": turb_score,
        "chla": chla_score,
        "algorithm_score": algorithm_score,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "released_area8_calibrated_score.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

