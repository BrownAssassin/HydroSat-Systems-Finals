from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error


def parameter_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype="float64")
    y_pred_arr = np.asarray(y_pred, dtype="float64")
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mean_truth = float(np.mean(y_true_arr)) if y_true_arr.size else 0.0
    nrmse = float(rmse / mean_truth) if mean_truth else float("inf")
    score = float((0.5 * max(0.0, r2) + 0.5 * max(0.0, 1.0 - nrmse)) * 100.0)
    return {
        "count": int(y_true_arr.size),
        "rmse": rmse,
        "r2": r2,
        "mean_truth": mean_truth,
        "nrmse": nrmse,
        "score": score,
    }


def algorithm_score(turbidity_metrics: Mapping[str, float], chla_metrics: Mapping[str, float]) -> float:
    return float(0.5 * float(turbidity_metrics["score"]) + 0.5 * float(chla_metrics["score"]))


def pair_summary(turbidity_metrics: Mapping[str, float], chla_metrics: Mapping[str, float]) -> dict[str, object]:
    return {
        "turbidity": dict(turbidity_metrics),
        "chla": dict(chla_metrics),
        "algorithm_score": algorithm_score(turbidity_metrics, chla_metrics),
    }
