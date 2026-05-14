from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from .scoring import parameter_metrics
from .train_baseline import candidate_models, feature_columns, make_pipeline, rmse, validation_groups


DEFAULT_MODELS = {
    "turbidity": ["hgb_log", "lightgbm_log", "xgboost_log", "extra_log", "catboost"],
    "chla": ["extra", "xgboost", "lightgbm_log", "hgb", "catboost"],
}

TEST_LABEL_STATS = {
    "chla": {"min": 0.18, "max": 5.3},
    "turbidity": {"min": 0.1, "max": 22.8},
}


def filter_to_test_range(part: pd.DataFrame, target: str, enabled: bool, padding: float) -> pd.DataFrame:
    if not enabled:
        return part
    stats = TEST_LABEL_STATS[target]
    low = stats["min"]
    high = stats["max"]
    span = high - low
    low -= span * padding
    high += span * padding
    filtered = part[(part["y"] >= low) & (part["y"] <= high)].copy()
    print(
        f"{target}: test-range filter kept {len(filtered)}/{len(part)} rows "
        f"for y in [{low:.4f}, {high:.4f}]",
        flush=True,
    )
    if len(filtered) < max(30, min(100, len(part) // 4)):
        print(f"{target}: filter kept too few rows; falling back to all rows", flush=True)
        return part
    return filtered


def _clip_fit_target(y: np.ndarray, upper_quantile: float | None) -> np.ndarray:
    if upper_quantile is None:
        return y
    upper = np.quantile(y, upper_quantile)
    return np.clip(y, 0, upper)


def _selection_sort_key(row: dict[str, float], selection_metric: str) -> tuple[float, float, float]:
    if selection_metric == "score":
        return (-float(row["score"]), float(row["rmse"]), -float(row["r2"]))
    return (float(row["rmse"]), -float(row["score"]), -float(row["r2"]))


def _member_weights(selected: list[dict[str, float]], selection_metric: str) -> list[float]:
    if not selected:
        return []
    if selection_metric == "score":
        raw = [max(float(row["score"]), 0.0) + 1e-6 for row in selected]
    else:
        raw = [1.0 / max(float(row["rmse"]), 1e-6) for row in selected]
    total = float(sum(raw))
    if total <= 0:
        return [1.0 / len(selected)] * len(selected)
    return [value / total for value in raw]


def train_target_ensemble(
    features_path: Path,
    target: str,
    model_dir: Path,
    model_names: list[str],
    top_n: int,
    max_features: int | None,
    random_state: int,
    clip_quantile: float | None,
    group_by: str,
    filter_test_range: bool,
    filter_range_padding: float,
    selection_metric: str,
    write_model: bool = True,
) -> dict:
    df = pd.read_csv(features_path)
    part = df[df["target"] == target].copy()
    if part.empty:
        raise RuntimeError(f"No rows for target {target} in {features_path}")
    part = filter_to_test_range(part, target, filter_test_range, filter_range_padding)

    cols = feature_columns(part)
    groups = validation_groups(part, group_by)
    n_splits = min(5, groups.nunique())
    cv = GroupKFold(n_splits=n_splits)
    X = part[cols]
    y = part["y"].to_numpy(dtype="float32")
    models = candidate_models(random_state)

    results = []
    for name in model_names:
        if name not in models:
            raise ValueError(f"Unknown model {name}. Available: {', '.join(models)}")
        print(f"ensemble cv {target}/{name}: rows={len(part)} features={len(cols)}", flush=True)
        preds = np.zeros(len(part), dtype="float32")
        for train_idx, val_idx in cv.split(X, y, groups=groups):
            pipe = make_pipeline(clone(models[name]), cols, max_features=max_features)
            pipe.fit(X.iloc[train_idx], _clip_fit_target(y[train_idx], clip_quantile))
            preds[val_idx] = pipe.predict(X.iloc[val_idx])
        preds = np.clip(preds, 0, None)
        metrics = parameter_metrics(y, preds)
        results.append(
            {
                "name": name,
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "nrmse": metrics["nrmse"],
                "score": metrics["score"],
                "oof_pred": preds,
            }
        )

    results.sort(key=lambda row: _selection_sort_key(row, selection_metric))
    selected = results[:top_n]
    weights = _member_weights(selected, selection_metric)
    ensemble_pred = np.average(
        np.vstack([row["oof_pred"] for row in selected]),
        axis=0,
        weights=np.asarray(weights, dtype="float64"),
    )
    ensemble_pred = np.clip(ensemble_pred, 0, None)
    ensemble_metrics = parameter_metrics(y, ensemble_pred)
    summary = {
        "target": target,
        "features_path": str(features_path),
        "rows": int(len(part)),
        "columns": int(len(cols)),
        "selected": [
            {
                "name": row["name"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "nrmse": row["nrmse"],
                "score": row["score"],
                "weight": weights[idx],
            }
            for idx, row in enumerate(selected)
        ],
        "candidate_results": [
            {
                "name": row["name"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "nrmse": row["nrmse"],
                "score": row["score"],
            }
            for row in results
        ],
        "ensemble_rmse": ensemble_metrics["rmse"],
        "ensemble_r2": ensemble_metrics["r2"],
        "ensemble_nrmse": ensemble_metrics["nrmse"],
        "ensemble_score": ensemble_metrics["score"],
        "clip_quantile": clip_quantile,
        "max_features": max_features,
        "group_by": group_by,
        "selection_metric": selection_metric,
        "filter_test_range": bool(filter_test_range),
        "filter_range_padding": float(filter_range_padding),
    }

    members = []
    if write_model:
        y_fit = _clip_fit_target(y, clip_quantile)
        for idx, row in enumerate(selected):
            name = row["name"]
            print(f"fitting final {target}/{name}", flush=True)
            pipe = make_pipeline(clone(models[name]), cols, max_features=max_features)
            pipe.fit(X, y_fit)
            members.append({"name": name, "pipeline": pipe, "columns": cols, "weight": weights[idx]})

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "ensemble",
                "target": target,
                "members": members,
                "summary": summary,
            },
            model_dir / f"{target}_ensemble.joblib",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-turbidity", type=Path, default=Path("artifacts/features/train_features_v2.csv"))
    parser.add_argument("--features-chla", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--target", choices=["turbidity", "chla", "both"], default="both")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--group-by", choices=["image", "area"], default="image")
    parser.add_argument("--turbidity-clip-quantile", type=float, default=None)
    parser.add_argument("--filter-test-range", action="store_true")
    parser.add_argument("--filter-range-padding", type=float, default=0.05)
    parser.add_argument("--selection-metric", choices=["score", "rmse"], default="score")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--no-write-models", action="store_true")
    parser.add_argument("--turbidity-models", default=",".join(DEFAULT_MODELS["turbidity"]))
    parser.add_argument("--chla-models", default=",".join(DEFAULT_MODELS["chla"]))
    args = parser.parse_args()

    summaries = []
    if args.target in {"turbidity", "both"}:
        summaries.append(
            train_target_ensemble(
                args.features_turbidity,
                "turbidity",
                args.model_dir,
                [name.strip() for name in args.turbidity_models.split(",") if name.strip()],
                args.top_n,
                args.max_features,
                args.random_state,
                args.turbidity_clip_quantile,
                args.group_by,
                args.filter_test_range,
                args.filter_range_padding,
                args.selection_metric,
                write_model=not args.no_write_models,
            )
        )
    if args.target in {"chla", "both"}:
        summaries.append(
            train_target_ensemble(
                args.features_chla,
                "chla",
                args.model_dir,
                [name.strip() for name in args.chla_models.split(",") if name.strip()],
                args.top_n,
                args.max_features,
                args.random_state,
                None,
                args.group_by,
                args.filter_test_range,
                args.filter_range_padding,
                args.selection_metric,
                write_model=not args.no_write_models,
            )
        )

    for summary in summaries:
        print(
            f"\n{summary['target']} ensemble: "
            f"score={summary['ensemble_score']:.4f} rmse={summary['ensemble_rmse']:.4f} "
            f"r2={summary['ensemble_r2']:.4f} nrmse={summary['ensemble_nrmse']:.4f}"
        )
        for row in summary["selected"]:
            print(
                f"  {row['name']}: score={row['score']:.4f} rmse={row['rmse']:.4f} "
                f"r2={row['r2']:.4f} nrmse={row['nrmse']:.4f} weight={row['weight']:.4f}"
            )
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
