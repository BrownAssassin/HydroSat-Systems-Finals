from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor

from .scoring import parameter_metrics

META_COLUMNS = {
    "filename",
    "image_path",
    "area",
    "target",
    "y",
    "lon",
    "lat",
    "pixel_row",
    "pixel_col",
    "pixel_row_norm",
    "pixel_col_norm",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def candidate_models(random_state: int):
    raw_models = {
        "hgb": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=random_state),
        "extra": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=random_state),
    }
    try:
        from lightgbm import LGBMRegressor

        raw_models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=24,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor

        raw_models["xgboost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor

        raw_models["catboost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.04,
            depth=5,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    except ImportError:
        pass
    models = dict(raw_models)
    for name, model in raw_models.items():
        models[f"{name}_log"] = TransformedTargetRegressor(
            regressor=clone(model),
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return models


def make_pipeline(model, columns: list[str], max_features: int | None) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    pre = ColumnTransformer([("num", numeric, columns)], remainder="drop")
    steps = [("pre", pre)]
    if max_features is not None and max_features > 0 and max_features < len(columns):
        steps.append(("select", SelectKBest(score_func=f_regression, k=max_features)))
    steps.append(("model", model))
    return Pipeline(steps)


def rmse(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def validation_groups(part: pd.DataFrame, group_by: str) -> pd.Series:
    if group_by == "area":
        return part["area"].astype(str)
    if group_by == "image":
        return part["area"].astype(str) + "_" + part["filename"].astype(str)
    raise ValueError(f"Unknown group_by: {group_by}")


def train_one(df: pd.DataFrame, target: str, model_dir: Path, random_state: int, max_features: int | None, group_by: str) -> dict:
    part = df[df["target"] == target].copy()
    if part.empty:
        raise RuntimeError(f"No rows for target {target}")
    cols = feature_columns(part)
    groups = validation_groups(part, group_by)
    n_splits = min(5, groups.nunique())
    cv = GroupKFold(n_splits=n_splits)
    X = part[cols]
    y = part["y"].to_numpy(dtype="float32")

    results = []
    best_name = None
    best_selection_key = (float("inf"), float("-inf"))
    for name, model in candidate_models(random_state).items():
        print(f"training {target}/{name} on {len(part)} rows and {len(cols)} numeric features", flush=True)
        preds = np.zeros(len(part), dtype="float32")
        for train_idx, val_idx in cv.split(X, y, groups=groups):
            pipe = make_pipeline(clone(model), cols, max_features=max_features)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            preds[val_idx] = pipe.predict(X.iloc[val_idx])
        metrics = parameter_metrics(y, np.clip(preds, 0, None))
        result = {
            "target": target,
            "model": name,
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "nrmse": metrics["nrmse"],
            "score": metrics["score"],
            "rows": int(len(part)),
        }
        results.append(result)
        selection_key = (-result["score"], result["rmse"])
        if selection_key < best_selection_key:
            best_selection_key = selection_key
            best_name = name

    final_pipe = make_pipeline(candidate_models(random_state)[str(best_name)], cols, max_features=max_features)
    final_pipe.fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_result = next(row for row in results if row["model"] == best_name)
    joblib.dump(
        {
            "pipeline": final_pipe,
            "columns": cols,
            "target": target,
            "summary": {
                "model": best_name,
                "rmse": best_result["rmse"],
                "r2": best_result["r2"],
                "nrmse": best_result["nrmse"],
                "score": best_result["score"],
                "group_by": group_by,
            },
        },
        model_dir / f"{target}.joblib",
    )
    return {"best": best_name, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=250)
    parser.add_argument("--group-by", choices=["image", "area"], default="image")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    summary = {
        "turbidity": train_one(df, "turbidity", args.model_dir, args.random_state, args.max_features, args.group_by),
        "chla": train_one(df, "chla", args.model_dir, args.random_state, args.max_features, args.group_by),
    }
    for target, info in summary.items():
        print(f"{target}: best={info['best']}")
        for row in info["results"]:
            print(
                f"  {row['model']}: score={row['score']:.4f} rmse={row['rmse']:.4f} "
                f"r2={row['r2']:.4f} nrmse={row['nrmse']:.4f} rows={row['rows']}"
            )


if __name__ == "__main__":
    main()
