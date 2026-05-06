"""Training entrypoint for the final-round tabular baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*LGBMRegressor.*",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .config import (
    ALL_TARGETS,
    CHLA_TARGET,
    DEFAULT_DIAGNOSTICS_DIR,
    DEFAULT_FEATURES_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_PATCH_SIZE,
    PUBLIC_TEST_LABEL_STATS,
    TURBIDITY_TARGET,
)
from .features import build_training_table

META_COLUMNS = {
    "target",
    "filename",
    "image_path",
    "area",
    "source_csv",
    "lon",
    "lat",
    "y",
    "patch_size",
    "pixel_row",
    "pixel_col",
    "pixel_row_norm",
    "pixel_col_norm",
}

EXCLUDED_DEFAULT_FEATURES = {
    "pixel_row",
    "pixel_col",
    "pixel_row_norm",
    "pixel_col_norm",
    "lon",
    "lat",
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the numeric training feature columns used by baseline models."""

    columns: list[str] = []
    for column in frame.columns:
        if column in META_COLUMNS or column in EXCLUDED_DEFAULT_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def validation_groups(frame: pd.DataFrame, group_by: str) -> pd.Series:
    """Select the grouping column for grouped cross-validation."""

    if group_by == "area":
        return frame["area"].astype(str)
    if group_by == "filename":
        return frame["filename"].astype(str)
    raise ValueError(f"Unsupported group mode '{group_by}'. Expected 'area' or 'filename'.")


def candidate_models(random_state: int) -> dict[str, object]:
    """Return the baseline candidate estimators."""

    raw_models: dict[str, object] = {
        "hgb": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            l2_regularization=0.05,
            max_depth=None,
            random_state=random_state,
        ),
        "extra": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        raw_models["lightgbm"] = LGBMRegressor(
            n_estimators=400,
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
            n_estimators=400,
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

    return {
        name: TransformedTargetRegressor(
            regressor=clone(model),
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
        for name, model in raw_models.items()
    }


def make_pipeline(model: object, columns: list[str], max_features: int | None) -> Pipeline:
    """Build the preprocessing and model pipeline for tabular training."""

    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocess = ColumnTransformer(
        [("num", numeric, columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps: list[tuple[str, object]] = [("preprocess", preprocess)]
    if max_features is not None and 0 < max_features < len(columns):
        steps.append(("select", SelectKBest(score_func=f_regression, k=max_features)))
    steps.append(("model", model))
    return Pipeline(steps)


def variant_weights(y: np.ndarray, target: str, variant: str) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    """Return optional sample weights for one training variant."""

    if variant == "full":
        return None, {"variant": "full"}

    if variant != "test_range_weighted":
        raise ValueError(f"Unsupported training variant '{variant}'.")

    stats = PUBLIC_TEST_LABEL_STATS[target]
    span = stats["max"] - stats["min"]
    low = max(0.0, stats["min"] - 0.25 * span)
    high = stats["max"] + 0.50 * span
    weights = np.full(y.shape, 0.75, dtype="float32")
    in_range = (y >= low) & (y <= high)
    weights[in_range] = 3.0

    metadata = {
        "variant": variant,
        "range_low": float(low),
        "range_high": float(high),
        "rows_in_range": int(in_range.sum()),
        "rows_total": int(y.shape[0]),
        "weight_in_range": 3.0,
        "weight_out_of_range": 0.75,
    }
    return weights, metadata


def evaluate_candidate(
    frame: pd.DataFrame,
    target: str,
    model_name: str,
    model: object,
    variant: str,
    columns: list[str],
    group_by: str,
    max_features: int | None,
) -> dict[str, object]:
    """Run grouped cross-validation for one model/variant configuration."""

    X = frame[columns]
    y = frame["y"].to_numpy(dtype="float32")
    groups = validation_groups(frame, group_by)
    unique_groups = groups.nunique()
    n_splits = min(5, int(unique_groups))
    if n_splits < 2:
        raise RuntimeError(f"Need at least 2 unique groups for {group_by}-grouped CV on {target}.")

    sample_weights, variant_meta = variant_weights(y, target, variant)
    cv = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(frame), dtype="float32")
    folds: list[dict[str, float | int]] = []

    for fold_index, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        pipeline = make_pipeline(clone(model), columns, max_features=max_features)
        fit_params = {}
        if sample_weights is not None:
            fit_params["model__sample_weight"] = sample_weights[train_idx]
        pipeline.fit(X.iloc[train_idx], y[train_idx], **fit_params)
        fold_pred = np.clip(pipeline.predict(X.iloc[val_idx]), 0, None)
        oof_pred[val_idx] = fold_pred
        folds.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "rmse": rmse(y[val_idx], fold_pred),
                "r2": float(r2_score(y[val_idx], fold_pred)),
            }
        )

    return {
        "target": target,
        "model_name": model_name,
        "variant": variant,
        "group_by": group_by,
        "rmse": rmse(y, oof_pred),
        "r2": float(r2_score(y, oof_pred)),
        "folds": folds,
        "variant_metadata": variant_meta,
        "oof_pred": oof_pred.tolist(),
    }


def select_best_configuration(
    feature_frame: pd.DataFrame,
    target: str,
    random_state: int,
    max_features: int | None,
) -> tuple[dict[str, object], dict[str, object], list[str], list[dict[str, object]]]:
    """Evaluate all primary configurations and return the best one."""

    part = feature_frame[(feature_frame["target"] == target) & feature_frame["y"].notna()].copy()
    if part.empty:
        raise RuntimeError(f"No feature rows found for target '{target}'.")

    columns = feature_columns(part)
    models = candidate_models(random_state=random_state)
    variants = ("full", "test_range_weighted")
    results: list[dict[str, object]] = []
    best_config: dict[str, object] | None = None
    best_model: dict[str, object] | None = None

    for variant in variants:
        for model_name, model in models.items():
            print(f"{target}: primary cv for model={model_name} variant={variant}", flush=True)
            result = evaluate_candidate(
                frame=part,
                target=target,
                model_name=model_name,
                model=model,
                variant=variant,
                columns=columns,
                group_by="area",
                max_features=max_features,
            )
            result["rows"] = int(len(part))
            result["columns"] = int(len(columns))
            results.append(result)
            if best_config is None or float(result["rmse"]) < float(best_config["rmse"]):
                best_config = result
                best_model = {"name": model_name, "estimator": model}

    assert best_config is not None
    assert best_model is not None
    return best_config, best_model, columns, results


def secondary_validation(
    feature_frame: pd.DataFrame,
    target: str,
    best_model_name: str,
    best_model: object,
    best_variant: str,
    columns: list[str],
    max_features: int | None,
) -> dict[str, object]:
    """Run the selected configuration with filename-grouped CV for diagnostics."""

    part = feature_frame[(feature_frame["target"] == target) & feature_frame["y"].notna()].copy()
    print(f"{target}: secondary filename-grouped cv for model={best_model_name} variant={best_variant}", flush=True)
    return evaluate_candidate(
        frame=part,
        target=target,
        model_name=best_model_name,
        model=best_model,
        variant=best_variant,
        columns=columns,
        group_by="filename",
        max_features=max_features,
    )


def fit_selected_model(
    feature_frame: pd.DataFrame,
    target: str,
    selected_model: object,
    selected_variant: str,
    columns: list[str],
    patch_size: int,
    max_features: int | None,
    primary_summary: dict[str, object],
    secondary_summary: dict[str, object],
    output_dir: Path,
) -> Path:
    """Fit the selected configuration on all target rows and save the deployable bundle."""

    part = feature_frame[(feature_frame["target"] == target) & feature_frame["y"].notna()].copy()
    X = part[columns]
    y = part["y"].to_numpy(dtype="float32")
    sample_weights, variant_meta = variant_weights(y, target, selected_variant)

    pipeline = make_pipeline(clone(selected_model), columns, max_features=max_features)
    fit_params = {}
    if sample_weights is not None:
        fit_params["model__sample_weight"] = sample_weights
    pipeline.fit(X, y, **fit_params)

    bundle = {
        "kind": "single_model",
        "target": target,
        "patch_size": int(patch_size),
        "pipeline": pipeline,
        "feature_columns": columns,
        "selected_model": str(primary_summary["model_name"]),
        "selected_variant": str(selected_variant),
        "variant_metadata": variant_meta,
        "cv": {
            "primary_area_grouped": {
                "rmse": float(primary_summary["rmse"]),
                "r2": float(primary_summary["r2"]),
                "folds": primary_summary["folds"],
            },
            "secondary_filename_grouped": {
                "rmse": float(secondary_summary["rmse"]),
                "r2": float(secondary_summary["r2"]),
                "folds": secondary_summary["folds"],
            },
        },
        "training_rows": int(len(part)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{target}.joblib"
    joblib.dump(bundle, model_path)
    return model_path


def build_or_load_features(
    features_path: Path,
    data_root: Path | None,
    patch_size: int,
    progress_every: int,
    rebuild: bool,
) -> pd.DataFrame:
    """Load an existing feature table or build one from the normalized data root."""

    if rebuild or not features_path.exists():
        print(f"building training features at {features_path}", flush=True)
        feature_frame = build_training_table(
            data_root=data_root,
            patch_size=patch_size,
            progress_every=progress_every,
        )
        features_path.parent.mkdir(parents=True, exist_ok=True)
        feature_frame.to_csv(features_path, index=False)
        return feature_frame

    print(f"loading features from {features_path}", flush=True)
    return pd.read_csv(features_path)


def save_diagnostics(
    target: str,
    diagnostics_dir: Path,
    primary_results: list[dict[str, object]],
    selected_primary: dict[str, object],
    selected_secondary: dict[str, object],
) -> Path:
    """Save model-selection diagnostics for one target."""

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / f"{target}_summary.json"
    payload = {
        "target": target,
        "primary_results": [
            {
                key: value
                for key, value in result.items()
                if key != "oof_pred"
            }
            for result in primary_results
        ],
        "selected_primary": {
            key: value
            for key, value in selected_primary.items()
            if key != "oof_pred"
        },
        "selected_secondary": {
            key: value
            for key, value in selected_secondary.items()
            if key != "oof_pred"
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def train_target(
    feature_frame: pd.DataFrame,
    target: str,
    models_dir: Path,
    diagnostics_dir: Path,
    patch_size: int,
    random_state: int,
    max_features: int | None,
) -> dict[str, object]:
    """Train and save the best baseline model for one target."""

    best_primary, best_model, columns, primary_results = select_best_configuration(
        feature_frame=feature_frame,
        target=target,
        random_state=random_state,
        max_features=max_features,
    )
    secondary = secondary_validation(
        feature_frame=feature_frame,
        target=target,
        best_model_name=str(best_primary["model_name"]),
        best_model=best_model["estimator"],
        best_variant=str(best_primary["variant"]),
        columns=columns,
        max_features=max_features,
    )
    model_path = fit_selected_model(
        feature_frame=feature_frame,
        target=target,
        selected_model=best_model["estimator"],
        selected_variant=str(best_primary["variant"]),
        columns=columns,
        patch_size=patch_size,
        max_features=max_features,
        primary_summary=best_primary,
        secondary_summary=secondary,
        output_dir=models_dir,
    )
    diagnostics_path = save_diagnostics(
        target=target,
        diagnostics_dir=diagnostics_dir,
        primary_results=primary_results,
        selected_primary=best_primary,
        selected_secondary=secondary,
    )
    return {
        "target": target,
        "model_path": model_path,
        "diagnostics_path": diagnostics_path,
        "selected_model": str(best_primary["model_name"]),
        "selected_variant": str(best_primary["variant"]),
        "primary_rmse": float(best_primary["rmse"]),
        "secondary_rmse": float(secondary["rmse"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the HydroSat final-round tabular baseline.")
    parser.add_argument("--target", default="all", choices=("all", TURBIDITY_TARGET, CHLA_TARGET), help="Target to train.")
    parser.add_argument("--data-root", type=Path, default=None, help="Normalized raw-data root. Needed when building features.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH, help="Unified training feature table.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory where deployable model bundles are saved.")
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAGNOSTICS_DIR, help="Directory for training diagnostics.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size metadata saved with the trained model.")
    parser.add_argument("--progress-every", type=int, default=100, help="Progress cadence when building features.")
    parser.add_argument("--rebuild-features", action="store_true", help="Rebuild the training feature table before training.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used by candidate models.")
    parser.add_argument("--max-features", type=int, default=180, help="Optional SelectKBest feature count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    feature_frame = build_or_load_features(
        features_path=args.features,
        data_root=args.data_root,
        patch_size=args.patch_size,
        progress_every=args.progress_every,
        rebuild=args.rebuild_features,
    )

    targets = ALL_TARGETS if args.target == "all" else (args.target,)
    summaries = []
    for target in targets:
        summaries.append(
            train_target(
                feature_frame=feature_frame,
                target=target,
                models_dir=args.models_dir,
                diagnostics_dir=args.diagnostics_dir,
                patch_size=args.patch_size,
                random_state=args.random_state,
                max_features=args.max_features,
            )
        )

    for summary in summaries:
        print(
            f"{summary['target']}: saved {summary['model_path']} "
            f"(model={summary['selected_model']} variant={summary['selected_variant']} "
            f"area_rmse={summary['primary_rmse']:.4f} filename_rmse={summary['secondary_rmse']:.4f})",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
