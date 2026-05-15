from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .scoring import parameter_metrics
from .train_baseline import candidate_models, feature_columns, make_pipeline, validation_groups
from .train_ensemble import _clip_fit_target, _member_weights, _selection_sort_key, filter_to_test_range


REGIME_THRESHOLDS = [1.6, 5.0]
REGIME_NAMES = ["low", "medium", "high"]


def assign_regimes(y: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(y), dtype="int64")
    labels[y > REGIME_THRESHOLDS[0]] = 1
    labels[y > REGIME_THRESHOLDS[1]] = 2
    return labels


def make_classifier_pipeline(columns: list[str], max_features: int | None, random_state: int) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    pre = ColumnTransformer([("num", numeric, columns)], remainder="drop")
    steps: list[tuple[str, object]] = [("pre", pre)]
    if max_features is not None and max_features > 0 and max_features < len(columns):
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=max_features)))
    steps.append(
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps)


def score_regime_members(
    *,
    part: pd.DataFrame,
    model_names: list[str],
    max_features: int | None,
    random_state: int,
    group_by: str,
) -> list[dict[str, float]]:
    cols = feature_columns(part)
    groups = validation_groups(part, group_by)
    n_splits = min(3, groups.nunique())
    if n_splits < 2:
        return [{"name": name, "score": 0.0, "rmse": float("inf"), "r2": float("-inf")} for name in model_names]
    cv = GroupKFold(n_splits=n_splits)
    X = part[cols]
    y = part["y"].to_numpy(dtype="float32")
    models = candidate_models(random_state)
    rows = []
    for name in model_names:
        preds = np.zeros(len(part), dtype="float32")
        for train_idx, val_idx in cv.split(X, y, groups=groups):
            pipe = make_pipeline(clone(models[name]), cols, max_features=max_features)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            preds[val_idx] = pipe.predict(X.iloc[val_idx])
        metrics = parameter_metrics(y, np.clip(preds, 0, None))
        rows.append({"name": name, "score": metrics["score"], "rmse": metrics["rmse"], "r2": metrics["r2"]})
    rows.sort(key=lambda row: _selection_sort_key(row, "score"))
    return rows


def train_regime_ensemble(
    *,
    features_path: Path,
    model_dir: Path,
    random_state: int,
    max_features: int | None,
    group_by: str,
    filter_test_range: bool,
    filter_range_padding: float,
) -> dict[str, object]:
    df = pd.read_csv(features_path)
    part = df[df["target"] == "turbidity"].copy()
    if part.empty:
        raise RuntimeError(f"No turbidity rows in {features_path}")
    part = filter_to_test_range(part, "turbidity", filter_test_range, filter_range_padding)
    cols = feature_columns(part)
    X = part[cols]
    y = part["y"].to_numpy(dtype="float32")
    regime_labels = assign_regimes(y)
    groups = validation_groups(part, group_by)

    models = ["lightgbm", "lightgbm_log", "extra", "extra_log", "catboost", "xgboost_log", "hgb_log"]
    regime_templates: list[dict[str, object]] = []
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        regime_part = part[regime_labels == regime_id].copy()
        if regime_part.empty:
            continue
        scored = score_regime_members(
            part=regime_part,
            model_names=models,
            max_features=max_features,
            random_state=random_state,
            group_by=group_by,
        )
        selected = scored[:3]
        weights = _member_weights(selected, "score")
        regime_templates.append(
            {
                "id": regime_id,
                "name": regime_name,
                "selected": [
                    {
                        "name": row["name"],
                        "score": row["score"],
                        "rmse": row["rmse"],
                        "r2": row["r2"],
                        "weight": weights[idx],
                    }
                    for idx, row in enumerate(selected)
                ],
            }
        )

    classifier = make_classifier_pipeline(cols, max_features=max_features, random_state=random_state)
    n_splits = min(5, groups.nunique())
    cv = GroupKFold(n_splits=n_splits)
    candidate_model_map = candidate_models(random_state)

    oof = np.zeros(len(part), dtype="float32")
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        regime_train = regime_labels[train_idx]
        X_val = X.iloc[val_idx]

        clf = clone(classifier)
        clf.fit(X_train, regime_train)
        proba_raw = clf.predict_proba(X_val)
        proba = np.zeros((len(val_idx), 3), dtype="float64")
        for class_pos, class_id in enumerate(clf.named_steps["model"].classes_):
            proba[:, int(class_id)] = proba_raw[:, class_pos]

        regime_predictions: list[np.ndarray] = []
        for template in regime_templates:
            regime_id = int(template["id"])
            member_preds = []
            member_weights = []
            regime_mask = regime_train == regime_id
            if np.count_nonzero(regime_mask) < 12:
                regime_mask = np.ones_like(regime_train, dtype=bool)
            for member in template["selected"]:
                pipe = make_pipeline(clone(candidate_model_map[member["name"]]), cols, max_features=max_features)
                pipe.fit(X_train.loc[regime_mask], _clip_fit_target(y_train[regime_mask], None))
                member_preds.append(pipe.predict(X_val))
                member_weights.append(float(member["weight"]))
            regime_predictions.append(
                np.clip(
                    np.average(np.vstack(member_preds), axis=0, weights=np.asarray(member_weights, dtype="float64")),
                    0,
                    None,
                )
            )
        oof[val_idx] = np.sum(np.vstack(regime_predictions).T * proba[:, : len(regime_predictions)], axis=1)

    metrics = parameter_metrics(y, np.clip(oof, 0, None))

    final_classifier = make_classifier_pipeline(cols, max_features=max_features, random_state=random_state)
    final_classifier.fit(X, regime_labels)

    regimes = []
    for template in regime_templates:
        regime_id = int(template["id"])
        regime_mask = regime_labels == regime_id
        members = []
        for member in template["selected"]:
            pipe = make_pipeline(clone(candidate_model_map[member["name"]]), cols, max_features=max_features)
            pipe.fit(X.loc[regime_mask], y[regime_mask])
            members.append(
                {
                    "name": member["name"],
                    "pipeline": pipe,
                    "columns": cols,
                    "weight": float(member["weight"]),
                }
            )
        regimes.append(
            {
                "name": template["name"],
                "range": (
                    [None, REGIME_THRESHOLDS[0]]
                    if regime_id == 0
                    else [REGIME_THRESHOLDS[0], REGIME_THRESHOLDS[1]]
                    if regime_id == 1
                    else [REGIME_THRESHOLDS[1], None]
                ),
                "members": members,
            }
        )

    summary = {
        "kind": "regime_ensemble",
        "target": "turbidity",
        "features_path": str(features_path),
        "rows": int(len(part)),
        "columns": int(len(cols)),
        "ensemble_rmse": metrics["rmse"],
        "ensemble_r2": metrics["r2"],
        "ensemble_nrmse": metrics["nrmse"],
        "ensemble_score": metrics["score"],
        "group_by": group_by,
        "filter_test_range": bool(filter_test_range),
        "filter_range_padding": float(filter_range_padding),
        "thresholds": list(REGIME_THRESHOLDS),
        "regimes": [
            {
                "name": template["name"],
                "selected": template["selected"],
            }
            for template in regime_templates
        ],
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kind": "regime_ensemble",
            "target": "turbidity",
            "columns": cols,
            "classifier": final_classifier,
            "regimes": regimes,
            "summary": summary,
        },
        model_dir / "turbidity_ensemble.joblib",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/patch_32/train_features_v2.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--group-by", choices=["image", "area"], default="area")
    parser.add_argument("--filter-test-range", action="store_true")
    parser.add_argument("--filter-range-padding", type=float, default=0.10)
    args = parser.parse_args()

    summary = train_regime_ensemble(
        features_path=args.features,
        model_dir=args.model_dir,
        random_state=args.random_state,
        max_features=args.max_features,
        group_by=args.group_by,
        filter_test_range=args.filter_test_range,
        filter_range_padding=args.filter_range_padding,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
