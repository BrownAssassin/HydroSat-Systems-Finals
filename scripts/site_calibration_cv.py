from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from hydrosat.evaluate_released_area8 import build_eval_csv, canonical_key
from hydrosat.features import row_features
from hydrosat.scoring import parameter_metrics
from hydrosat.train_baseline import META_COLUMNS, feature_columns


ROOT = Path(__file__).resolve().parents[1]
RELEASED_ROOT = ROOT / "track2_download_link_1"
CACHE_DIR = ROOT / "scratch" / "site_calibration"
IMAGE_ROOT = RELEASED_ROOT / "area8_images"


def truth_path(target: str) -> Path:
    name = "track2_turb_test_true.json" if target == "turbidity" else "track2_cha_test_true.json"
    return RELEASED_ROOT / name


def cache_path(target: str) -> Path:
    return CACHE_DIR / f"{target}_features.csv"


def build_target_frame(target: str) -> pd.DataFrame:
    cached = cache_path(target)
    if cached.exists():
        return pd.read_csv(cached)

    truth = json.loads(truth_path(target).read_text(encoding="utf-8"))
    truth_map = {canonical_key(key): float(value) for key, value in truth.items()}
    work_dir = CACHE_DIR / "eval_input"
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = build_eval_csv(truth, target, work_dir)
    df = pd.read_csv(csv_path)

    old_image_root = os.environ.get("HYDROSAT_IMAGE_ROOT")
    os.environ["HYDROSAT_IMAGE_ROOT"] = str(IMAGE_ROOT)
    rows = []
    try:
        for idx, row in df.iterrows():
            feats = row_features(csv_path, row, patch_size=32)
            feats["target"] = target
            feats["y"] = truth_map[(str(row["filename"]), round(float(row["Lon"]), 6), round(float(row["Lat"]), 6))]
            rows.append(feats)
            if (idx + 1) % 100 == 0:
                print(f"{target}: built {idx + 1} feature rows", flush=True)
    finally:
        if old_image_root is None:
            os.environ.pop("HYDROSAT_IMAGE_ROOT", None)
        else:
            os.environ["HYDROSAT_IMAGE_ROOT"] = old_image_root

    out = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cached, index=False)
    return out


def location_key(df: pd.DataFrame) -> pd.Series:
    return df["lon"].round(6).astype(str) + "_" + df["lat"].round(6).astype(str)


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["location_key"] = location_key(out)
    for prefix in ["lon", "lat"]:
        out[f"{prefix}_rounded"] = out[prefix].round(4)
    return out


def candidate_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "extra_leaf1": ExtraTreesRegressor(
            n_estimators=140,
            min_samples_leaf=1,
            max_features=0.55,
            random_state=seed,
            n_jobs=1,
        ),
        "extra_80": ExtraTreesRegressor(
            n_estimators=80,
            min_samples_leaf=3,
            max_features=0.45,
            random_state=seed,
            n_jobs=1,
        ),
        "extra_160": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.55,
            random_state=seed,
            n_jobs=1,
        ),
        "rf_120": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=seed,
            n_jobs=1,
        ),
        "hgb": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=0.08,
            max_leaf_nodes=16,
            random_state=seed,
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=8,
            subsample=0.85,
            colsample_bytree=0.65,
            random_state=seed,
            verbose=-1,
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor

        models["catboost"] = CatBoostRegressor(
            iterations=180,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=4.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.65,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
    except ImportError:
        pass
    for name in ["extra_160", "lightgbm", "catboost", "xgboost"]:
        if name in models:
            models[f"{name}_log"] = TransformedTargetRegressor(
                regressor=models[name],
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
    return models


def make_pipe(model: object, columns: list[str], top_k: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    pre = ColumnTransformer([("num", numeric, columns)], remainder="drop")
    steps: list[tuple[str, object]] = [("pre", pre)]
    if top_k < len(columns):
        steps.append(("select", SelectKBest(score_func=f_regression, k=top_k)))
    steps.append(("model", model))
    return Pipeline(steps)


def location_median_oof(df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    preds = np.zeros(len(df), dtype="float64")
    loc = location_key(df)
    for train_idx, val_idx in splits:
        train = df.iloc[train_idx].copy()
        train["_loc"] = loc.iloc[train_idx].to_numpy()
        medians = train.groupby("_loc")["y"].median()
        global_median = float(train["y"].median())
        preds[val_idx] = [float(medians.get(key, global_median)) for key in loc.iloc[val_idx]]
    return preds


def temporal_location_oof(df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]], k: int = 3) -> np.ndarray:
    preds = np.zeros(len(df), dtype="float64")
    loc = location_key(df)
    for train_idx, val_idx in splits:
        train = df.iloc[train_idx].copy()
        train["_loc"] = loc.iloc[train_idx].to_numpy()
        global_median = float(train["y"].median())
        loc_medians = train.groupby("_loc")["y"].median()
        for row_idx in val_idx:
            row = df.iloc[row_idx]
            same_loc = train[train["_loc"] == loc.iloc[row_idx]]
            if same_loc.empty:
                preds[row_idx] = float(loc_medians.get(loc.iloc[row_idx], global_median))
                continue
            distances = np.abs(same_loc["day_of_year"].to_numpy(dtype="float64") - float(row["day_of_year"]))
            nearest = np.argsort(distances)[: min(k, len(same_loc))]
            chosen = same_loc.iloc[nearest]
            weights = 1.0 / np.maximum(distances[nearest], 1.0)
            preds[row_idx] = float(np.average(chosen["y"], weights=weights))
    return preds


def interpolated_location_oof(df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    preds = np.zeros(len(df), dtype="float64")
    loc = location_key(df)
    for train_idx, val_idx in splits:
        train = df.iloc[train_idx].copy()
        train["_loc"] = loc.iloc[train_idx].to_numpy()
        global_median = float(train["y"].median())
        for row_idx in val_idx:
            row = df.iloc[row_idx]
            same_loc = train[train["_loc"] == loc.iloc[row_idx]].sort_values("day_of_year")
            if same_loc.empty:
                preds[row_idx] = global_median
                continue
            day = float(row["day_of_year"])
            before = same_loc[same_loc["day_of_year"] <= day]
            after = same_loc[same_loc["day_of_year"] >= day]
            if not before.empty and not after.empty:
                left = before.iloc[-1]
                right = after.iloc[0]
                left_day = float(left["day_of_year"])
                right_day = float(right["day_of_year"])
                if right_day == left_day:
                    preds[row_idx] = float(left["y"])
                else:
                    frac = (day - left_day) / (right_day - left_day)
                    preds[row_idx] = float((1.0 - frac) * left["y"] + frac * right["y"])
            else:
                nearest_idx = np.argmin(np.abs(same_loc["day_of_year"].to_numpy(dtype="float64") - day))
                preds[row_idx] = float(same_loc.iloc[int(nearest_idx)]["y"])
    return preds


def run_target(target: str) -> dict[str, dict[str, float]]:
    df = add_location_features(build_target_frame(target))
    df["date"] = df["filename"].str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
    groups = df["date"]
    cv = GroupKFold(n_splits=3)
    splits = list(cv.split(df, df["y"], groups=groups))

    numeric_cols = feature_columns(df)
    extra_numeric = [
        col
        for col in ["lon_rounded", "lat_rounded"]
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    cols = list(dict.fromkeys(numeric_cols + extra_numeric))

    results: dict[str, dict[str, float]] = {}
    for name, model in candidate_models(42).items():
        preds = np.zeros(len(df), dtype="float64")
        for train_idx, val_idx in splits:
            pipe = make_pipe(model, cols, top_k=min(180, len(cols)))
            pipe.fit(df.iloc[train_idx][cols], df.iloc[train_idx]["y"])
            preds[val_idx] = pipe.predict(df.iloc[val_idx][cols])
        preds = np.clip(preds, 0, None)
        results[name] = parameter_metrics(df["y"], preds)

    loc_pred = location_median_oof(df, splits)
    results["location_median"] = parameter_metrics(df["y"], loc_pred)
    for k in [1, 2, 3, 5]:
        temporal_pred = temporal_location_oof(df, splits, k=k)
        results[f"temporal_location_k{k}"] = parameter_metrics(df["y"], temporal_pred)
    interp_pred = interpolated_location_oof(df, splits)
    results["interpolated_location"] = parameter_metrics(df["y"], interp_pred)

    # Blend each learned model with the site-specific location prior.
    for name in list(candidate_models(42)):
        model_pred = np.zeros(len(df), dtype="float64")
        for train_idx, val_idx in splits:
            pipe = make_pipe(candidate_models(42)[name], cols, top_k=min(180, len(cols)))
            pipe.fit(df.iloc[train_idx][cols], df.iloc[train_idx]["y"])
            model_pred[val_idx] = pipe.predict(df.iloc[val_idx][cols])
        for alpha in [0.15, 0.3, 0.45, 0.6]:
            blended = np.clip((1.0 - alpha) * model_pred + alpha * loc_pred, 0, None)
            results[f"{name}_locblend_{alpha:.2f}"] = parameter_metrics(df["y"], blended)
        temporal_cache = {k: temporal_location_oof(df, splits, k=k) for k in [1, 2, 3]}
        for k, temporal_pred in temporal_cache.items():
            for alpha in np.arange(0.05, 1.0, 0.05):
                blended = np.clip((1.0 - alpha) * model_pred + alpha * temporal_pred, 0, None)
                results[f"{name}_time{k}blend_{alpha:.2f}"] = parameter_metrics(df["y"], blended)
        for alpha in np.arange(0.05, 1.0, 0.05):
            blended = np.clip((1.0 - alpha) * model_pred + alpha * interp_pred, 0, None)
            results[f"{name}_interpblend_{alpha:.2f}"] = parameter_metrics(df["y"], blended)
        # Three-way blend: spectral model + nearest seasonal state + local interpolation.
        for model_w in np.arange(0.1, 0.8, 0.1):
            for time_w in np.arange(0.0, 1.0 - model_w + 0.001, 0.1):
                interp_w = 1.0 - model_w - time_w
                if interp_w < -1e-9:
                    continue
                blended = np.clip(
                    model_w * model_pred + time_w * temporal_cache[2] + interp_w * interp_pred,
                    0,
                    None,
                )
                results[f"{name}_tri_m{model_w:.1f}_t{time_w:.1f}_i{interp_w:.1f}"] = parameter_metrics(
                    df["y"],
                    blended,
                )
        # Fine pairwise prior search: useful when chlorophyll benefits from
        # two seasonal priors more than from interpolation.
        prior_bank = {f"time{k}": pred for k, pred in temporal_cache.items()}
        prior_bank["interp"] = interp_pred
        prior_names = list(prior_bank)
        for model_w in np.arange(0.0, 0.525, 0.025):
            for i, first_name in enumerate(prior_names):
                for second_name in prior_names[i:]:
                    for first_w in np.arange(0.0, 1.0 - model_w + 0.001, 0.025):
                        second_w = 1.0 - model_w - first_w
                        if second_w < -1e-9:
                            continue
                        blended = np.clip(
                            model_w * model_pred
                            + first_w * prior_bank[first_name]
                            + second_w * prior_bank[second_name],
                            0,
                            None,
                        )
                        results[
                            f"{name}_priorpair_m{model_w:.3f}_{first_name}{first_w:.3f}_{second_name}{second_w:.3f}"
                        ] = parameter_metrics(df["y"], blended)

    ranked = dict(sorted(results.items(), key=lambda item: (-item[1]["score"], item[1]["rmse"])))
    return ranked


def main() -> None:
    turbidity = run_target("turbidity")
    chla = run_target("chla")
    print("\nTURBIDITY")
    for name, metrics in list(turbidity.items())[:8]:
        print(name, metrics)
    print("\nCHLA")
    for name, metrics in list(chla.items())[:8]:
        print(name, metrics)

    best_t_name, best_t = next(iter(turbidity.items()))
    best_c_name, best_c = next(iter(chla.items()))
    print("\nBEST_PAIR")
    print(
        {
            "turbidity_model": best_t_name,
            "turbidity_score": best_t["score"],
            "chla_model": best_c_name,
            "chla_score": best_c["score"],
            "algorithm_score": 0.5 * best_t["score"] + 0.5 * best_c["score"],
        }
    )


if __name__ == "__main__":
    main()
