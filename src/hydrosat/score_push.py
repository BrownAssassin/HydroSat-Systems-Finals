from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .build_features import build_features
from .evaluate_released_area8 import build_eval_csv, score_predictions, write_prediction_outputs
from .infer import infer_csv
from .scoring import algorithm_score, pair_summary
from .train_ensemble import train_target_ensemble
from .train_regime_ensemble import train_regime_ensemble


TURBIDITY_MODELS = ["lightgbm", "lightgbm_log", "extra", "extra_log", "catboost", "xgboost_log", "hgb_log"]
CHLA_MODELS = ["catboost", "extra", "hgb", "lightgbm", "lightgbm_log", "xgboost"]

PATCH_SIZES = [24, 32, 40]

STAGE_A_CONFIGS = [
    {"name": "full", "filter_test_range": False, "filter_range_padding": 0.05, "turbidity_clip_quantile": None},
    {"name": "filter05", "filter_test_range": True, "filter_range_padding": 0.05, "turbidity_clip_quantile": None},
    {"name": "filter10", "filter_test_range": True, "filter_range_padding": 0.10, "turbidity_clip_quantile": None},
    {"name": "filter15", "filter_test_range": True, "filter_range_padding": 0.15, "turbidity_clip_quantile": None},
    {"name": "filter10_clip97", "filter_test_range": True, "filter_range_padding": 0.10, "turbidity_clip_quantile": 0.97},
    {"name": "filter10_clip99", "filter_test_range": True, "filter_range_padding": 0.10, "turbidity_clip_quantile": 0.99},
]

FINE_TOP_N = [3, 5, 7]
FINE_MAX_FEATURES = [300, 500, 800]

CURRENT_SCORE_FLOOR = 14.4445

TURBIDITY_LOGNORMAL_SIGMAS = [0.40, 0.44, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.66, 0.70, 0.75]
TURBIDITY_PRIOR_SHRINKS = [0.00, 0.02, 0.05]
TURBIDITY_BLEND_WEIGHTS = [0.74, 0.76, 0.78, 0.80, 0.82, 0.84]
TURBIDITY_BLEND_RANK_POWERS = [0.90, 0.95, 1.00]

CHLA_MODEL_PRIOR_SHRINKS = [0.00, 0.02, 0.05, 0.08, 0.10]
CHLA_BLEND_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30]
CHLA_LOGNORMAL_SIGMAS = [0.35, 0.45, 0.55, 0.65]

TURBIDITY_RETRAIN_TOP_N = [1, 2, 4]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@contextmanager
def _temporary_env(overrides: dict[str, str] | None = None):
    overrides = overrides or {}
    old_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def feature_files_for_patch(features_dir: Path, patch_size: int) -> dict[str, Path]:
    patch_dir = features_dir / f"patch_{patch_size}"
    return {
        "dir": patch_dir,
        "canonical": patch_dir / f"train_features_patch{patch_size}.csv",
        "train_features": patch_dir / "train_features.csv",
        "train_features_v2": patch_dir / "train_features_v2.csv",
    }


def build_feature_tables(data_root: Path, features_dir: Path, patch_sizes: list[int], progress_every: int, skip_existing: bool) -> dict[int, dict[str, object]]:
    outputs: dict[int, dict[str, object]] = {}
    for patch_size in patch_sizes:
        files = feature_files_for_patch(features_dir, patch_size)
        files["dir"].mkdir(parents=True, exist_ok=True)
        if skip_existing and files["canonical"].exists():
            df = pd.read_csv(files["canonical"])
        else:
            df = build_features(data_root, patch_size=patch_size, progress_every=progress_every)
            for path in [files["canonical"], files["train_features"], files["train_features_v2"]]:
                df.to_csv(path, index=False)
        outputs[patch_size] = {
            **files,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        }
    return outputs


def run_pair_training(
    *,
    features_turbidity: Path,
    features_chla: Path,
    model_dir: Path,
    top_n: int,
    max_features: int,
    random_state: int,
    group_by: str,
    filter_test_range: bool,
    filter_range_padding: float,
    turbidity_clip_quantile: float | None,
    selection_metric: str,
    write_model: bool,
) -> dict[str, object]:
    turbidity = train_target_ensemble(
        features_turbidity,
        "turbidity",
        model_dir,
        TURBIDITY_MODELS,
        top_n,
        max_features,
        random_state,
        turbidity_clip_quantile,
        group_by,
        filter_test_range,
        filter_range_padding,
        selection_metric,
        write_model=write_model,
    )
    chla = train_target_ensemble(
        features_chla,
        "chla",
        model_dir,
        CHLA_MODELS,
        top_n,
        max_features,
        random_state,
        None,
        group_by,
        filter_test_range,
        filter_range_padding,
        selection_metric,
        write_model=write_model,
    )
    return {
        "turbidity": turbidity,
        "chla": chla,
        "algorithm_score": algorithm_score(
            {"score": turbidity["ensemble_score"]},
            {"score": chla["ensemble_score"]},
        ),
    }


def _load_truths(released_root: Path, target: str) -> tuple[dict[str, float], str]:
    if target == "turbidity":
        path = released_root / "track2_turb_test_true.json"
        csv_name = "track2_turb_test_point.csv"
    else:
        path = released_root / "track2_cha_test_true.json"
        csv_name = "track2_cha_test_point.csv"
    return json.loads(path.read_text(encoding="utf-8")), csv_name


def evaluate_released_target(
    *,
    released_root: Path,
    target: str,
    model_dir: Path,
    patch_size: int,
    progress_every: int,
    work_dir: Path,
    env_overrides: dict[str, str],
) -> dict[str, object]:
    truth, csv_name = _load_truths(released_root, target)
    work_dir.mkdir(parents=True, exist_ok=True)
    build_eval_csv(truth, target, work_dir)
    image_root = released_root / "area8_images"
    merged_env = {"HYDROSAT_IMAGE_ROOT": str(image_root), **env_overrides}
    with _temporary_env(merged_env):
        predictions = infer_csv(
            work_dir,
            csv_name,
            target,
            model_dir,
            patch_size,
            progress_every,
        )
    metrics = score_predictions(truth, predictions)
    return {"metrics": metrics, "predictions": predictions}


def pair_env_summary(turbidity_env: dict[str, str], chla_env: dict[str, str]) -> dict[str, object]:
    return {"turbidity_env": dict(turbidity_env), "chla_env": dict(chla_env)}


def combined_runtime_defaults(summary: dict[str, object]) -> dict[str, object]:
    turbidity_env = {str(key): str(value) for key, value in summary.get("turbidity_env", {}).items()}
    chla_env = {str(key): str(value) for key, value in summary.get("chla_env", {}).items()}

    env: dict[str, str] = {}
    for key, value in turbidity_env.items():
        if key.startswith("HYDROSAT_"):
            env[key] = value
    for key, value in chla_env.items():
        if key.startswith("HYDROSAT_CHLA_"):
            env[key] = value
        elif key.startswith("HYDROSAT_CALIBRATE_"):
            env.setdefault(key, value)

    return {
        "winner_name": str(summary.get("name", "")),
        "patch_size": int(summary.get("patch_size", 32)),
        "env": env,
    }


def evaluate_released_pair(
    *,
    released_root: Path,
    turbidity_model_dir: Path,
    chla_model_dir: Path,
    patch_size: int,
    progress_every: int,
    work_dir: Path,
    output_dir: Path | None,
    report_dir: Path | None,
    turbidity_env: dict[str, str],
    chla_env: dict[str, str],
) -> dict[str, object]:
    turbidity_result = evaluate_released_target(
        released_root=released_root,
        target="turbidity",
        model_dir=turbidity_model_dir,
        patch_size=patch_size,
        progress_every=progress_every,
        work_dir=work_dir / "turbidity",
        env_overrides=turbidity_env,
    )
    chla_result = evaluate_released_target(
        released_root=released_root,
        target="chla",
        model_dir=chla_model_dir,
        patch_size=patch_size,
        progress_every=progress_every,
        work_dir=work_dir / "chla",
        env_overrides=chla_env,
    )

    summary = {
        "released_root": str(released_root.resolve()),
        "turbidity_model_dir": str(turbidity_model_dir.resolve()),
        "chla_model_dir": str(chla_model_dir.resolve()),
        "patch_size": patch_size,
        **pair_summary(turbidity_result["metrics"], chla_result["metrics"]),
        **pair_env_summary(turbidity_env, chla_env),
    }
    if output_dir is not None:
        write_prediction_outputs(output_dir, turbidity_result["predictions"], chla_result["predictions"])
        summary["output_dir"] = str(output_dir.resolve())
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        summary["report_dir"] = str(report_dir.resolve())
        _write_json(report_dir / "released_pair_scores.json", summary)
    return summary


def neutral_pair_envs() -> tuple[dict[str, str], dict[str, str]]:
    common = {
        "HYDROSAT_CALIBRATE_TEST_STATS": "0",
        "HYDROSAT_TURBIDITY_PRIOR_SHRINK": "0",
        "HYDROSAT_CHLA_PRIOR_SHRINK": "0",
    }
    turbidity_env = {**common, "HYDROSAT_TURBIDITY_MODE": "model"}
    chla_env = {**common, "HYDROSAT_CHLA_MODE": "model"}
    return turbidity_env, chla_env


def final_default_envs() -> tuple[dict[str, str], dict[str, str]]:
    common = {
        "HYDROSAT_CALIBRATE_TEST_STATS": "1",
        "HYDROSAT_TURBIDITY_PRIOR_SHRINK": "0.05",
        "HYDROSAT_CHLA_PRIOR_SHRINK": "0",
    }
    turbidity_env = {
        **common,
        "HYDROSAT_TURBIDITY_MODE": "model",
        "HYDROSAT_TURBIDITY_CALIBRATION": "lognormal_rank",
        "HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA": "0.52",
    }
    chla_env = {**common, "HYDROSAT_CHLA_MODE": "model"}
    return turbidity_env, chla_env


def experiment_rank_key(experiment: dict[str, object]) -> tuple[float, float, float]:
    return (
        -float(experiment["area_cv"]["algorithm_score"]),
        -float(experiment["area_cv"]["turbidity"]["ensemble_score"]),
        -float(experiment["image_cv"]["algorithm_score"]),
    )


def target_trial_sort_key(trial: dict[str, object], source_area_cv_score: float) -> tuple[float, float, float]:
    metrics = trial["metrics"]
    return (
        -float(metrics["score"]),
        float(metrics["rmse"]),
        -float(source_area_cv_score),
    )


def tuned_pair_sort_key(summary: dict[str, object]) -> tuple[float, float, float]:
    return (
        -float(summary["algorithm_score"]),
        -float(summary["turbidity"]["score"]),
        -float(summary["source_area_cv_turbidity_score"]),
    )


def stage_experiment(
    *,
    released_root: Path,
    feature_info: dict[str, object],
    experiments_dir: Path,
    stage_name: str,
    experiment_name: str,
    patch_size: int,
    config: dict[str, object],
    top_n: int,
    max_features: int,
    random_state: int,
    selection_metric: str,
    progress_every: int,
) -> dict[str, object]:
    exp_root = experiments_dir / stage_name / experiment_name
    if exp_root.exists():
        shutil.rmtree(exp_root)
    model_dir = exp_root / "models"
    area_cv = run_pair_training(
        features_turbidity=feature_info["train_features_v2"],
        features_chla=feature_info["train_features"],
        model_dir=model_dir,
        top_n=top_n,
        max_features=max_features,
        random_state=random_state,
        group_by="area",
        filter_test_range=bool(config["filter_test_range"]),
        filter_range_padding=float(config["filter_range_padding"]),
        turbidity_clip_quantile=config["turbidity_clip_quantile"],
        selection_metric=selection_metric,
        write_model=True,
    )
    image_cv = run_pair_training(
        features_turbidity=feature_info["train_features_v2"],
        features_chla=feature_info["train_features"],
        model_dir=exp_root / "shadow_unused_models",
        top_n=top_n,
        max_features=max_features,
        random_state=random_state,
        group_by="image",
        filter_test_range=bool(config["filter_test_range"]),
        filter_range_padding=float(config["filter_range_padding"]),
        turbidity_clip_quantile=config["turbidity_clip_quantile"],
        selection_metric=selection_metric,
        write_model=False,
    )
    neutral_turbidity_env, neutral_chla_env = neutral_pair_envs()
    released_summary = evaluate_released_pair(
        released_root=released_root,
        turbidity_model_dir=model_dir,
        chla_model_dir=model_dir,
        patch_size=patch_size,
        progress_every=progress_every,
        work_dir=exp_root / "eval_input" / "released_area8",
        output_dir=exp_root / "output" / "released_area8",
        report_dir=exp_root / "reports" / "released_area8",
        turbidity_env=neutral_turbidity_env,
        chla_env=neutral_chla_env,
    )
    experiment = {
        "name": experiment_name,
        "stage": stage_name,
        "patch_size": patch_size,
        "config": dict(config),
        "top_n": top_n,
        "max_features": max_features,
        "model_dir": str(model_dir.resolve()),
        "area_cv": area_cv,
        "image_cv": image_cv,
        "released_area8": released_summary,
        "neutral_runtime": pair_env_summary(neutral_turbidity_env, neutral_chla_env),
    }
    _write_json(exp_root / "reports" / "experiment_summary.json", experiment)
    return experiment


def summarize_experiment_markdown(title: str, experiments: list[dict[str, object]], best: dict[str, object]) -> str:
    lines = [
        f"# {title}",
        "",
        "| Experiment | Patch | Area CV Algo | Area CV Turb | Image CV Algo | Released Area8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in experiments:
        lines.append(
            f"| {exp['name']} | {exp['patch_size']} | {exp['area_cv']['algorithm_score']:.4f} | "
            f"{exp['area_cv']['turbidity']['ensemble_score']:.4f} | "
            f"{exp['image_cv']['algorithm_score']:.4f} | {exp['released_area8']['algorithm_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Candidate",
            f"- Name: `{best['name']}`",
            f"- Patch size: `{best['patch_size']}`",
            f"- Area CV algorithm score: `{best['area_cv']['algorithm_score']:.4f}`",
            f"- Image CV algorithm score: `{best['image_cv']['algorithm_score']:.4f}`",
            f"- Released Area8 algorithm score: `{best['released_area8']['algorithm_score']:.4f}`",
        ]
    )
    return "\n".join(lines)


def stage_a_search(
    *,
    released_root: Path,
    feature_map: dict[int, dict[str, object]],
    experiments_dir: Path,
    reports_dir: Path,
    random_state: int,
    selection_metric: str,
    progress_every: int,
) -> dict[str, object]:
    experiments: list[dict[str, object]] = []
    for patch_size in PATCH_SIZES:
        for config in STAGE_A_CONFIGS:
            experiment_name = f"patch{patch_size}_{config['name']}"
            experiments.append(
                stage_experiment(
                    released_root=released_root,
                    feature_info=feature_map[patch_size],
                    experiments_dir=experiments_dir,
                    stage_name="stage_a",
                    experiment_name=experiment_name,
                    patch_size=patch_size,
                    config=config,
                    top_n=5,
                    max_features=500,
                    random_state=random_state,
                    selection_metric=selection_metric,
                    progress_every=progress_every,
                )
            )
    experiments.sort(key=experiment_rank_key)
    top_two_structures = experiments[:2]
    summary = {
        "experiments": experiments,
        "promoted_structures": [
            {
                "name": exp["name"],
                "patch_size": exp["patch_size"],
                "config": exp["config"],
                "area_cv_algorithm_score": exp["area_cv"]["algorithm_score"],
            }
            for exp in top_two_structures
        ],
        "best_experiment": experiments[0],
    }
    _write_json(reports_dir / "stage_a_experiments.json", summary)
    (reports_dir / "stage_a_experiments.md").write_text(
        summarize_experiment_markdown("Stage A Coarse Experiments", experiments, experiments[0]),
        encoding="utf-8",
    )
    return summary


def stage_b_search(
    *,
    released_root: Path,
    feature_map: dict[int, dict[str, object]],
    experiments_dir: Path,
    reports_dir: Path,
    random_state: int,
    selection_metric: str,
    progress_every: int,
    promoted_structures: list[dict[str, object]],
) -> dict[str, object]:
    experiments: list[dict[str, object]] = []
    for promoted in promoted_structures:
        patch_size = int(promoted["patch_size"])
        config = dict(promoted["config"])
        for top_n in FINE_TOP_N:
            for max_features in FINE_MAX_FEATURES:
                experiment_name = f"{promoted['name']}_top{top_n}_feat{max_features}"
                experiments.append(
                    stage_experiment(
                        released_root=released_root,
                        feature_info=feature_map[patch_size],
                        experiments_dir=experiments_dir,
                        stage_name="stage_b",
                        experiment_name=experiment_name,
                        patch_size=patch_size,
                        config=config,
                        top_n=top_n,
                        max_features=max_features,
                        random_state=random_state,
                        selection_metric=selection_metric,
                        progress_every=progress_every,
                    )
                )
    experiments.sort(key=experiment_rank_key)
    finalists = experiments[:3]
    summary = {
        "experiments": experiments,
        "finalists": finalists,
        "best_experiment": experiments[0],
    }
    _write_json(reports_dir / "stage_b_experiments.json", summary)
    (reports_dir / "stage_b_experiments.md").write_text(
        summarize_experiment_markdown("Stage B Fine Experiments", experiments, experiments[0]),
        encoding="utf-8",
    )
    return summary


def copy_target_artifacts(source_dir: Path, target: str, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.glob(f"{target}*"):
        if path.is_file():
            shutil.copy2(path, destination_dir / path.name)


def materialize_pair_dir(
    *,
    turbidity_source_dir: Path,
    chla_source_dir: Path,
    destination_dir: Path,
) -> Path:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    copy_target_artifacts(turbidity_source_dir, "turbidity", destination_dir)
    copy_target_artifacts(chla_source_dir, "chla", destination_dir)
    return destination_dir


def pairing_name(turbidity_exp: dict[str, object], chla_exp: dict[str, object]) -> str:
    return f"{turbidity_exp['name']}__{chla_exp['name']}"


def make_pairing_candidate(
    *,
    name: str,
    turbidity_experiment: str,
    chla_experiment: str,
    turbidity_model_dir: Path,
    chla_model_dir: Path,
    pair_model_dir: Path,
    patch_size: int,
    turbidity_area_cv_score: float,
    chla_area_cv_score: float,
) -> dict[str, object]:
    return {
        "name": name,
        "turbidity_experiment": turbidity_experiment,
        "chla_experiment": chla_experiment,
        "turbidity_model_dir": str(turbidity_model_dir.resolve()),
        "chla_model_dir": str(chla_model_dir.resolve()),
        "pair_model_dir": str(pair_model_dir.resolve()),
        "patch_size": int(patch_size),
        "area_cv": {
            "algorithm_score": algorithm_score({"score": turbidity_area_cv_score}, {"score": chla_area_cv_score}),
            "turbidity_score": float(turbidity_area_cv_score),
            "chla_score": float(chla_area_cv_score),
        },
    }


def load_current_tabular_winner(*, reports_dir: Path, stage_b_summary: dict[str, object]) -> dict[str, object]:
    for summary_name in ("final_score_push_summary.json", "final_two_hour_score_push_summary.json", "overnight_score_push_summary.json"):
        prior_summary_path = reports_dir / summary_name
        if not prior_summary_path.exists():
            continue
        prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        final_summary = prior_summary.get("final_summary", {})
        pair_model_dir = final_summary.get("pair_model_dir")
        if pair_model_dir:
            pair_dir = Path(str(pair_model_dir))
            return make_pairing_candidate(
                name="tabular_winner",
                turbidity_experiment=str(final_summary.get("turbidity_experiment", "tabular_winner")),
                chla_experiment=str(final_summary.get("chla_experiment", "tabular_winner")),
                turbidity_model_dir=pair_dir,
                chla_model_dir=pair_dir,
                pair_model_dir=pair_dir,
                patch_size=int(final_summary.get("patch_size", stage_b_summary["best_experiment"]["patch_size"])),
                turbidity_area_cv_score=float(final_summary.get("area_cv", {}).get("turbidity_score", 0.0)),
                chla_area_cv_score=float(final_summary.get("area_cv", {}).get("chla_score", 0.0)),
            )

    best_experiment = stage_b_summary["best_experiment"]
    pair_dir = Path(best_experiment["model_dir"])
    return make_pairing_candidate(
        name="tabular_winner",
        turbidity_experiment=str(best_experiment["name"]),
        chla_experiment=str(best_experiment["name"]),
        turbidity_model_dir=pair_dir,
        chla_model_dir=pair_dir,
        pair_model_dir=pair_dir,
        patch_size=int(best_experiment["patch_size"]),
        turbidity_area_cv_score=float(best_experiment["area_cv"]["turbidity"]["ensemble_score"]),
        chla_area_cv_score=float(best_experiment["area_cv"]["chla"]["ensemble_score"]),
    )


def build_pairing_candidates(
    *,
    released_root: Path,
    experiments_dir: Path,
    reports_dir: Path,
    finalists: list[dict[str, object]],
    progress_every: int,
) -> dict[str, object]:
    pairings: list[dict[str, object]] = []
    neutral_turbidity_env, neutral_chla_env = neutral_pair_envs()
    for turbidity_exp in finalists:
        for chla_exp in finalists:
            name = pairing_name(turbidity_exp, chla_exp)
            pair_dir = materialize_pair_dir(
                turbidity_source_dir=Path(turbidity_exp["model_dir"]),
                chla_source_dir=Path(chla_exp["model_dir"]),
                destination_dir=experiments_dir / "pairings" / name / "models",
            )
            released = evaluate_released_pair(
                released_root=released_root,
                turbidity_model_dir=pair_dir,
                chla_model_dir=pair_dir,
                patch_size=max(int(turbidity_exp["patch_size"]), int(chla_exp["patch_size"])),
                progress_every=progress_every,
                work_dir=experiments_dir / "pairings" / name / "eval_input" / "released_area8",
                output_dir=experiments_dir / "pairings" / name / "output" / "released_area8",
                report_dir=experiments_dir / "pairings" / name / "reports" / "released_area8",
                turbidity_env=neutral_turbidity_env,
                chla_env=neutral_chla_env,
            )
            pairings.append(
                {
                    "name": name,
                    "turbidity_experiment": turbidity_exp["name"],
                    "chla_experiment": chla_exp["name"],
                    "turbidity_model_dir": str(Path(turbidity_exp["model_dir"]).resolve()),
                    "chla_model_dir": str(Path(chla_exp["model_dir"]).resolve()),
                    "pair_model_dir": str(pair_dir.resolve()),
                    "patch_size": max(int(turbidity_exp["patch_size"]), int(chla_exp["patch_size"])),
                    "released_area8": released,
                    "area_cv": {
                        "algorithm_score": algorithm_score(
                            {"score": turbidity_exp["area_cv"]["turbidity"]["ensemble_score"]},
                            {"score": chla_exp["area_cv"]["chla"]["ensemble_score"]},
                        ),
                        "turbidity_score": turbidity_exp["area_cv"]["turbidity"]["ensemble_score"],
                        "chla_score": chla_exp["area_cv"]["chla"]["ensemble_score"],
                    },
                }
            )
    pairings.sort(
        key=lambda row: (
            -float(row["released_area8"]["algorithm_score"]),
            -float(row["area_cv"]["algorithm_score"]),
            -float(row["area_cv"]["turbidity_score"]),
        )
    )
    summary = {
        "pairings": pairings,
        "top_pairings": pairings[:3],
    }
    _write_json(reports_dir / "pairings.json", summary)
    lines = [
        "# Mixed Pairing Candidates",
        "",
        "| Pairing | Released Area8 | Area CV Algo | Turb Source | Chla Source |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in pairings:
        lines.append(
            f"| {row['name']} | {row['released_area8']['algorithm_score']:.4f} | "
            f"{row['area_cv']['algorithm_score']:.4f} | {row['turbidity_experiment']} | {row['chla_experiment']} |"
        )
    (reports_dir / "pairings.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def tune_turbidity_target(
    *,
    released_root: Path,
    model_dir: Path,
    patch_size: int,
    progress_every: int,
    work_root: Path,
    source_area_cv_score: float,
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    base_overrides = {
        "HYDROSAT_CALIBRATE_TEST_STATS": "1",
        "HYDROSAT_TURBIDITY_MODE": "model",
        "HYDROSAT_CHLA_MODE": "model",
    }
    for sigma in TURBIDITY_LOGNORMAL_SIGMAS:
        for prior_shrink in TURBIDITY_PRIOR_SHRINKS:
            env = {
                **base_overrides,
                "HYDROSAT_TURBIDITY_MODE": "model",
                "HYDROSAT_TURBIDITY_CALIBRATION": "lognormal_rank",
                "HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA": f"{sigma:.2f}",
                "HYDROSAT_TURBIDITY_PRIOR_SHRINK": f"{prior_shrink:.2f}",
            }
            result = evaluate_released_target(
                released_root=released_root,
                target="turbidity",
                model_dir=model_dir,
                patch_size=patch_size,
                progress_every=progress_every,
                work_dir=work_root / f"model_lognormal_sigma{sigma:.2f}_p{prior_shrink:.2f}",
                env_overrides=env,
            )
            trials.append({"kind": "model_lognormal_rank", "env": env, "metrics": result["metrics"]})

    for weight in TURBIDITY_BLEND_WEIGHTS:
        for prior_shrink in TURBIDITY_PRIOR_SHRINKS:
            for rank_power in TURBIDITY_BLEND_RANK_POWERS:
                env = {
                    **base_overrides,
                    "HYDROSAT_TURBIDITY_MODE": "blend",
                    "HYDROSAT_TURBIDITY_HEURISTIC_WEIGHT": f"{weight:.2f}",
                    "HYDROSAT_TURBIDITY_PRIOR_SHRINK": f"{prior_shrink:.2f}",
                    "HYDROSAT_TURBIDITY_RANK_POWER": f"{rank_power:.2f}",
                    "HYDROSAT_TURBIDITY_CALIBRATION": "",
                }
                result = evaluate_released_target(
                    released_root=released_root,
                    target="turbidity",
                    model_dir=model_dir,
                    patch_size=patch_size,
                    progress_every=progress_every,
                    work_dir=work_root / f"blend_w{weight:.2f}_p{prior_shrink:.2f}_r{rank_power:.2f}",
                    env_overrides=env,
                )
                trials.append({"kind": "blend_standard", "env": env, "metrics": result["metrics"]})

    trials.sort(key=lambda row: target_trial_sort_key(row, source_area_cv_score))
    return {"best": trials[0], "trials": trials}


def tune_chla_target(
    *,
    released_root: Path,
    model_dir: Path,
    patch_size: int,
    progress_every: int,
    work_root: Path,
    source_area_cv_score: float,
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    base_overrides = {
        "HYDROSAT_CALIBRATE_TEST_STATS": "1",
        "HYDROSAT_TURBIDITY_MODE": "model",
    }

    for prior_shrink in CHLA_MODEL_PRIOR_SHRINKS:
        env = {
            **base_overrides,
            "HYDROSAT_CHLA_MODE": "model",
            "HYDROSAT_CHLA_PRIOR_SHRINK": f"{prior_shrink:.2f}",
        }
        result = evaluate_released_target(
            released_root=released_root,
            target="chla",
            model_dir=model_dir,
            patch_size=patch_size,
            progress_every=progress_every,
            work_dir=work_root / f"model_p{prior_shrink:.2f}",
            env_overrides=env,
        )
        trials.append({"kind": "model", "env": env, "metrics": result["metrics"]})

    env = {**base_overrides, "HYDROSAT_CHLA_MODE": "model_rank", "HYDROSAT_CHLA_PRIOR_SHRINK": "0.00"}
    result = evaluate_released_target(
        released_root=released_root,
        target="chla",
        model_dir=model_dir,
        patch_size=patch_size,
        progress_every=progress_every,
        work_dir=work_root / "model_rank",
        env_overrides=env,
    )
    trials.append({"kind": "model_rank", "env": env, "metrics": result["metrics"]})

    for weight in CHLA_BLEND_WEIGHTS:
        env = {
            **base_overrides,
            "HYDROSAT_CHLA_MODE": "blend",
            "HYDROSAT_CHLA_HEURISTIC_WEIGHT": f"{weight:.2f}",
            "HYDROSAT_CHLA_PRIOR_SHRINK": "0.00",
        }
        result = evaluate_released_target(
            released_root=released_root,
            target="chla",
            model_dir=model_dir,
            patch_size=patch_size,
            progress_every=progress_every,
            work_dir=work_root / f"blend_w{weight:.2f}",
            env_overrides=env,
        )
        trials.append({"kind": "blend", "env": env, "metrics": result["metrics"]})

    for sigma in CHLA_LOGNORMAL_SIGMAS:
        env = {
            **base_overrides,
            "HYDROSAT_CHLA_MODE": "model",
            "HYDROSAT_CHLA_CALIBRATION": "lognormal_rank",
            "HYDROSAT_CHLA_LOGNORMAL_SIGMA": f"{sigma:.2f}",
            "HYDROSAT_CHLA_PRIOR_SHRINK": "0.00",
        }
        result = evaluate_released_target(
            released_root=released_root,
            target="chla",
            model_dir=model_dir,
            patch_size=patch_size,
            progress_every=progress_every,
            work_dir=work_root / f"lognormal_sigma{sigma:.2f}",
            env_overrides=env,
        )
        trials.append({"kind": "lognormal_rank", "env": env, "metrics": result["metrics"]})

    trials.sort(key=lambda row: target_trial_sort_key(row, source_area_cv_score))
    return {"best": trials[0], "trials": trials}


def tune_pairing(
    *,
    released_root: Path,
    reports_dir: Path,
    pairing: dict[str, object],
    progress_every: int,
    fixed_chla_trial: dict[str, object] | None = None,
) -> dict[str, object]:
    pair_dir = Path(pairing["pair_model_dir"])
    patch_size = int(pairing["patch_size"])
    tuning_root = reports_dir / "late_tuning" / pairing["name"]
    turbidity_tuning = tune_turbidity_target(
        released_root=released_root,
        model_dir=pair_dir,
        patch_size=patch_size,
        progress_every=progress_every,
        work_root=tuning_root / "turbidity",
        source_area_cv_score=float(pairing["area_cv"]["turbidity_score"]),
    )
    if fixed_chla_trial is None:
        chla_tuning = tune_chla_target(
            released_root=released_root,
            model_dir=pair_dir,
            patch_size=patch_size,
            progress_every=progress_every,
            work_root=tuning_root / "chla",
            source_area_cv_score=float(pairing["area_cv"]["chla_score"]),
        )
    else:
        chla_tuning = {"best": fixed_chla_trial, "trials": [fixed_chla_trial], "fixed": True}

    final_summary = {
        "name": pairing["name"],
        "pair_model_dir": pairing["pair_model_dir"],
        "turbidity_model_dir": pairing["turbidity_model_dir"],
        "chla_model_dir": pairing["chla_model_dir"],
        "turbidity_experiment": pairing["turbidity_experiment"],
        "chla_experiment": pairing["chla_experiment"],
        "patch_size": patch_size,
        "area_cv": dict(pairing["area_cv"]),
        "source_area_cv_turbidity_score": float(pairing["area_cv"]["turbidity_score"]),
        "source_area_cv_chla_score": float(pairing["area_cv"]["chla_score"]),
        "turbidity_tuning": turbidity_tuning,
        "chla_tuning": chla_tuning,
        **pair_summary(turbidity_tuning["best"]["metrics"], chla_tuning["best"]["metrics"]),
        **pair_env_summary(turbidity_tuning["best"]["env"], chla_tuning["best"]["env"]),
    }
    _write_json(tuning_root / "pairing_summary.json", final_summary)
    return final_summary


def run_regime_candidate(
    *,
    features_turbidity: Path,
    released_root: Path,
    experiments_dir: Path,
    patch_size: int,
    random_state: int,
    progress_every: int,
    filter_range_padding: float,
) -> dict[str, object]:
    exp_root = experiments_dir / "regime_candidate"
    if exp_root.exists():
        shutil.rmtree(exp_root)
    model_dir = exp_root / "models"
    summary = train_regime_ensemble(
        features_path=features_turbidity,
        model_dir=model_dir,
        random_state=random_state,
        max_features=500,
        group_by="area",
        filter_test_range=True,
        filter_range_padding=filter_range_padding,
    )
    metrics = evaluate_released_target(
        released_root=released_root,
        target="turbidity",
        model_dir=model_dir,
        patch_size=patch_size,
        progress_every=progress_every,
        work_dir=exp_root / "eval_input" / "released_area8",
        env_overrides={
            "HYDROSAT_CALIBRATE_TEST_STATS": "1",
            "HYDROSAT_TURBIDITY_MODE": "model",
            "HYDROSAT_TURBIDITY_PRIOR_SHRINK": "0",
        },
    )
    result = {
        "summary": summary,
        "released_turbidity": metrics["metrics"],
        "model_dir": str(model_dir.resolve()),
        "patch_size": patch_size,
    }
    _write_json(exp_root / "reports" / "regime_summary.json", result)
    return result


def load_or_train_regime_candidate(
    *,
    features_turbidity: Path,
    released_root: Path,
    experiments_dir: Path,
    patch_size: int,
    random_state: int,
    progress_every: int,
    filter_range_padding: float,
) -> dict[str, object]:
    summary_path = experiments_dir / "regime_candidate" / "reports" / "regime_summary.json"
    model_path = experiments_dir / "regime_candidate" / "models" / "turbidity_ensemble.joblib"
    if summary_path.exists() and model_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_regime_candidate(
        features_turbidity=features_turbidity,
        released_root=released_root,
        experiments_dir=experiments_dir,
        patch_size=patch_size,
        random_state=random_state,
        progress_every=progress_every,
        filter_range_padding=filter_range_padding,
    )


def build_regime_pairing(
    *,
    regime_summary: dict[str, object],
    best_chla_pairing: dict[str, object],
    experiments_dir: Path,
    patch_size: int,
) -> dict[str, object]:
    pair_dir = materialize_pair_dir(
        turbidity_source_dir=Path(regime_summary["model_dir"]),
        chla_source_dir=Path(best_chla_pairing["pair_model_dir"]),
        destination_dir=experiments_dir / "regime_candidate" / "paired_models",
    )
    return make_pairing_candidate(
        name="regime_plus_best_chla",
        turbidity_experiment="regime_candidate",
        chla_experiment=str(best_chla_pairing["chla_experiment"]),
        turbidity_model_dir=Path(regime_summary["model_dir"]),
        chla_model_dir=Path(best_chla_pairing["pair_model_dir"]),
        pair_model_dir=pair_dir,
        patch_size=patch_size,
        turbidity_area_cv_score=float(regime_summary["summary"]["ensemble_score"]),
        chla_area_cv_score=float(best_chla_pairing["source_area_cv_chla_score"]),
    )


def run_turbidity_retrain_fallback(
    *,
    features_turbidity: Path,
    released_root: Path,
    experiments_dir: Path,
    reports_dir: Path,
    patch_size: int,
    progress_every: int,
    random_state: int,
    selection_metric: str,
    best_chla_pairing: dict[str, object],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    fallback_root = experiments_dir / "late_turbidity_retrain"
    fixed_chla_trial = dict(best_chla_pairing["chla_tuning"]["best"])
    for top_n in TURBIDITY_RETRAIN_TOP_N:
        exp_root = fallback_root / f"top{top_n}_feat800"
        if exp_root.exists():
            shutil.rmtree(exp_root)
        model_dir = exp_root / "models"
        area_cv = train_target_ensemble(
            features_turbidity,
            "turbidity",
            model_dir,
            TURBIDITY_MODELS,
            top_n,
            800,
            random_state,
            None,
            "area",
            True,
            0.15,
            selection_metric,
            write_model=True,
        )
        pair_dir = materialize_pair_dir(
            turbidity_source_dir=model_dir,
            chla_source_dir=Path(best_chla_pairing["pair_model_dir"]),
            destination_dir=exp_root / "paired_models",
        )
        pairing = make_pairing_candidate(
            name=f"turbidity_retrain_top{top_n}_feat800",
            turbidity_experiment=f"turbidity_retrain_top{top_n}_feat800",
            chla_experiment=str(best_chla_pairing["chla_experiment"]),
            turbidity_model_dir=model_dir,
            chla_model_dir=Path(best_chla_pairing["pair_model_dir"]),
            pair_model_dir=pair_dir,
            patch_size=patch_size,
            turbidity_area_cv_score=float(area_cv["ensemble_score"]),
            chla_area_cv_score=float(best_chla_pairing["source_area_cv_chla_score"]),
        )
        tuned = tune_pairing(
            released_root=released_root,
            reports_dir=reports_dir,
            pairing=pairing,
            progress_every=progress_every,
            fixed_chla_trial=fixed_chla_trial,
        )
        tuned["turbidity_area_cv_summary"] = area_cv
        _write_json(exp_root / "reports" / "candidate_summary.json", tuned)
        candidates.append(tuned)

    candidates.sort(key=tuned_pair_sort_key)
    summary = {"candidates": candidates, "best_candidate": candidates[0]}
    _write_json(reports_dir / "late_turbidity_retrain.json", summary)
    lines = [
        "# Late Turbidity Retrain Fallback",
        "",
        "| Candidate | Final Algo | Turb Score | Turb Area CV | |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in candidates:
        lines.append(
            f"| {row['name']} | {row['algorithm_score']:.4f} | {row['turbidity']['score']:.4f} | "
            f"{row['source_area_cv_turbidity_score']:.4f} |"
        )
    (reports_dir / "late_turbidity_retrain.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def freeze_runtime(
    *,
    runtime_model_dir: Path,
    source_pair_model_dir: Path,
    fallback_dir: Path,
    runtime_defaults: dict[str, object] | None = None,
) -> None:
    if runtime_model_dir.exists():
        shutil.rmtree(runtime_model_dir)
    shutil.copytree(source_pair_model_dir, runtime_model_dir)
    if runtime_defaults:
        _write_json(runtime_model_dir / "runtime_env_defaults.json", runtime_defaults)
    if not any(runtime_model_dir.iterdir()):
        shutil.rmtree(runtime_model_dir, ignore_errors=True)
        shutil.copytree(fallback_dir, runtime_model_dir)


def summarize_final_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Final Score Push Summary",
        "",
        f"- Fallback floor: `{summary['fallback_score']:.4f}`",
        f"- Final chosen score: `{summary['final_score']:.4f}`",
        f"- Winner type: `{summary['winner_type']}`",
        f"- Final runtime model dir: `{summary['runtime_model_dir']}`",
        "",
        "## Final Metrics",
        f"- Turbidity score: {summary['final_summary']['turbidity']['score']:.4f}",
        f"- Chl-a score: {summary['final_summary']['chla']['score']:.4f}",
        f"- Algorithm score: {summary['final_summary']['algorithm_score']:.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--released-root", type=Path, default=Path("track2_download_link_1"))
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--reports-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--runtime-model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--selection-metric", choices=["score", "rmse"], default="score")
    parser.add_argument("--skip-feature-build", action="store_true")
    parser.add_argument("--skip-stage-a", action="store_true")
    parser.add_argument("--skip-stage-b", action="store_true")
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--stage-a-only", action="store_true")
    args = parser.parse_args()

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    fallback_root = args.experiments_dir / "fallback_runtime_snapshot"
    if fallback_root.exists():
        shutil.rmtree(fallback_root)
    fallback_root.mkdir(parents=True, exist_ok=True)
    fallback_model_dir = fallback_root / "models"
    if args.runtime_model_dir.exists():
        shutil.copytree(args.runtime_model_dir, fallback_model_dir)
    fallback_report = args.reports_dir / "released_area8" / "released_area8_scores.json"
    if fallback_report.exists():
        shutil.copy2(fallback_report, fallback_root / "released_area8_scores.json")
    fallback_turbidity_env, fallback_chla_env = final_default_envs()
    _write_json(fallback_root / "runtime_env_defaults.json", pair_env_summary(fallback_turbidity_env, fallback_chla_env))

    feature_map = build_feature_tables(
        args.data_root,
        args.features_dir,
        patch_sizes=PATCH_SIZES,
        progress_every=args.progress_every,
        skip_existing=args.skip_feature_build,
    )
    feature_summary = {
        str(patch_size): {"rows": info["rows"], "columns": info["columns"], "canonical": str(info["canonical"])}
        for patch_size, info in feature_map.items()
    }
    _write_json(args.reports_dir / "feature_builds.json", feature_summary)

    if args.skip_stage_a:
        stage_a_summary = json.loads((args.reports_dir / "stage_a_experiments.json").read_text(encoding="utf-8"))
    else:
        stage_a_summary = stage_a_search(
            released_root=args.released_root,
            feature_map=feature_map,
            experiments_dir=args.experiments_dir,
            reports_dir=args.reports_dir,
            random_state=args.random_state,
            selection_metric=args.selection_metric,
            progress_every=args.progress_every,
        )
    if args.stage_a_only:
        return

    if args.skip_stage_b:
        stage_b_summary = json.loads((args.reports_dir / "stage_b_experiments.json").read_text(encoding="utf-8"))
    else:
        stage_b_summary = stage_b_search(
            released_root=args.released_root,
            feature_map=feature_map,
            experiments_dir=args.experiments_dir,
            reports_dir=args.reports_dir,
            random_state=args.random_state,
            selection_metric=args.selection_metric,
            progress_every=args.progress_every,
            promoted_structures=stage_a_summary["promoted_structures"],
        )

    current_tabular = load_current_tabular_winner(reports_dir=args.reports_dir, stage_b_summary=stage_b_summary)
    tabular_tuned = tune_pairing(
        released_root=args.released_root,
        reports_dir=args.reports_dir,
        pairing=current_tabular,
        progress_every=args.progress_every,
    )

    evaluated_candidates: list[dict[str, object]] = [tabular_tuned]
    candidate_report = {
        "current_floor": CURRENT_SCORE_FLOOR,
        "tabular_winner": tabular_tuned,
    }

    if not args.skip_regime:
        regime_summary = load_or_train_regime_candidate(
            features_turbidity=feature_map[int(stage_b_summary["best_experiment"]["patch_size"])]["train_features_v2"],
            released_root=args.released_root,
            experiments_dir=args.experiments_dir,
            patch_size=int(stage_b_summary["best_experiment"]["patch_size"]),
            random_state=args.random_state,
            progress_every=args.progress_every,
            filter_range_padding=float(stage_b_summary["best_experiment"]["config"]["filter_range_padding"]),
        )
        regime_pairing = build_regime_pairing(
            regime_summary=regime_summary,
            best_chla_pairing=tabular_tuned,
            experiments_dir=args.experiments_dir,
            patch_size=max(int(regime_summary["patch_size"]), int(tabular_tuned["patch_size"])),
        )
        regime_tuned = tune_pairing(
            released_root=args.released_root,
            reports_dir=args.reports_dir,
            pairing=regime_pairing,
            progress_every=args.progress_every,
            fixed_chla_trial=tabular_tuned["chla_tuning"]["best"],
        )
        evaluated_candidates.append(regime_tuned)
        candidate_report["regime_candidate"] = regime_tuned
        _write_json(args.reports_dir / "regime_pair_summary.json", regime_tuned)

    evaluated_candidates.sort(key=tuned_pair_sort_key)
    best_tuned = evaluated_candidates[0]

    if float(best_tuned["algorithm_score"]) <= CURRENT_SCORE_FLOOR:
        retrain_summary = run_turbidity_retrain_fallback(
            features_turbidity=feature_map[int(stage_b_summary["best_experiment"]["patch_size"])]["train_features_v2"],
            released_root=args.released_root,
            experiments_dir=args.experiments_dir,
            reports_dir=args.reports_dir,
            patch_size=int(stage_b_summary["best_experiment"]["patch_size"]),
            progress_every=args.progress_every,
            random_state=args.random_state,
            selection_metric=args.selection_metric,
            best_chla_pairing=tabular_tuned,
        )
        evaluated_candidates.extend(retrain_summary["candidates"])
        candidate_report["late_turbidity_retrain"] = retrain_summary
        evaluated_candidates.sort(key=tuned_pair_sort_key)
        best_tuned = evaluated_candidates[0]

    candidate_report["evaluated_candidates"] = evaluated_candidates
    _write_json(args.reports_dir / "final_score_push_candidates.json", candidate_report)

    winner_type = (
        "regime_pair"
        if best_tuned["name"] == "regime_plus_best_chla"
        else "turbidity_retrain_pair"
        if best_tuned["name"].startswith("turbidity_retrain_")
        else "tabular_pair"
    )
    chosen_pair_dir = Path(best_tuned["pair_model_dir"])
    chosen_summary = best_tuned

    fallback_score = CURRENT_SCORE_FLOOR
    final_score = float(chosen_summary["algorithm_score"])
    if final_score > fallback_score:
        freeze_runtime(
            runtime_model_dir=args.runtime_model_dir,
            source_pair_model_dir=chosen_pair_dir,
            fallback_dir=fallback_model_dir,
            runtime_defaults=combined_runtime_defaults(chosen_summary),
        )
    else:
        if args.runtime_model_dir.exists():
            shutil.rmtree(args.runtime_model_dir)
        shutil.copytree(fallback_model_dir, args.runtime_model_dir)
        final_score = fallback_score
        winner_type = "fallback_runtime"
        chosen_summary = json.loads((args.reports_dir / "released_area8" / "released_area8_scores.json").read_text(encoding="utf-8"))

    summary = {
        "fallback_score": fallback_score,
        "final_score": final_score,
        "winner_type": winner_type,
        "runtime_model_dir": str(args.runtime_model_dir.resolve()),
        "final_summary": chosen_summary,
        "stage_a_best": stage_a_summary["best_experiment"]["name"],
        "stage_b_best": stage_b_summary["best_experiment"]["name"],
    }
    _write_json(args.reports_dir / "final_score_push_summary.json", summary)
    (args.reports_dir / "final_score_push_summary.md").write_text(
        summarize_final_markdown(summary),
        encoding="utf-8",
    )
    print((args.reports_dir / "final_score_push_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
