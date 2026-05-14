from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .build_features import build_features
from .evaluate_released_area8 import evaluate_released_area8
from .scoring import algorithm_score
from .train_ensemble import train_target_ensemble


TURBIDITY_MODELS = ["lightgbm", "extra", "catboost", "lightgbm_log", "extra_log"]
CHLA_MODELS = ["catboost", "extra", "hgb", "lightgbm", "lightgbm_log"]

EXPERIMENTS = [
    {
        "name": "area_full_score",
        "filter_test_range": False,
        "filter_range_padding": 0.05,
        "turbidity_clip_quantile": None,
    },
    {
        "name": "area_filter05_score",
        "filter_test_range": True,
        "filter_range_padding": 0.05,
        "turbidity_clip_quantile": None,
    },
    {
        "name": "area_filter10_score",
        "filter_test_range": True,
        "filter_range_padding": 0.10,
        "turbidity_clip_quantile": None,
    },
    {
        "name": "area_clip99_score",
        "filter_test_range": False,
        "filter_range_padding": 0.05,
        "turbidity_clip_quantile": 0.99,
    },
]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_feature_tables(data_root: Path, features_dir: Path, patch_size: int, progress_every: int) -> dict[str, Path]:
    features_dir.mkdir(parents=True, exist_ok=True)
    df = build_features(data_root, patch_size=patch_size, progress_every=progress_every)
    canonical = features_dir / "train_features_area32.csv"
    legacy_a = features_dir / "train_features.csv"
    legacy_b = features_dir / "train_features_v2.csv"
    for path in [canonical, legacy_a, legacy_b]:
        df.to_csv(path, index=False)
    return {
        "canonical": canonical,
        "train_features": legacy_a,
        "train_features_v2": legacy_b,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }


def run_pair(
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


def summarize_experiment_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Score Push Experiments",
        "",
        f"- Built at patch size `{summary['patch_size']}`",
        f"- Feature rows `{summary['feature_rows']}`",
        f"- Feature columns `{summary['feature_columns']}`",
        "",
        "| Experiment | Area CV Algo | Area CV Turb | Image CV Algo | Released Area8 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for exp in summary["experiments"]:
        lines.append(
            f"| {exp['name']} | {exp['area_cv']['algorithm_score']:.4f} | "
            f"{exp['area_cv']['turbidity']['ensemble_score']:.4f} | "
            f"{exp['image_cv']['algorithm_score']:.4f} | "
            f"{exp['released_area8']['algorithm_score']:.4f} |"
        )
    best = summary["best_experiment"]
    lines.extend(
        [
            "",
            "## Best Candidate",
            f"- Name: `{best['name']}`",
            f"- Area CV algorithm score: `{best['area_cv']['algorithm_score']:.4f}`",
            f"- Image CV algorithm score: `{best['image_cv']['algorithm_score']:.4f}`",
            f"- Released Area8 algorithm score: `{best['released_area8']['algorithm_score']:.4f}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--released-root", type=Path, default=Path("track2_download_link_1"))
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--reports-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--runtime-model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--selection-metric", choices=["score", "rmse"], default="score")
    parser.add_argument("--skip-feature-build", action="store_true")
    parser.add_argument("--no-freeze-best", action="store_true")
    parser.add_argument("--only-experiments", default="")
    args = parser.parse_args()

    if args.skip_feature_build:
        feature_info = {
            "canonical": args.features_dir / "train_features_area32.csv",
            "train_features": args.features_dir / "train_features.csv",
            "train_features_v2": args.features_dir / "train_features_v2.csv",
        }
        df = pd.read_csv(feature_info["canonical"])
        feature_info["rows"] = int(len(df))
        feature_info["columns"] = int(len(df.columns))
    else:
        feature_info = build_feature_tables(args.data_root, args.features_dir, args.patch_size, args.progress_every)

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    fallback_dir = args.experiments_dir / "fallback_models_before_score_push"
    if fallback_dir.exists():
        shutil.rmtree(fallback_dir)
    if args.runtime_model_dir.exists():
        shutil.copytree(args.runtime_model_dir, fallback_dir)

    selected_names = {name.strip() for name in args.only_experiments.split(",") if name.strip()}
    experiment_configs = [config for config in EXPERIMENTS if not selected_names or config["name"] in selected_names]
    if not experiment_configs:
        raise RuntimeError("No experiments selected")

    experiments: list[dict[str, object]] = []
    for config in experiment_configs:
        name = config["name"]
        exp_root = args.experiments_dir / name
        if exp_root.exists():
            shutil.rmtree(exp_root)
        model_dir = exp_root / "models"
        area_cv = run_pair(
            feature_info["train_features_v2"],
            feature_info["train_features"],
            model_dir,
            args.top_n,
            args.max_features,
            args.random_state,
            "area",
            bool(config["filter_test_range"]),
            float(config["filter_range_padding"]),
            config["turbidity_clip_quantile"],
            args.selection_metric,
            write_model=True,
        )
        image_cv = run_pair(
            feature_info["train_features_v2"],
            feature_info["train_features"],
            exp_root / "shadow_unused_models",
            args.top_n,
            args.max_features,
            args.random_state,
            "image",
            bool(config["filter_test_range"]),
            float(config["filter_range_padding"]),
            config["turbidity_clip_quantile"],
            args.selection_metric,
            write_model=False,
        )
        released = evaluate_released_area8(
            released_root=args.released_root,
            model_dir=model_dir,
            work_dir=exp_root / "eval_input" / "released_area8",
            output_dir=exp_root / "output" / "released_area8",
            report_dir=exp_root / "reports" / "released_area8",
            patch_size=args.patch_size,
            progress_every=args.progress_every,
        )
        experiment = {
            "name": name,
            "config": config,
            "area_cv": area_cv,
            "image_cv": image_cv,
            "released_area8": released,
        }
        _write_json(exp_root / "reports" / "experiment_summary.json", experiment)
        experiments.append(experiment)

    experiments.sort(
        key=lambda exp: (
            -float(exp["area_cv"]["algorithm_score"]),
            -float(exp["area_cv"]["turbidity"]["ensemble_score"]),
            -float(exp["image_cv"]["algorithm_score"]),
        )
    )
    best = experiments[0]
    if not args.no_freeze_best:
        if args.runtime_model_dir.exists():
            shutil.rmtree(args.runtime_model_dir)
        shutil.copytree(args.experiments_dir / best["name"] / "models", args.runtime_model_dir)

    summary = {
        "patch_size": args.patch_size,
        "feature_rows": feature_info["rows"],
        "feature_columns": feature_info["columns"],
        "experiments": experiments,
        "best_experiment": best,
        "fallback_dir": str(fallback_dir.resolve()),
    }
    summary_json = args.reports_dir / "score_push_experiments.json"
    summary_md = args.reports_dir / "score_push_experiments.md"
    _write_json(summary_json, summary)
    summary_md.write_text(summarize_experiment_markdown(summary), encoding="utf-8")
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
