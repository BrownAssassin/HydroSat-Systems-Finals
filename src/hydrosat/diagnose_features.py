from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .train_baseline import META_COLUMNS, feature_columns


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.4f}"


def summarize_target(df: pd.DataFrame, target: str) -> None:
    part = df[df["target"] == target].copy()
    if part.empty:
        print(f"\n{target}: no rows")
        return

    y = part["y"]
    print(f"\n{target}")
    print(f"rows={len(part)} images={part['filename'].nunique()} areas={part['area'].nunique()}")
    print(
        "y "
        f"min={_fmt(y.min())} p10={_fmt(y.quantile(0.10))} p50={_fmt(y.quantile(0.50))} "
        f"p90={_fmt(y.quantile(0.90))} max={_fmt(y.max())} mean={_fmt(y.mean())} std={_fmt(y.std())}"
    )
    print("by area:")
    area_summary = (
        part.groupby("area")["y"]
        .agg(rows="count", mean="mean", median="median", std="std", min="min", max="max")
        .sort_index()
    )
    print(area_summary.to_string(float_format=lambda x: f"{x:.3f}"))

    numeric_cols = feature_columns(part)
    missing = part[numeric_cols].isna().mean().sort_values(ascending=False).head(12)
    if missing.iloc[0] > 0:
        print("highest feature missingness:")
        print(missing[missing > 0].to_string(float_format=lambda x: f"{x:.3f}"))

    corr_rows = []
    for col in numeric_cols:
        values = part[col]
        if values.nunique(dropna=True) < 2:
            continue
        corr = values.corr(y, method="spearman")
        if pd.notna(corr):
            corr_rows.append((col, corr))
    top_corr = sorted(corr_rows, key=lambda item: abs(item[1]), reverse=True)[:20]
    print("top absolute Spearman correlations:")
    for name, corr in top_corr:
        print(f"  {name}: {corr:.3f}")

    high_outliers = part[y > y.quantile(0.99)][["area", "filename", "lon", "lat", "y"]]
    if not high_outliers.empty:
        print("top 1% high-label points:")
        print(high_outliers.sort_values("y", ascending=False).head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/train_features.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    print(f"loaded {len(df)} rows x {len(df.columns)} columns from {args.features}")
    print(f"numeric feature columns: {len([c for c in df.columns if c not in META_COLUMNS and np.issubdtype(df[c].dtype, np.number)])}")
    for target in ("turbidity", "chla"):
        summarize_target(df, target)


if __name__ == "__main__":
    main()

