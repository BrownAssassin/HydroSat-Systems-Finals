from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import row_features
from .paths import find_train_csvs, target_column, target_from_csv_name


def build_features(data_root: Path, patch_size: int, progress_every: int = 100) -> pd.DataFrame:
    rows: list[dict] = []
    seen = 0
    for csv_path in find_train_csvs(data_root):
        target = target_from_csv_name(csv_path)
        value_col = target_column(target)
        df = pd.read_csv(csv_path)
        print(f"reading {csv_path} ({len(df)} rows)")
        for _, row in df.iterrows():
            value = row.get(value_col)
            if pd.isna(value):
                continue
            feats = row_features(csv_path, row, patch_size=patch_size)
            feats["target"] = target
            feats["y"] = float(value)
            rows.append(feats)
            seen += 1
            if progress_every and seen % progress_every == 0:
                print(f"processed {seen} labeled rows")
    if not rows:
        raise RuntimeError(f"No training rows found under {data_root}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    features = build_features(args.data_root, patch_size=args.patch_size, progress_every=args.progress_every)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    print(f"wrote {len(features)} rows x {len(features.columns)} columns to {args.out}")


if __name__ == "__main__":
    main()
