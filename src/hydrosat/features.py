"""Tabular feature extraction for the final-round baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL_TARGETS, DEFAULT_FEATURES_PATH, DEFAULT_PATCH_SIZE
from .data import PointRecord, load_training_records
from .raster import read_patch

EPS = 1e-6


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def _band(data: np.ndarray, one_based_index: int) -> np.ndarray:
    idx = one_based_index - 1
    if idx >= data.shape[0]:
        return np.full(data.shape[1:], np.nan, dtype="float32")
    return data[idx]


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.isfinite(values)
    if not finite.any():
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }

    clean = values[finite]
    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_std": float(np.std(clean)),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_p10": float(np.percentile(clean, 10)),
        f"{prefix}_p25": float(np.percentile(clean, 25)),
        f"{prefix}_p50": float(np.percentile(clean, 50)),
        f"{prefix}_p75": float(np.percentile(clean, 75)),
        f"{prefix}_p90": float(np.percentile(clean, 90)),
        f"{prefix}_max": float(np.max(clean)),
    }


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, values, np.nan)


def _center_window(data: np.ndarray, radius: int) -> np.ndarray:
    height, width = data.shape[-2:]
    row = height // 2
    col = width // 2
    return data[
        :,
        max(0, row - radius) : min(height, row + radius + 1),
        max(0, col - radius) : min(width, col + radius + 1),
    ]


def spectral_features(data: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Extract spectral, masked, center-window, and texture-like patch features."""

    feats: dict[str, float] = {}
    finite_fraction = np.isfinite(data).mean(axis=(1, 2))
    for idx in range(data.shape[0]):
        band = data[idx]
        name = f"{prefix}b{idx + 1:02d}"
        feats.update(_stats(band, name))
        feats[f"{name}_finite_fraction"] = float(finite_fraction[idx])

    coastal = _band(data, 1)
    blue = _band(data, 2)
    green = _band(data, 3)
    red = _band(data, 4)
    red_edge1 = _band(data, 5)
    red_edge2 = _band(data, 6)
    red_edge3 = _band(data, 7)
    nir = _band(data, 8)
    narrow_nir = _band(data, 9)
    water_vapor = _band(data, 10)
    swir1 = _band(data, 11)
    swir2 = _band(data, 12)

    indices = {
        "ndwi_green_nir": _safe_ratio(green, nir),
        "ndwi_green_narrow_nir": _safe_ratio(green, narrow_nir),
        "mndwi_green_swir": _safe_ratio(green, swir1),
        "mndwi_green_swir2": _safe_ratio(green, swir2),
        "ndti_red_green": _safe_ratio(red, green),
        "ndci_rededge_red": _safe_ratio(red_edge1, red),
        "chlorophyll_rededge": _safe_ratio(nir, red_edge1),
        "ndvi_nir_red": _safe_ratio(nir, red),
        "red_green_ratio": red / (green + EPS),
        "blue_green_ratio": blue / (green + EPS),
        "coastal_blue_ratio": coastal / (blue + EPS),
        "red_blue_ratio": red / (blue + EPS),
        "nir_red_ratio": nir / (red + EPS),
        "rededge_green_ratio": red_edge1 / (green + EPS),
        "rededge2_green_ratio": red_edge2 / (green + EPS),
        "rededge3_green_ratio": red_edge3 / (green + EPS),
        "swir_nir_ratio": swir1 / (nir + EPS),
        "swir2_nir_ratio": swir2 / (nir + EPS),
        "red_swir_ratio": red / (swir1 + EPS),
        "green_swir_ratio": green / (swir1 + EPS),
        "turbidity_simple": red + green,
        "visible_brightness": (blue + green + red) / 3.0,
        "rededge_slope": (red_edge3 - red_edge1) / (red_edge3 + red_edge1 + EPS),
        "aerosol_water_vapor_ratio": coastal / (water_vapor + EPS),
    }

    water = (indices["ndwi_green_nir"] > 0.0) | (indices["mndwi_green_swir"] > 0.0)
    strict_water = (indices["ndwi_green_nir"] > 0.05) & (indices["mndwi_green_swir"] > 0.05)

    for name, values in indices.items():
        feats.update(_stats(values, f"{prefix}{name}"))
        feats.update(_stats(_masked(values, water), f"{prefix}water_{name}"))
        feats.update(_stats(_masked(values, strict_water), f"{prefix}strict_water_{name}"))

    feats[f"{prefix}water_fraction"] = float(np.mean(water))
    feats[f"{prefix}strict_water_fraction"] = float(np.mean(strict_water))
    feats[f"{prefix}valid_fraction"] = float(np.isfinite(data).mean())

    for idx in range(data.shape[0]):
        band = data[idx]
        feats.update(_stats(_masked(band, water), f"{prefix}water_b{idx + 1:02d}"))
        feats.update(_stats(_masked(band, strict_water), f"{prefix}strict_water_b{idx + 1:02d}"))

    for radius in (1, 2, 4):
        center = _center_window(data, radius)
        center_prefix = f"{prefix}center{radius * 2 + 1}_"
        for idx in range(center.shape[0]):
            feats.update(_stats(center[idx], f"{center_prefix}b{idx + 1:02d}"))

    for idx in range(data.shape[0]):
        band = data[idx]
        diffs = []
        if band.shape[0] > 1:
            diffs.append(np.diff(band, axis=0))
        if band.shape[1] > 1:
            diffs.append(np.diff(band, axis=1))
        if diffs:
            grad = np.concatenate([d.reshape(-1) for d in diffs])
            feats.update(_stats(np.abs(grad), f"{prefix}b{idx + 1:02d}_abs_gradient"))
    return feats


def feature_row(record: PointRecord, patch_size: int = DEFAULT_PATCH_SIZE) -> dict[str, float | int | str]:
    """Build one baseline feature row from a point record."""

    patch = read_patch(record.image_path, record.lon, record.lat, size=patch_size)
    date_text = Path(record.filename).stem.split("_")[-1]
    date = datetime.strptime(date_text, "%Y-%m-%d")

    feats: dict[str, float | int | str] = {
        "target": record.target,
        "filename": record.filename,
        "image_path": str(record.image_path),
        "area": record.area,
        "source_csv": str(record.source_csv),
        "lon": float(record.lon),
        "lat": float(record.lat),
        "y": float(record.value) if record.value is not None else np.nan,
        "patch_size": int(patch_size),
        "pixel_row": int(patch.row),
        "pixel_col": int(patch.col),
        "pixel_row_norm": float(patch.row / max(patch.profile.height - 1, 1)),
        "pixel_col_norm": float(patch.col / max(patch.profile.width - 1, 1)),
        "image_height": int(patch.profile.height),
        "image_width": int(patch.profile.width),
        "month": int(date.month),
        "day_of_year": int(date.timetuple().tm_yday),
        "season_sin": float(np.sin(2 * np.pi * date.timetuple().tm_yday / 366.0)),
        "season_cos": float(np.cos(2 * np.pi * date.timetuple().tm_yday / 366.0)),
    }
    feats.update(spectral_features(patch.data))
    return feats


def build_feature_table(
    records: list[PointRecord],
    patch_size: int = DEFAULT_PATCH_SIZE,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Build a feature table for a list of point records."""

    rows: list[dict[str, float | int | str]] = []
    for index, record in enumerate(records, start=1):
        rows.append(feature_row(record, patch_size=patch_size))
        if progress_every and index % progress_every == 0:
            print(f"built features for {index}/{len(records)} point records", flush=True)

    if not rows:
        raise RuntimeError("No point records were provided for feature building.")
    return pd.DataFrame(rows)


def build_training_table(
    data_root: Path | str | None = None,
    targets: tuple[str, ...] | list[str] | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    progress_every: int = 100,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load training records and convert them into one unified feature table."""

    selected_targets = tuple(targets) if targets is not None else ALL_TARGETS
    records = load_training_records(data_root=data_root, targets=selected_targets)
    if limit is not None:
        records = records[:limit]
    return build_feature_table(records, patch_size=patch_size, progress_every=progress_every)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the HydroSat training feature table.")
    parser.add_argument("--data-root", type=Path, default=None, help="Normalized raw-data root. Defaults to data/raw.")
    parser.add_argument("--out", type=Path, default=DEFAULT_FEATURES_PATH, help="Output CSV path for the training feature table.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Square patch size to extract around each point.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N rows while building features.")
    parser.add_argument("--targets", nargs="+", default=list(ALL_TARGETS), help="Targets to include: turbidity and/or chla.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    targets = tuple(args.targets)
    features = build_training_table(
        data_root=args.data_root,
        targets=targets,
        patch_size=args.patch_size,
        progress_every=args.progress_every,
        limit=args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    print(f"wrote {len(features)} rows x {len(features.columns)} columns to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
