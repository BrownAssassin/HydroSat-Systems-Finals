from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import find_image_for_row
from .raster import image_profile, read_patch


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
    return data[:, max(0, row - radius) : min(height, row + radius + 1), max(0, col - radius) : min(width, col + radius + 1)]


def spectral_features(data: np.ndarray, prefix: str = "") -> dict[str, float]:
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
        "swir_nir_ratio": swir1 / (nir + EPS),
        "swir2_nir_ratio": swir2 / (nir + EPS),
        "red_swir_ratio": red / (swir1 + EPS),
        "green_swir_ratio": green / (swir1 + EPS),
        "turbidity_simple": red + green,
        "visible_brightness": (blue + green + red) / 3.0,
        "rededge_slope": (red_edge3 - red_edge1) / (red_edge3 + red_edge1 + EPS),
        "aerosol_water_vapor_ratio": coastal / (water_vapor + EPS),
    }

    water = (indices["ndwi_green_nir"] > 0) | (indices["mndwi_green_swir"] > 0)
    strict_water = (indices["ndwi_green_nir"] > 0.05) & (indices["mndwi_green_swir"] > 0.05)

    for name, values in indices.items():
        feats.update(_stats(values, f"{prefix}{name}"))
        feats.update(_stats(_masked(values, water), f"{prefix}water_{name}"))

    feats[f"{prefix}water_fraction"] = float(np.nanmean(water))
    feats[f"{prefix}strict_water_fraction"] = float(np.nanmean(strict_water))
    feats[f"{prefix}valid_fraction"] = float(np.isfinite(data).mean())

    for idx in range(data.shape[0]):
        band = data[idx]
        name = f"{prefix}water_b{idx + 1:02d}"
        feats.update(_stats(_masked(band, water), name))

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


def row_features(csv_path: Path, row: pd.Series, patch_size: int) -> dict[str, float | int | str]:
    image_path = find_image_for_row(csv_path, str(row["filename"]))
    patch = read_patch(image_path, float(row["Lon"]), float(row["Lat"]), size=patch_size)
    profile = image_profile(str(image_path))
    date = pd.to_datetime(Path(str(row["filename"])).stem.split("_")[-1])
    feats: dict[str, float | int | str] = {
        "filename": str(row["filename"]),
        "image_path": str(image_path),
        "area": csv_path.parent.name,
        "lon": float(row["Lon"]),
        "lat": float(row["Lat"]),
        "pixel_row": int(patch.row),
        "pixel_col": int(patch.col),
        "patch_size": int(patch_size),
        "pixel_row_norm": float(patch.row / max(int(profile["height"]) - 1, 1)),
        "pixel_col_norm": float(patch.col / max(int(profile["width"]) - 1, 1)),
        "month": int(date.month),
        "day_of_year": int(date.dayofyear),
        "season_sin": float(np.sin(2 * np.pi * date.dayofyear / 366.0)),
        "season_cos": float(np.cos(2 * np.pi * date.dayofyear / 366.0)),
    }
    feats.update(spectral_features(patch.data))
    return feats
