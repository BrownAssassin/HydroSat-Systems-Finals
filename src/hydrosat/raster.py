"""Raster access and patch extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RasterProfile:
    """Raster dimensions and CRS metadata used by the feature pipeline."""

    path: Path
    band_count: int
    height: int
    width: int
    crs: str | None


@dataclass(frozen=True)
class Patch:
    """A centered patch extracted from a georeferenced raster."""

    data: np.ndarray
    row: int
    col: int
    profile: RasterProfile


def _import_rasterio():
    try:
        import rasterio
        from rasterio.windows import Window
        from rasterio.warp import transform
    except ImportError as exc:  # pragma: no cover - exercised in environment smoke checks
        raise RuntimeError(
            "rasterio is required for GeoTIFF feature extraction. "
            "Install the project dependencies in the Python 3.12 environment first."
        ) from exc
    return rasterio, Window, transform


@lru_cache(maxsize=256)
def _open_dataset(path: str) -> Any:
    rasterio, _, _ = _import_rasterio()
    return rasterio.open(path)


@lru_cache(maxsize=256)
def image_profile(path: str | Path) -> RasterProfile:
    """Return cached raster metadata for one image."""

    src = _open_dataset(str(path))
    return RasterProfile(
        path=Path(path),
        band_count=int(src.count),
        height=int(src.height),
        width=int(src.width),
        crs=str(src.crs) if src.crs else None,
    )


def lonlat_to_pixel(path: str | Path, lon: float, lat: float) -> tuple[int, int]:
    """Convert WGS84 lon/lat coordinates to integer pixel indices."""

    rasterio, _, transform = _import_rasterio()
    src = _open_dataset(str(path))

    x = float(lon)
    y = float(lat)
    if src.crs and src.crs.to_epsg() != 4326:
        x_values, y_values = transform("EPSG:4326", src.crs, [x], [y])
        x = float(x_values[0])
        y = float(y_values[0])

    row, col = src.index(x, y)
    return int(row), int(col)


def read_patch(path: str | Path, lon: float, lat: float, size: int) -> Patch:
    """Read a boundless square patch centered on one georeferenced point."""

    _, Window, _ = _import_rasterio()
    src = _open_dataset(str(path))
    profile = image_profile(path)
    row, col = lonlat_to_pixel(path, lon, lat)

    half = size // 2
    row_off = row - half
    col_off = col - half
    window = Window(col_off=col_off, row_off=row_off, width=size, height=size)

    data = src.read(window=window, boundless=True, fill_value=np.nan).astype("float32")
    data[data <= -9990] = np.nan

    return Patch(data=data, row=row, col=col, profile=profile)
