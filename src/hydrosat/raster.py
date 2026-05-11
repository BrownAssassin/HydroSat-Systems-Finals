from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Patch:
    data: np.ndarray
    row: int
    col: int


def _import_rasterio():
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required to read the multi-band GeoTIFF files. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return rasterio, Window


@lru_cache(maxsize=256)
def _open_dataset(path: str) -> Any:
    rasterio, _ = _import_rasterio()
    return rasterio.open(path)


@lru_cache(maxsize=64)
def image_profile(path: str) -> dict:
    src = _open_dataset(path)
    return {
        "count": src.count,
        "height": src.height,
        "width": src.width,
        "crs": str(src.crs),
        "bounds": tuple(src.bounds),
        "transform": tuple(src.transform),
    }


def read_patch(path: Path, lon: float, lat: float, size: int = 32) -> Patch:
    _, Window = _import_rasterio()
    half = size // 2
    src = _open_dataset(str(path))
    row, col = src.index(lon, lat)
    row_off = row - half
    col_off = col - half
    window = Window(col_off=col_off, row_off=row_off, width=size, height=size)
    data = src.read(window=window, boundless=True, fill_value=np.nan).astype("float32")
    data[data <= -9990] = np.nan
    return Patch(data=data, row=row, col=col)
