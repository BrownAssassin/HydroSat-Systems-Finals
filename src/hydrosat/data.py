"""Dataset discovery helpers for the final-round data layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

from .config import (
    CHLA_TARGET,
    DEFAULT_DATA_ROOT,
    SAMPLE_INPUT_ROOT_NAME,
    TARGET_POINT_FILENAMES,
    TRAIN_AREAS,
    TRAIN_ROOT_NAME,
    TURBIDITY_TARGET,
)


@dataclass(frozen=True)
class PointsTableInfo:
    """Summary information for one point-label CSV."""

    target: str
    path: Path
    row_count: int
    unique_filenames: int


@dataclass(frozen=True)
class AreaDataset:
    """Training-area metadata in the normalized layout."""

    name: str
    root: Path
    image_dir: Path
    image_count: int
    turbidity_points: PointsTableInfo | None
    chla_points: PointsTableInfo | None


@dataclass(frozen=True)
class TrainingLayout:
    """Complete training-layout summary."""

    root: Path
    areas: tuple[AreaDataset, ...]

    @property
    def total_turbidity_rows(self) -> int:
        return sum(area.turbidity_points.row_count for area in self.areas if area.turbidity_points)

    @property
    def total_chla_rows(self) -> int:
        return sum(area.chla_points.row_count for area in self.areas if area.chla_points)


@dataclass(frozen=True)
class InputLayout:
    """Competition-style inference input layout."""

    root: Path
    image_dir: Path
    turbidity_points: PointsTableInfo
    chla_points: PointsTableInfo


def get_default_data_root() -> Path:
    """Return the normalized raw-data root."""

    return DEFAULT_DATA_ROOT


def discover_training_layout(data_root: Path | str | None = None) -> TrainingLayout:
    """Discover the normalized training tree under ``data/raw/train``."""

    root = _resolve_data_root(data_root) / TRAIN_ROOT_NAME
    if not root.exists():
        raise FileNotFoundError(f"Training root not found: {root}")

    areas: list[AreaDataset] = []
    for area_name in TRAIN_AREAS:
        area_root = root / area_name
        if not area_root.exists():
            raise FileNotFoundError(f"Expected training area is missing: {area_root}")
        image_dir = area_root / f"{area_name}_images"
        if not image_dir.exists():
            raise FileNotFoundError(f"Expected image directory is missing: {image_dir}")

        turb_path = area_root / f"track2_turb_train_point_{area_name}.csv"
        chla_path = area_root / f"track2_cha_train_point_{area_name}.csv"

        areas.append(
            AreaDataset(
                name=area_name,
                root=area_root,
                image_dir=image_dir,
                image_count=_count_images(image_dir),
                turbidity_points=_summarize_points_table(turb_path, TURBIDITY_TARGET) if turb_path.exists() else None,
                chla_points=_summarize_points_table(chla_path, CHLA_TARGET) if chla_path.exists() else None,
            )
        )

    return TrainingLayout(root=root, areas=tuple(areas))


def discover_input_layout(
    input_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> InputLayout:
    """Discover either the sample input layout or a mounted competition input directory."""

    root = Path(input_root) if input_root is not None else _resolve_data_root(data_root) / SAMPLE_INPUT_ROOT_NAME
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Inference input root not found: {root}")

    image_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.endswith("_images")
    )
    if len(image_dirs) != 1:
        raise ValueError(f"Expected exactly one '*_images' directory under {root}, found {len(image_dirs)}")

    turb_path = root / TARGET_POINT_FILENAMES[TURBIDITY_TARGET]
    chla_path = root / TARGET_POINT_FILENAMES[CHLA_TARGET]
    if not turb_path.exists():
        raise FileNotFoundError(f"Missing turbidity point file: {turb_path}")
    if not chla_path.exists():
        raise FileNotFoundError(f"Missing chlorophyll-a point file: {chla_path}")

    return InputLayout(
        root=root,
        image_dir=image_dirs[0],
        turbidity_points=_summarize_points_table(turb_path, TURBIDITY_TARGET),
        chla_points=_summarize_points_table(chla_path, CHLA_TARGET),
    )


def _resolve_data_root(data_root: Path | str | None) -> Path:
    return (Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT).resolve()


def _count_images(image_dir: Path) -> int:
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".tif")


def _summarize_points_table(path: Path, target: str) -> PointsTableInfo:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    unique_filenames = len({row["filename"] for row in rows if row.get("filename")})
    return PointsTableInfo(
        target=target,
        path=path,
        row_count=len(rows),
        unique_filenames=unique_filenames,
    )
