from __future__ import annotations

import os
from pathlib import Path


def find_train_csvs(data_root: Path) -> list[Path]:
    return sorted(data_root.rglob("track2_*_train_point_area*.csv"))


def find_image_for_row(csv_path: Path, filename: str) -> Path:
    area_dir = csv_path.parent
    candidates = [area_dir / f"{area_dir.name}_images" / filename]
    image_root = os.environ.get("HYDROSAT_IMAGE_ROOT")
    if image_root:
        candidates.append(Path(image_root) / filename)
    area_prefix = filename.split("_", 1)[0]
    candidates.append(area_dir / f"{area_prefix}_images" / filename)
    candidates.extend(area_dir.glob(f"**/{filename}"))
    for image_path in candidates:
        if image_path.exists():
            return image_path
    checked = ", ".join(str(path) for path in candidates[:2])
    raise FileNotFoundError(f"Missing image for {csv_path.name}: checked {checked}")


def target_from_csv_name(csv_path: Path) -> str:
    name = csv_path.name.lower()
    if "_turb_" in name:
        return "turbidity"
    if "_cha_" in name:
        return "chla"
    raise ValueError(f"Cannot infer target type from {csv_path}")


def target_column(target: str) -> str:
    if target == "turbidity":
        return "turb_value"
    if target == "chla":
        return "cha_value"
    raise ValueError(f"Unknown target: {target}")
