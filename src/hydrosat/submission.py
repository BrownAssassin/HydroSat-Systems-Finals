"""Helpers for exporting a submission-safe mirror of this repository."""

from __future__ import annotations

from pathlib import Path
import shutil

FORBIDDEN_EXPORT_PREFIXES = ("data/raw", "artifacts", ".git")


def load_manifest(manifest_path: Path | str) -> list[str]:
    """Load relative manifest entries from a text file."""

    path = Path(manifest_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    entries = [line for line in lines if line and not line.startswith("#")]
    validate_manifest_entries(entries)
    return entries


def validate_manifest_entries(entries: list[str]) -> None:
    """Validate manifest entries before export."""

    for entry in entries:
        normalized = entry.replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("Manifest entries must not be empty.")
        if Path(normalized).is_absolute():
            raise ValueError(f"Manifest entry must be relative: {entry}")
        if normalized.startswith("../") or "/../" in normalized:
            raise ValueError(f"Manifest entry must stay inside the repo: {entry}")
        for prefix in FORBIDDEN_EXPORT_PREFIXES:
            if normalized == prefix or normalized.startswith(f"{prefix}/"):
                raise ValueError(f"Manifest entry must not include forbidden path '{entry}'.")


def export_submission_tree(
    source_root: Path | str,
    destination_root: Path | str,
    manifest_entries: list[str],
) -> list[Path]:
    """Copy the manifest-defined subset into a separate destination tree."""

    src_root = Path(source_root).resolve()
    dest_root = Path(destination_root).resolve()
    if dest_root == src_root or src_root in dest_root.parents:
        raise ValueError("Destination must be outside the source repository tree.")

    exported: list[Path] = []
    for entry in manifest_entries:
        src = (src_root / entry).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Manifest entry does not exist: {entry}")
        _ensure_within_source_root(src_root, src)

        dest = dest_root / entry
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
        exported.append(dest)
    return exported


def _ensure_within_source_root(source_root: Path, candidate: Path) -> None:
    if candidate != source_root and source_root not in candidate.parents:
        raise ValueError(f"Path escapes the source repository: {candidate}")
