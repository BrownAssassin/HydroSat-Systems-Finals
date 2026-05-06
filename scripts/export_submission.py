"""Copy the submission-safe repo subset into a separate GitLab working tree."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the submission-safe subset of this repo.")
    parser.add_argument("--dest", required=True, type=Path, help="Destination directory for the exported submission tree.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to submission-manifest.txt at repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    import sys

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from hydrosat.submission import export_submission_tree, load_manifest

    manifest_path = args.manifest or repo_root / "submission-manifest.txt"
    manifest_entries = load_manifest(manifest_path)
    exported = export_submission_tree(repo_root, args.dest, manifest_entries)

    print(f"Exported {len(exported)} manifest entr{'y' if len(exported) == 1 else 'ies'} to {args.dest}")
    for path in exported:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
