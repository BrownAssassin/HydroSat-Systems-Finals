from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrosat.submission import FORBIDDEN_EXPORT_PREFIXES, export_submission_tree, load_manifest


class ExportSubmissionTest(unittest.TestCase):
    def test_manifest_excludes_forbidden_prefixes(self) -> None:
        manifest_entries = load_manifest(REPO_ROOT / "submission-manifest.txt")
        for entry in manifest_entries:
            normalized = entry.replace("\\", "/").strip("/")
            for prefix in FORBIDDEN_EXPORT_PREFIXES:
                self.assertFalse(
                    normalized == prefix or normalized.startswith(f"{prefix}/"),
                    msg=f"Manifest entry should not include forbidden prefix '{prefix}': {entry}",
                )

    def test_export_copies_only_manifest_subset(self) -> None:
        manifest_entries = load_manifest(REPO_ROOT / "submission-manifest.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "submission"
            export_submission_tree(REPO_ROOT, dest, manifest_entries)

            exported_top_level = sorted(path.name for path in dest.iterdir())
            self.assertEqual(
                exported_top_level,
                sorted(
                    [
                        ".gitattributes",
                        "Dockerfile",
                        "README.md",
                        "models",
                        "pyproject.toml",
                        "requirements.txt",
                        "run.sh",
                        "src",
                    ]
                ),
            )
            self.assertFalse((dest / "data").exists())
            self.assertFalse((dest / "artifacts").exists())
            local_models = sorted(path.name for path in (REPO_ROOT / "models").iterdir() if path.is_file())
            exported_models = sorted(path.name for path in (dest / "models").iterdir() if path.is_file())
            self.assertEqual(exported_models, local_models)
