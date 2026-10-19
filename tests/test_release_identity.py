"""Regression coverage for identity outside a checkout and retained result exports."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))

from packed_bed import release


IDENTITY = {"schema_version": 1, "engine_version": "1.0.0", "ui_version": "1.0.0",
            "source_commit": "a" * 40, "source_dirty": False, "build_id": "0123456789ab"}


class ReleaseIdentityTests(unittest.TestCase):
    def test_release_inputs_match(self):
        identity = release.source_identity(ROOT)
        self.assertEqual(identity["engine_version"], identity["ui_version"])
        self.assertEqual(identity["schema_version"], release.IDENTITY_SCHEMA_VERSION)

    def test_source_versions_and_dirty_state(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "desktop").mkdir()
            (root / "pyproject.toml").write_text('[project]\nname="multisolid-cl"\nversion="1.0.0"\n')
            desktop = root / "desktop/pyproject.toml"
            desktop.write_text('[project]\nname="multisolid-cl-ui"\nversion="1.0.0"\n'
                               'dependencies=["multisolid-cl[compiled]==1.0.0"]\n')
            for command in (("init",), ("add", "."),
                            ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")):
                subprocess.run(["git", "-C", str(root), *command], check=True, capture_output=True)
            clean = release.source_identity(root)
            self.assertEqual(len(clean["source_commit"]), 40)
            self.assertFalse(clean["source_dirty"])
            desktop.write_text(desktop.read_text() + "# changed\n")
            dirty = release.source_identity(root)
            self.assertTrue(dirty["source_dirty"])
            self.assertEqual(dirty["source_commit"], clean["source_commit"])
            desktop.write_text(desktop.read_text().replace('==1.0.0', '==0.2.0'))
            with self.assertRaisesRegex(ValueError, "dependency"):
                release.source_identity(root)
            desktop.write_text(desktop.read_text().replace('version="1.0.0"', 'version="0.2.0"'))
            with self.assertRaisesRegex(ValueError, "versions must match"):
                release.source_identity(root)

    def test_installed_wheels_use_stamps_without_git_or_ui(self):
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "site-packages/packed_bed"
            package.mkdir(parents=True)
            (package / release.BUILD_IDENTITY_FILE).write_text(json.dumps({**IDENTITY, "build_id": None}))
            with patch.object(release, "__file__", str(package / "release.py")), \
                    patch.object(release.util, "find_spec", return_value=None), \
                    patch.object(release.metadata, "version", return_value="1.0.0"), \
                    patch.object(release.subprocess, "run", side_effect=AssertionError("Git must not run")):
                self.assertEqual(release.application_identity(), {**IDENTITY, "build_id": None, "ui_version": None})

    def test_unstamped_installation_reports_unknown_commit(self):
        with TemporaryDirectory() as temporary:
            with patch.object(release, "__file__", str(Path(temporary) / "packed_bed/release.py")), \
                    patch.object(release.util, "find_spec", return_value=None), \
                    patch.object(release.metadata, "version", return_value="1.0.0"), \
                    patch.object(release.subprocess, "run", side_effect=AssertionError("Git must not run")):
                identity = release.application_identity()
                self.assertIsNone(identity["source_commit"])
                self.assertIsNone(identity["source_dirty"])

    def test_relocated_desktop_uses_release_record_without_git(self):
        with TemporaryDirectory() as temporary:
            original = Path(temporary) / "original"
            original.mkdir()
            (original / "release.json").write_text(json.dumps({**IDENTITY, "version": "1.0.0"}))
            relocated = Path(temporary) / "relocated application α"
            shutil.move(original, relocated)
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "executable", str(relocated / "MultiSolid.exe")), \
                    patch.object(release.subprocess, "run", side_effect=AssertionError("Git must not run")):
                self.assertEqual(release.application_identity(), IDENTITY)
                (relocated / "release.json").write_text("broken JSON")
                self.assertIsNone(release.application_identity()["source_commit"])
                self.assertIsNone(release.application_identity()["build_id"])

    def test_windows_numeric_version_keeps_candidate_labels_out(self):
        from tools.windows_resources import numeric_version
        for text, expected in (("1.0.0", (1, 0, 0, 0)), ("1.0.0rc1", (1, 0, 0, 0)),
                               ("1.2.dev3", (1, 2, 0, 0)), ("1.2.3.post4", (1, 2, 3, 0)),
                               ("1.2.3+local", (1, 2, 3, 0)), ("1.2.3.4", (1, 2, 3, 4))):
            with self.subTest(version=text):
                self.assertEqual(numeric_version(text), expected)
        for text in ("1.2.3.4.5", "65536.0", "invalid"):
            with self.subTest(version=text), self.assertRaises(ValueError):
                numeric_version(text)

    def test_freezer_rejects_a_wheel_from_another_commit(self):
        from tools.release_metadata import write_metadata
        expected = release.source_identity(ROOT)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for package in ("packed_bed", "packed_bed_ui"):
                (root / package).mkdir()
                stamp = dict(expected)
                if package == "packed_bed_ui":
                    stamp["source_commit"] = "b" * 40
                (root / package / release.BUILD_IDENTITY_FILE).write_text(json.dumps(stamp))
            distribution = SimpleNamespace(locate_file=lambda relative: root / relative)
            with patch.object(release, "application_identity", return_value=expected), \
                    patch("tools.release_metadata.metadata.distribution", return_value=distribution):
                with self.assertRaisesRegex(ValueError, "multisolid-cl-ui.*source identity"):
                    write_metadata(root, ROOT)

    def test_success_and_failure_manifests_and_retained_export(self):
        from packed_bed.config import load_case
        from packed_bed.reports import RunResult, write_run_manifest
        from packed_bed_ui.project import read_documents, write_documents, write_json
        from packed_bed_ui.workbook import run_information, information_rows
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_documents(folder / "inputs", read_documents(ROOT / "examples/simulations/default_case"))
            case = load_case(folder / "inputs/run.yaml")
            output = folder / "output"
            output.mkdir()
            result = RunResult(case, output, application=dict(IDENTITY))
            for status in ("success", "failed"):
                path = write_run_manifest(replace(result, status=status),
                                          failure_stage="simulation" if status == "failed" else None)
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest["application"], IDENTITY)
                self.assertEqual(manifest["schema_version"], release.MANIFEST_SCHEMA_VERSION)
                self.assertEqual(manifest["environment"]["git"], {"commit": "a" * 40, "dirty": False})
                self.assertEqual(manifest["environment"]["packages"]["multisolid-cl"], "1.0.0")
                self.assertEqual(manifest["environment"]["packages"]["multisolid-cl-ui"], "1.0.0")
            write_json(folder / "snapshot.json", {"case_id": "case", "attempt_id": "attempt", "application": IDENTITY})
            with patch.object(release, "application_identity", return_value={**IDENTITY, "engine_version": "2.0.0"}):
                information = run_information(folder)
                self.assertEqual(information["Run"]["application"], IDENTITY)
                self.assertEqual(information["Manifest"]["application"], IDENTITY)
                rows = list(information_rows(information, [], {"axes": {}}))
                self.assertIn(["Manifest", "application.build_id", IDENTITY["build_id"]], rows)

    def test_snapshot_and_about_include_identity(self):
        from packed_bed_ui.project import Project, read_documents, read_json
        from packed_bed_ui.help import diagnostic_information
        with TemporaryDirectory() as temporary:
            project = Project.create(Path(temporary) / "Project", "Identity test")
            documents = read_documents(ROOT / "examples/simulations/default_case")
            # These tests also run before CI has built the compiled runtime.
            documents["run"]["solver"] = {"backend": "daetools", "name": "superlu", "threads": 1,
                                          "relative_tolerance": 1e-7}
            case = project.add_case("Example", documents)
            with patch("packed_bed_ui.project.application_identity", return_value=dict(IDENTITY)):
                pending = case.prepare("identity-test")
            self.assertEqual(read_json(pending / "snapshot.json")["application"], IDENTITY)
            with patch("packed_bed_ui.help.application_identity", return_value=dict(IDENTITY)):
                diagnostics = diagnostic_information()
            for key in ("engine_version", "ui_version", "source_commit", "build_id"):
                self.assertIn(IDENTITY[key], diagnostics)


if __name__ == "__main__":
    unittest.main()
