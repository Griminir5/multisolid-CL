"""Check installer payloads, download integrity, and the four-file release contract."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from download_inputs import download, fetch
from prepare_release_assets import prepare
from release_metadata import sha256


class DownloadTests(unittest.TestCase):
    def test_cached_download_must_match_pinned_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "installer.exe"
            path.write_bytes(b"verified input")
            spec = {"url": "https://example.invalid/unused", "sha256": sha256(path)}
            self.assertEqual(download(path.name, spec, root), path)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                download(path.name, spec, root)

    def test_archive_fetch_still_extracts_and_reuses_verified_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "input.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("source/file.txt", "contents")
            spec = {"url": "https://example.invalid/unused", "sha256": sha256(archive)}
            destination = fetch(archive.name, spec, root)
            self.assertEqual((destination / "source/file.txt").read_text(), "contents")
            self.assertEqual(fetch(archive.name, spec, root), destination)

    def test_inno_setup_is_prepared_in_portable_mode_and_cached(self):
        import prepare_inno_setup
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiler = root / "inno-6.7.3/ISCC.exe"
            def install(command, **kwargs):
                self.assertIn("/PORTABLE=1", command)
                self.assertIn("/CURRENTUSER", command)
                self.assertIn("/NORESTART", command)
                compiler.parent.mkdir()
                compiler.write_bytes(b"compiler fixture")
            with patch.object(prepare_inno_setup.sys, "platform", "win32"), \
                    patch.object(prepare_inno_setup, "download", return_value=root / "installer.exe"), \
                    patch.object(prepare_inno_setup.subprocess, "run", side_effect=install) as execute:
                self.assertEqual(prepare_inno_setup.prepare(root), compiler)
                self.assertEqual(prepare_inno_setup.prepare(root), compiler)
                self.assertEqual(execute.call_count, 1)


class ReleaseAssetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        self.artifacts = self.root / "artifacts"
        self.files = []
        for platform, suffixes in (("windows", ("setup.exe", "portable.zip")),
                                   ("linux", ("setup.deb", "portable.tar.gz"))):
            folder = self.artifacts / f"MultiSolid-{platform}"
            folder.mkdir(parents=True)
            lines = []
            for suffix in suffixes:
                path = folder / f"MultiSolid-{self.version}-{platform}-x64-{suffix}"
                path.write_bytes(path.name.encode())
                self.files.append(path)
                lines.append(f"{sha256(path)}  {path.name}\n")
            (folder / "SHA256SUMS.txt").write_text("".join(lines))

    def test_collects_exactly_four_downloads_with_combined_checksums(self):
        output = self.root / "downloads"
        paths = prepare(self.artifacts, output, f"v{self.version}")
        self.assertEqual({p.name for p in paths}, {p.name for p in self.files})
        self.assertEqual(len((output / "SHA256SUMS.txt").read_text().splitlines()), 4)
        self.assertEqual(len(list(output.iterdir())), 5)
        self.assertTrue((self.root / "release-notes.md").is_file())

    def test_missing_installer_prevents_release(self):
        self.files[0].unlink()
        with self.assertRaisesRegex(ValueError, "Expected exactly"):
            prepare(self.artifacts, self.root / "downloads", f"v{self.version}")

    def test_tampered_download_prevents_release(self):
        self.files[0].write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            prepare(self.artifacts, self.root / "downloads", f"v{self.version}")

    def test_wrong_tag_prevents_release(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare(self.artifacts, self.root / "downloads", "v999.0.0")


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("dpkg-deb"), "Linux package tools required")
class LinuxPackageTests(unittest.TestCase):
    def test_installer_and_portable_payloads_preserve_the_validated_bundle(self):
        from check_desktop_release import verify_inventory
        from check_installers import check_portable
        from package_linux import package
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "MultiSolid"
            (bundle / "_internal").mkdir(parents=True)
            executable = bundle / "MultiSolid"
            executable.write_bytes(b"#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            (bundle / "_internal/library.so.1").write_bytes(b"library")
            (bundle / "_internal/library.so").symlink_to("library.so.1")
            release = {"version": "1.0.0rc1", "build_id": "test", "platform": "linux", "minimum_glibc": "2.38"}
            (bundle / "release.json").write_text(json.dumps(release))
            inventory = {p.relative_to(bundle).as_posix(): sha256(p) for p in bundle.rglob("*") if p.is_file()}
            (bundle / "bundle-files.json").write_text(json.dumps(inventory))
            validation = root / "validation.json"
            validation.write_text(json.dumps({"passed": True, "executable_sha256": sha256(executable),
                                              "inventory_sha256": sha256(bundle / "bundle-files.json")}))
            installer, archive = package(bundle, root / "downloads", validation)
            check_portable(archive, root / "Portable α", release)
            extracted = root / "deb-root"
            subprocess.run(["dpkg-deb", "--extract", str(installer), str(extracted)], check=True, capture_output=True)
            verify_inventory(extracted / "opt/multisolid")
            self.assertTrue((extracted / "opt/multisolid/_internal/library.so").is_symlink())
            self.assertTrue(os.access(extracted / "opt/multisolid/MultiSolid", os.X_OK))
            self.assertEqual((extracted / "usr/bin/multisolid").read_text(),
                             '#!/bin/sh\nexec /opt/multisolid/MultiSolid "$@"\n')
            self.assertTrue((extracted / "usr/share/applications/multisolid.desktop").is_file())
            if shutil.which("desktop-file-validate"):
                subprocess.run(["desktop-file-validate", str(extracted / "usr/share/applications/multisolid.desktop")], check=True)
            version = subprocess.check_output(["dpkg-deb", "-f", str(installer), "Version"], text=True).strip()
            self.assertEqual(version, "1.0.0~rc1")
            validation.write_text(json.dumps({"passed": False}))
            with self.assertRaisesRegex(ValueError, "exact Linux bundle"):
                package(bundle, root / "rejected", validation)

    def test_candidate_debian_versions_sort_before_final_release(self):
        from package_linux import debian_version
        for earlier, later in (("1.0.0.dev1", "1.0.0a1"), ("1.0.0rc1", "1.0.0"),
                               ("1.0.0rc1.dev1", "1.0.0rc1"), ("1.0.0", "1.0.0.post1")):
            with self.subTest(earlier=earlier, later=later):
                subprocess.run(["dpkg", "--compare-versions", debian_version(earlier), "lt", debian_version(later)], check=True)


if __name__ == "__main__":
    unittest.main()
