"""Exercise the real installer recipe with a tiny payload before the desktop build."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_installers import check_windows
from release_metadata import sha256


@unittest.skipUnless(sys.platform == "win32" and os.environ.get("MULTISOLID_TEST_ISCC"),
                     "Requires Windows and MULTISOLID_TEST_ISCC pointing to Inno Setup")
class WindowsInstallerTests(unittest.TestCase):
    def test_real_installer_registers_custom_group_reinstalls_and_uninstalls(self):
        # CI supplies a persistent location so compiler/setup logs survive failures.
        if os.environ.get("MULTISOLID_INSTALLER_TEST_OUTPUT"):
            root = Path(os.environ["MULTISOLID_INSTALLER_TEST_OUTPUT"]).resolve()
            root.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
        bundle = root / "payload"
        bundle.mkdir()
        # A real PE executable keeps shortcut/icon creation representative. This
        # check never launches the payload; full desktop acceptance runs later.
        shutil.copy2(sys.executable, bundle / "MultiSolid.exe")
        shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE.txt")
        release = {"version": "1.0.0", "build_id": "installer-test"}
        (bundle / "release.json").write_text(json.dumps(release), encoding="utf-8")
        inventory = {p.name: sha256(p) for p in bundle.iterdir()}
        (bundle / "bundle-files.json").write_text(json.dumps(inventory), encoding="utf-8")
        with (root / "compiler.log").open("wb") as log:
            subprocess.run([os.environ["MULTISOLID_TEST_ISCC"], f"/DBundleDir={bundle}",
                            "/DAppVersion=1.0.0", "/DWindowsVersion=1.0.0.0",
                            "/DBuildId=installer-test", f"/O{root}",
                            str(ROOT / "desktop/installer/MultiSolid.iss")],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        external = root / "User project α.txt"
        external.write_text("Keep this project", encoding="utf-8")
        with patch("check_installers.check_desktop") as desktop:
            check_windows(root / "MultiSolid-1.0.0-windows-x64-setup.exe", root, release)
        desktop.assert_called_once()
        self.assertEqual(external.read_text(encoding="utf-8"), "Keep this project")


if __name__ == "__main__":
    unittest.main()
