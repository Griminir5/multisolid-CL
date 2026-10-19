"""Prepare the pinned Windows installer compiler in the build workspace."""
import json
from pathlib import Path
import subprocess
import sys

from download_inputs import download


def prepare(work):
    if sys.platform != "win32":
        raise RuntimeError("Inno Setup must be prepared on Windows.")
    work = Path(work).resolve()
    name = "innosetup-6.7.3.exe"
    spec = json.loads(Path(__file__).with_name("windows-release-inputs.json").read_text())[name]
    installer = download(name, spec, work / "downloads")
    destination = work / "inno-6.7.3"
    compiler = destination / "ISCC.exe"
    marker = destination / ".installer-sha256"
    if not compiler.is_file() or not marker.is_file() or marker.read_text() != spec["sha256"]:
        # Inno Setup's own portable mode avoids file associations and registration.
        subprocess.run([str(installer), "/PORTABLE=1", "/CURRENTUSER", "/VERYSILENT",
                        "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", "/NOICONS",
                        f"/DIR={destination}", f"/LOG={work / 'inno-setup.log'}"],
                       check=True, timeout=300)
        if not compiler.is_file():
            raise RuntimeError("Inno Setup did not install its compiler; see inno-setup.log.")
        marker.write_text(spec["sha256"], encoding="utf-8")
    return compiler
