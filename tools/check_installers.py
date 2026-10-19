"""Check final downloads, real installation/reinstallation, and safe removal."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from uuid import uuid4
import zipfile

from check_desktop_release import check as check_desktop, verify_inventory
from release_metadata import sha256


def run(command, log):
    with log.open("ab") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=stream, check=True, timeout=600)


def check_portable(archive, destination, release):
    destination.mkdir()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            if any(not (destination / name).resolve().is_relative_to(destination) for name in source.namelist()):
                raise ValueError("Unsafe portable ZIP member.")
            source.extractall(destination)
    else:
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")
    bundle = destination / "MultiSolid"
    verify_inventory(bundle)
    if json.loads((bundle / "release.json").read_text(encoding="utf-8")) != release:
        raise ValueError("The portable archive contains another application build.")
    executable = bundle / ("MultiSolid.exe" if sys.platform == "win32" else "MultiSolid")
    if sys.platform != "win32" and not os.access(executable, os.X_OK):
        raise ValueError("The portable executable lost its executable permission.")
    run([executable, "--help"], destination.parent / "portable.log")


def windows_installed():
    import winreg
    key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{E09BF0B1-78BE-4A87-960F-542D9563271B}_is1"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key):
            return True
    except FileNotFoundError:
        return False


def check_windows(installer, destination, release):
    if windows_installed():
        raise RuntimeError("Use a clean test account: MultiSolid is already installed for this user.")
    installation = destination / "Installed application α"
    group = "MultiSolid acceptance " + uuid4().hex
    shortcut = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs" / group / "MultiSolid.lnk"
    bundle = installation / "app" / f"{release['version']}-{release['build_id']}"
    uninstaller = installation / "unins000.exe"
    try:
        for attempt in ("install", "reinstall"):
            run([installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-",
                 f"/DIR={installation}", f"/GROUP={group}", "/TASKS=",
                 f"/LOG={destination / (attempt + '.log')}"], destination / "installer-process.log")
            verify_inventory(bundle)
            if not shortcut.is_file() or not windows_installed():
                raise ValueError("The installer did not register its Start menu shortcut and uninstaller.")
            if json.loads((bundle / "release.json").read_text(encoding="utf-8")) != release:
                raise ValueError("The installer contains another application build.")
        check_desktop(bundle, destination / "installed-check", relocate=False)
    finally:
        if uninstaller.is_file():
            run([uninstaller, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                 f"/LOG={destination / 'uninstall.log'}"], destination / "installer-process.log")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and (bundle.exists() or windows_installed()):
                time.sleep(.2)
    if bundle.exists() or shortcut.exists() or windows_installed():
        raise ValueError("Uninstall left application files, a shortcut, or installation registration behind.")


def check_linux(installer, destination, release, qt_platform):
    query = subprocess.run(["dpkg-query", "-W", "-f=${Status}", "multisolid"], capture_output=True, text=True)
    bundle = Path("/opt/multisolid")
    owned = (bundle, Path("/usr/bin/multisolid"), Path("/usr/share/applications/multisolid.desktop"),
             Path("/usr/share/icons/hicolor/scalable/apps/multisolid.svg"))
    if query.returncode == 0 or any(path.exists() or path.is_symlink() for path in owned):
        raise RuntimeError("Use a clean test machine: MultiSolid is already registered or installation files exist.")
    try:
        for attempt in ("install", "reinstall"):
            run(["sudo", "-n", "dpkg", "--install", installer], destination / (attempt + ".log"))
            verify_inventory(bundle)
            if json.loads((bundle / "release.json").read_text(encoding="utf-8")) != release:
                raise ValueError("The installer contains another application build.")
        run(["desktop-file-validate", "/usr/share/applications/multisolid.desktop"], destination / "desktop-entry.log")
        run(["/usr/bin/multisolid", "--help"], destination / "launcher.log")
        check_desktop(bundle, destination / "installed-check", qt_platform=qt_platform, relocate=False)
    finally:
        # This is only reached after the existing-installation guard above.
        run(["sudo", "-n", "dpkg", "--remove", "multisolid"], destination / "uninstall.log")
    if any(path.exists() or path.is_symlink() for path in owned):
        raise ValueError("Uninstall left application files or launchers behind.")


def check(downloads, bundle, destination, *, system=False, qt_platform=None):
    if sys.platform != "win32" and not system:
        raise ValueError("Linux installer checks require --system on a clean test machine with passwordless sudo.")
    downloads, bundle, destination = (Path(p).resolve() for p in (downloads, bundle, destination))
    destination.mkdir(parents=True, exist_ok=False)
    release = json.loads((bundle / "release.json").read_text(encoding="utf-8"))
    platform = "windows" if sys.platform == "win32" else "linux"
    suffixes = ("setup.exe", "portable.zip") if platform == "windows" else ("setup.deb", "portable.tar.gz")
    artifacts = [downloads / f"MultiSolid-{release['version']}-{platform}-x64-{suffix}" for suffix in suffixes]
    record = {"passed": False, "platform": platform, "release": release, "checks": []}
    sentinel = destination / "User projects" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("User project data must survive installation and removal.\n", encoding="utf-8")
    try:
        checksums = dict(line.split("  ", 1)[::-1] for line in (downloads / "SHA256SUMS.txt").read_text().splitlines())
        for path in artifacts:
            if checksums.get(path.name) != sha256(path):
                raise ValueError(f"Download checksum mismatch: {path.name}")
        record["artifacts"] = {path.name: sha256(path) for path in artifacts}
        check_portable(artifacts[1], destination / "Extracted portable α", release)
        record["checks"].append("portable archive inventory, release identity and launcher")
        if platform == "windows":
            check_windows(artifacts[0], destination, release)
        else:
            check_linux(artifacts[0], destination, release, qt_platform)
        if sentinel.read_text(encoding="utf-8") != "User project data must survive installation and removal.\n":
            raise ValueError("Uninstall changed an external project.")
        if not (destination / "installed-check/self-test/Project with spaces α/project.json").is_file():
            raise ValueError("The real simulation project did not survive uninstall.")
        record["checks"] += ["install and same-version reinstall", "installed desktop acceptance",
                             "uninstall and shortcut removal", "external simulation project preserved"]
        record["passed"] = True
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        (destination / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return destination / "validation.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", action="store_true", help="Linux: install/remove the .deb on this clean test machine using sudo")
    parser.add_argument("--qt-platform", choices=("xcb", "wayland", "offscreen"))
    args = parser.parse_args()
    print(check(args.downloads, args.bundle, args.output, system=args.system, qt_platform=args.qt_platform))
