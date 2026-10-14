"""Validate a relocated Windows or Linux desktop with developer paths removed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from release_metadata import sha256


def verify_inventory(bundle):
    bundle = Path(bundle).resolve()
    inventory = json.loads((bundle / "bundle-files.json").read_text())
    for path in bundle.rglob("*"):
        if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(bundle)):
            raise ValueError(f"Broken or external bundle symlink: {path.relative_to(bundle)}")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(inventory) | {"bundle-files.json"}:
        raise ValueError("The bundle has missing or unrecorded files; rebuild and validate it.")
    for name, expected in inventory.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or sha256(path) != expected:
            raise ValueError(f"Bundle inventory mismatch: {name}")


def check(bundle, destination, *, qt_platform=None):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    verify_inventory(bundle)
    destination.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("PYTHON", "MULTISOLID", "ZIG_", "QT_", "PYSIDE")) or key in {
            "CXX", "CC", "LIB", "INCLUDE", "LIBPATH", "VIRTUAL_ENV", "GVBINDIR", "FONTCONFIG_FILE",
            "LD_LIBRARY_PATH", "LD_PRELOAD", "MPLCONFIGDIR"}:
            environment.pop(key, None)
    if sys.platform == "win32":
        environment["PATH"] = os.pathsep.join((os.environ["SystemRoot"] + r"\System32", os.environ["SystemRoot"]))
        isolated = ("APPDATA", "LOCALAPPDATA", "TEMP", "TMP")
        name = "MultiSolid.exe"
    else:
        empty_path = destination / "empty-path"
        empty_path.mkdir()
        environment["PATH"] = str(empty_path)
        isolated = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "TMPDIR")
        name = "MultiSolid"
    for key in isolated:
        path = destination / key.lower()
        path.mkdir()
        environment[key] = str(path)
    if qt_platform:
        environment["QT_QPA_PLATFORM"] = qt_platform
    environment["MULTISOLID_PLUGIN_APPROVALS"] = str(destination / "approvals.json")
    record = {"bundle": str(bundle), "executable_sha256": sha256(bundle / name),
              "inventory_sha256": sha256(bundle / "bundle-files.json"),
              "environment": "Relocated; no developer compiler, Python, Graphviz or library search paths",
              "qt_platform": qt_platform or "desktop default", "clean_vm": False, "passed": False}
    relocated = destination / "Relocated application α" / "MultiSolid"
    try:
        shutil.copytree(bundle, relocated, symlinks=True)
        executable = relocated / name
        diagnostic_script = destination / "diagnostics.py"
        shutil.copy2(Path(__file__).with_name("release_check.py"), diagnostic_script)
        with (destination / "launcher.log").open("wb") as log:
            subprocess.run([str(executable), "--self-test", str(destination / "self-test"),
                            "--diagnostics-script", str(diagnostic_script)],
                           cwd=destination, env=environment, stdout=log, stderr=log, timeout=1800, check=True)
            result = json.loads((destination / "self-test/result.json").read_text())
            if result.get("passed") is not True or result.get("frozen") is not True:
                raise RuntimeError("The frozen application self-test did not pass.")
            plugin = destination / "plugin-check-input"
            shutil.copytree(Path(__file__).resolve().parents[1] / "examples/plugins/enthalpy_overrides", plugin,
                            ignore=shutil.ignore_patterns("__pycache__"))
            subprocess.run([str(executable), "--check-plugin", str(plugin)], cwd=destination,
                           env=environment, stdout=log, stderr=log, timeout=120, check=True)
        verify_inventory(relocated)
        verify_inventory(bundle)
        record.update(passed=True, checks=["relocated application and unchanged inventory", *result["checks"],
                                          "external Python plugin in a separate process"])
    finally:
        (destination / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return destination / "validation.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qt-platform", choices=("xcb", "wayland", "offscreen"),
                        help="Use xcb under xvfb-run for headless Linux GUI validation")
    args = parser.parse_args()
    print(check(args.bundle, args.output, qt_platform=args.qt_platform))


if __name__ == "__main__":
    main()
