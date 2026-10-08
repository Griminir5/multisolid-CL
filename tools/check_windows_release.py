"""Exercise a frozen Windows bundle with developer discovery paths removed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil

from release_metadata import sha256


def check(bundle, destination):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("PYTHON", "MULTISOLID", "ZIG_", "QT_", "PYSIDE")) or key in {
            "CXX", "CC", "LIB", "INCLUDE", "LIBPATH", "VIRTUAL_ENV", "GVBINDIR", "FONTCONFIG_FILE"}:
            environment.pop(key, None)
    environment["PATH"] = os.pathsep.join((os.environ["SystemRoot"] + r"\System32", os.environ["SystemRoot"]))
    for key in ("APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
        path = destination / key.lower()
        path.mkdir()
        environment[key] = str(path)
    environment["MULTISOLID_PLUGIN_APPROVALS"] = str(destination / "approvals.json")
    executable = bundle / "MultiSolid.exe"
    record = {"bundle": str(bundle), "executable_sha256": sha256(executable),
              "inventory_sha256": sha256(bundle / "bundle-files.json"),
              "environment": "Windows PATH only; isolated app data, temporary files and approvals",
              "clean_vm": False, "passed": False}
    try:
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
        record.update(passed=True, checks=result["checks"] + ["external Python plugin in a separate process"])
    finally:
        (destination / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return destination / "validation.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(check(args.bundle, args.output))
